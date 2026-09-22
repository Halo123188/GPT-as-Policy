"""In-simulator tactile recorder for the dual ARX X5 (import only inside a running Isaac Sim app).

The contact sensor and GelSight (virtual gel) are observation-only: contact reporting is enabled
on the robot links, but no collision shape, material, mass or solver setting changes, so the
rollout physics is the same as without sensing. GelSight real (``mounted=True``) replaces the
virtual gel with a flat compliant pad and a depth camera, and does change the fingertip contact;
see mounted.py.

Frames (per finger link, from Assets/Robots/x5/ARX.usd): +x runs from the finger root to the
tip, z spans the finger width, and the gripping face points along -y on link7 and +y on link8.
Pad images use u = x (fingertip at the top row) and v = z for both fingers, so both pads of a
gripper are drawn in the same gripper-fixed orientation.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import geometry as geo
from .mounted import MOUNT, MountedGel, mount_gel_pads

FINGERS = (('link7', -1.0), ('link8', 1.0))  # link, sign of the gripping-face normal along local y
# GelSight R1.5 calibration (the only TacSL calibration published for Isaac Sim 5.1): 320x240 px,
# 0.0877 mm/px -> a 28.1 x 21.0 mm window on the fingertip, covering the flat tip face.
# Only the image a real GelSight produces is simulated: TacSL's penalty force field is not.
GEL = dict(tip_u=0.071, v_center=-0.0012, rows=320, cols=240, mm_per_pixel=0.0877,
           calibration='gelsight_r15_data')
VIRTUAL = dict(thickness=0.001, margin=0.003)  # GelSight: a 1 mm gel layer that never touches physics
HEIGHT_POOL = 4  # stored height maps are 4x4 max-pooled (80 x 60)


def enable_contact_reporting():
    """Spawn the X5 with PhysX contact reporting on its links (before the environment exists)."""
    import env.robot_manager.robot_config.x5 as x5
    if getattr(x5.get_robot_config, 'tactile_contact_reporting', False):
        return
    original = x5.get_robot_config

    def with_contact_reporting():
        cfg = original()
        return cfg.replace(spawn=cfg.spawn.replace(activate_contact_sensors=True))

    with_contact_reporting.tactile_contact_reporting = True
    x5.get_robot_config = with_contact_reporting


def mount_gelsight():
    """GelSight real: spawn the X5 with gel pads and tactile cameras (before the environment exists)."""
    mount_gel_pads(GEL, FINGERS)


def _matrix(xform_cache, prim):
    return np.array(xform_cache.GetLocalToWorldTransform(prim), np.float64).T


def _collision_points(stage, root_path):
    """Collision geometry points of every enabled collider under a rigid body, in its frame."""
    from pxr import UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    cache = UsdGeom.XformCache()
    inverse_root = np.linalg.inv(_matrix(cache, root))
    pieces = []
    for prim in [p for p in iter_descendants(root) if p.HasAPI(UsdPhysics.CollisionAPI)]:
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        relative = inverse_root @ _matrix(cache, prim)
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), np.float64)
        elif prim.IsA(UsdGeom.Cube):
            points = geo.box_points([UsdGeom.Cube(prim).GetSizeAttr().Get()] * 3)
        else:
            # Collision on an Xform: its first mesh descendant carries the geometry.
            meshes = [m for m in iter_descendants(prim) if m.IsA(UsdGeom.Mesh)]
            if not meshes:
                continue
            relative = inverse_root @ _matrix(cache, meshes[0])
            points = np.asarray(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), np.float64)
        pieces.append((str(prim.GetPath()), geo.apply_transform(relative, points)))
    if not pieces:
        raise RuntimeError(f'No enabled collider under {root_path}')
    return pieces


def iter_descendants(prim):
    from pxr import Usd
    return iter(Usd.PrimRange(prim))


def _numpy(tensor):
    """Detached copy: on the CPU physics backend ``.cpu().numpy()`` would alias live sensor buffers."""
    return tensor.detach().cpu().numpy().copy()


def _pose(view):
    """World position (N, 3) and wxyz quaternion (N, 4)."""
    transform = view.get_transforms()
    return transform[:, :3], transform[:, [6, 3, 4, 5]]


class TactileRecorder:
    def __init__(self, env, output, calibration_root, mounted=False):
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import get_current_stage
        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import GelsightRender
        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

        started = time.monotonic()
        self.env, self.output = env, Path(output)/'tactile'
        self.output.mkdir(parents=True, exist_ok=True)
        sim = env.sim
        self.device = torch.device(sim.device)
        self.physics_dt = float(sim.physics_dt)
        self.substeps = max(1, round(1 / env.obs_manager.collect_freq / self.physics_dt))
        stage = get_current_stage()
        physics_view = SimulationManager.get_physics_sim_view()

        # Task blocks, ordered by their layout label (block0..block7).
        objects = env.scene_manager._rigid_and_dynamic_objects[0].values()
        blocks = []
        for obj in objects:
            label = str((getattr(obj, 'instance_config', None) or {}).get('label') or obj.instance_name)
            if label.startswith('block'):
                blocks.append((label, obj.usd_prim_path))
        if not blocks:
            raise RuntimeError('No task blocks found for tactile filtering')
        blocks.sort(key=lambda item: int(''.join(c for c in item[0] if c.isdigit()) or 0))
        self.block_labels = [label for label, _ in blocks]
        self.block_paths = [path for _, path in blocks]
        self.block_views = [physics_view.create_rigid_body_view(path) for path in self.block_paths]
        pieces, self.block_pieces = [], []
        for index, path in enumerate(self.block_paths):
            for collider, points in _collision_points(stage, path):
                normals, offsets = geo.hull_planes(points)
                pieces.append((index, normals, offsets, points))
                self.block_pieces.append(dict(block=self.block_labels[index], collider=collider,
                                              faces=len(normals)))
        self.blocks_convex = geo.ConvexSet.from_pieces(pieces, self.device)

        # Fingers: left arm first, link7 before link8.
        self.fingers = []
        arms = sorted(enumerate(env.robot_manager.robot_list),
                      key=lambda item: 0 if item[1].arm_name.startswith('left') else 1)
        for robot_index, robot in arms:
            side = robot.arm_name.split('_')[0]
            for link, sign in FINGERS:
                path = f'/World/envs/env_0/robot{robot_index}/{link}'
                if not stage.GetPrimAtPath(path).IsValid():
                    raise RuntimeError(f'Missing finger link prim {path}')
                cfg = ContactSensorCfg(
                    prim_path=f'/World/envs/env_.*/robot{robot_index}/{link}',
                    filter_prim_paths_expr=list(self.block_paths), history_length=self.substeps,
                    track_pose=True, track_contact_points=True, track_friction_forces=True,
                    max_contact_data_count_per_prim=128)
                sensor = ContactSensor(cfg)
                # Created after the timeline started playing: initialize by hand, as TacSL does for
                # its camera, then let InteractiveScene.update refresh it every physics step.
                sensor._initialize_impl()
                sensor._is_initialized = True
                name = f'{side}_{link}'
                sim.scene._sensors[f'tactile_{name}'] = sensor
                hull_points = np.concatenate([p for _, p in _collision_points(stage, path)])
                normals, offsets = geo.hull_planes(hull_points)
                self.fingers.append(dict(name=name, side=side, link=link, sign=sign, path=path,
                                         sensor=sensor, hull=(normals, offsets), hull_points=hull_points,
                                         inward=torch.tensor([0.0, sign, 0.0], device=self.device)))

        rows, cols, pixel = GEL['rows'], GEL['cols'], GEL['mm_per_pixel'] / 1000
        self.gel_u_range = (GEL['tip_u'] - rows * pixel, GEL['tip_u'])
        self.gel_v_range = (GEL['v_center'] - cols * pixel / 2, GEL['v_center'] + cols * pixel / 2)
        if mounted:
            self.mounted = MountedGel([f['path'] for f in self.fingers], GEL, self.device, sim.sim.render)
            gel = {**GEL, **MOUNT, 'kind': 'mounted'}
            self.gel_mask = np.ones((len(self.fingers), rows, cols), bool)
        else:
            self.mounted = None
            self._build_virtual_gel()
            gel = {**GEL, **VIRTUAL, 'kind': 'virtual', 'coverage': self.gel_coverage}
        cfg = GelSightRenderCfg(base_data_path=str(calibration_root), sensor_data_dir_name=GEL['calibration'],
            background_path='bg.jpg', calib_path='polycalib.npz', real_background='real_bg.npy',
            image_height=rows, image_width=cols, num_bins=120, mm_per_pixel=GEL['mm_per_pixel'])
        self.gel_renderer = GelsightRender(cfg, device=self.device)
        self.gel_nominal = _numpy(self.gel_renderer.render(torch.zeros(1, rows, cols, device=self.device))[0])
        self.rows, self.writer, self.closed = [], None, False
        self.meta = dict(schema='robodojo_tactile.v1', physics_dt=self.physics_dt, substeps=self.substeps,
            device=str(self.device), fingers=[dict(name=f['name'], side=f['side'], link=f['link'],
                prim_path=f['path'], gripping_face_normal_local=[0.0, f['sign'], 0.0]) for f in self.fingers],
            blocks=dict(zip(self.block_labels, self.block_paths)), block_colliders=self.block_pieces,
            gel=dict(gel, u_range=list(self.gel_u_range), v_range=list(self.gel_v_range),
                     calibration_root=str(calibration_root)),
            observation_only=not mounted, init_seconds=time.monotonic()-started)
        (self.output/'meta.json').write_text(json.dumps(self.meta, indent=2))

    # ------------------------------------------------------------------ setup
    def _build_virtual_gel(self):
        rows, cols, pixel = GEL['rows'], GEL['cols'], GEL['mm_per_pixel'] / 1000
        u, v = geo.pad_grid(self.gel_u_range, self.gel_v_range, rows, cols)
        self.gel_coverage = {}
        for finger in self.fingers:
            sign = finger['sign']
            inward = np.array([0.0, sign, 0.0])
            # Cast from well outside the gripping face back onto the finger hull: the gel is a
            # uniform layer glued to that surface. The tip tapers, so pixels past the hull edge
            # carry no gel and are masked out.
            far = np.stack([u, np.full_like(u, sign * 0.1), v], -1).reshape(-1, 3)
            normals, offsets = (torch.as_tensor(x, dtype=torch.float32) for x in finger['hull'])
            t, _ = geo.ray_entry(normals[None], offsets[None], torch.as_tensor(far, dtype=torch.float32)[None],
                                 torch.as_tensor(-inward, dtype=torch.float32).expand(1, len(far), 3))
            t = t[0].numpy()
            hit = np.isfinite(t)
            tip_plane = sign * np.abs(finger['hull_points'][:, 1]).max()
            base = far.copy()
            base[hit, 1] = far[hit, 1] - sign * t[hit]
            base[~hit, 1] = tip_plane
            self.gel_coverage[finger['name']] = float(hit.mean())
            finger['gel_base'] = torch.as_tensor(base, dtype=torch.float32, device=self.device)  # (R, 3)
            finger['gel_mask'] = torch.as_tensor(hit, device=self.device)
        self.gel_mask = np.stack([_numpy(f['gel_mask'].reshape(rows, cols)) for f in self.fingers])
        self.gel_centre = torch.as_tensor([np.mean(self.gel_u_range), 0.0, GEL['v_center']],
                                          dtype=torch.float32, device=self.device)
        self.gel_reach = float(np.hypot(rows * pixel, cols * pixel) / 2 + VIRTUAL['thickness'] + VIRTUAL['margin']
                               + 0.03)

    # ------------------------------------------------------------------ recording
    def record(self, step_id):
        started = time.monotonic()
        finger_state = [_pose(f['sensor'].body_physx_view) for f in self.fingers]
        block_state = [_pose(view) for view in self.block_views]
        block_pos = torch.cat([s[0] for s in block_state]).to(self.device)
        block_quat = torch.cat([s[1] for s in block_state]).to(self.device)
        row = dict(step=step_id, contact=[], finger_pose=[],
                   block_pose=_numpy(torch.cat([block_pos, block_quat], -1)))
        heights = []
        for finger, (pos, quat) in zip(self.fingers, finger_state):
            pos, quat = pos.to(self.device), quat.to(self.device)
            row['finger_pose'].append(_numpy(torch.cat([pos[0], quat[0]])))
            row['contact'].append(self._contact(finger, quat[0]))
            if self.mounted is None:
                heights.append(self._virtual_gel(finger, pos[0], quat[0], block_pos, block_quat))
        height = torch.stack(heights) if self.mounted is None else self.mounted.height()
        row['gel_depth'] = _numpy(height.flatten(1).max(1).values).tolist()
        rgb = _numpy(self.gel_renderer.render(height))
        self._write_gel_frame(rgb)
        pooled = torch.nn.functional.max_pool2d(height[:, None], HEIGHT_POOL)[:, 0]
        row['gel_height'] = _numpy(pooled).astype(np.float16)
        row['seconds'] = time.monotonic() - started
        self.rows.append(row)

    def _contact(self, finger, quat):
        """Contact sensor: IsaacLab ContactSensor readings for one finger."""
        data = finger['sensor'].data
        net = data.net_forces_w[0, 0]
        per_block = data.force_matrix_w[0, 0]                       # (B, 3) force on the finger
        per_block_history = data.force_matrix_w_history[0, :, 0]    # (substeps, B, 3)
        friction = data.friction_forces_w[0, 0]
        points = data.contact_pos_w[0, 0]
        outward = geo.quat_rotate(quat, -finger['inward'])          # block pushes the finger this way
        u_axis = geo.quat_rotate(quat, torch.tensor([1.0, 0, 0], device=self.device))
        v_axis = geo.quat_rotate(quat, torch.tensor([0, 0, 1.0], device=self.device))
        # force_matrix_w holds normal forces only; the tangential part is PhysX friction.
        normal = float(per_block.sum(0) @ outward)
        shear = torch.nan_to_num(friction).sum(0)                   # NaN for blocks not in contact
        peak = float(per_block_history.sum(1).norm(dim=-1).max())
        return dict(net_w=_numpy(net), net_peak=float(data.net_forces_w_history[0, :, 0].norm(dim=-1).max()),
                    block_force_w=_numpy(per_block), block_peak=peak,
                    friction_w=_numpy(friction), contact_pos_w=_numpy(points),
                    normal=normal, shear_uv=np.array([float(shear @ u_axis), float(shear @ v_axis)]))

    def _virtual_gel(self, finger, pos, quat, block_pos, block_quat):
        """GelSight: virtual-gel indentation, the input to TacSL's GelSight image renderer."""
        rows, cols = GEL['rows'], GEL['cols']
        thickness, margin = VIRTUAL['thickness'], VIRTUAL['margin']
        height = torch.zeros(rows * cols, device=self.device)
        centre = geo.quat_rotate(quat, self.gel_centre) + pos
        near = geo.near_pieces(self.blocks_convex, block_pos, centre, self.gel_reach)
        if len(near):
            convex = geo.subset(self.blocks_convex, near)
            # Rays from `margin` inside the finger surface outward along the gel normal.
            origins = geo.quat_rotate(quat, finger['gel_base'] - margin * finger['inward']) + pos
            directions = geo.quat_rotate(quat, finger['inward']).expand_as(origins)
            local_o, local_d = geo.to_piece_frames(convex, block_pos, block_quat, origins, directions)
            t, _ = geo.ray_entry(convex.normals, convex.offsets, local_o, local_d)
            gap = t.min(0).values - margin       # distance from the finger surface to the object
            height = torch.where(torch.isfinite(gap) & finger['gel_mask'],
                                 (thickness - gap).clamp(0, thickness + margin), height)
        return height.reshape(rows, cols)

    def _write_gel_frame(self, rgb):
        if self.writer is None:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(str(self.output/'gelsight.mp4'), fps=self.env.obs_manager.collect_freq,
                codec='libx264', pixelformat='yuv420p', quality=None, macro_block_size=2,
                output_params=['-crf', '16', '-movflags', '+faststart'])
        self.writer.append_data(np.concatenate(list(rgb), axis=1))

    # ------------------------------------------------------------------ output
    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.close()
        if not self.rows:
            return
        stack = lambda key: np.stack([np.stack(r[key]) if isinstance(r[key], list) else r[key] for r in self.rows])
        contact = lambda key: np.stack([[c[key] for c in r['contact']] for r in self.rows])
        np.savez_compressed(self.output/'tactile.npz',
            step=np.array([r['step'] for r in self.rows]), seconds=np.array([r['seconds'] for r in self.rows]),
            finger_pose=stack('finger_pose'), block_pose=stack('block_pose'),
            contact_net_w=contact('net_w'), contact_net_peak=contact('net_peak'),
            contact_block_force_w=contact('block_force_w'), contact_block_peak=contact('block_peak'),
            contact_friction_w=contact('friction_w'), contact_pos_w=contact('contact_pos_w'),
            contact_normal=contact('normal'), contact_shear_uv=contact('shear_uv'),
            gel_height=stack('gel_height'), gel_depth=np.array([r['gel_depth'] for r in self.rows]),
            gel_nominal=self.gel_nominal, gel_mask=self.gel_mask)
        self.meta.update(frames=len(self.rows), mean_record_seconds=float(np.mean([r['seconds'] for r in self.rows])))
        (self.output/'meta.json').write_text(json.dumps(self.meta, indent=2))
