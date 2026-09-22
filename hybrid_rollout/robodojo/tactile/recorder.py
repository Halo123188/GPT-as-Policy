"""In-simulator tactile recorder for the dual ARX X5 (import only inside a running Isaac Sim app).

All three models are observation-only: contact reporting is enabled on the robot links, but no
collision shape, material, mass or solver setting changes, so the rollout physics is the same as
without sensing.

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

FINGERS = (('link7', -1.0), ('link8', 1.0))  # link, sign of the gripping-face normal along local y
# Taxel skin over the distal 60 mm of the gripping face (2.5 x 2.4 mm cells). PhysX reports the
# flat fingertip pad's contacts at u = 55-74 mm, just past the hull tip (x = 71 mm) by the contact offset.
TAXEL = dict(u_range=(0.020, 0.080), v_range=(-0.024, 0.024), rows=24, cols=20, sigma=0.0015,
             inner_y=0.006)  # contacts with sign*y > inner_y lie on the gripping half of the finger
# GelSight R1.5 calibration (the only TacSL calibration published for Isaac Sim 5.1): 320x240 px,
# 0.0877 mm/px -> a 28.1 x 21.0 mm gel on the fingertip, covering the flat tip face.
GEL = dict(tip_u=0.071, v_center=-0.0012, rows=320, cols=240, mm_per_pixel=0.0877,
           thickness=0.001, margin=0.003, field=(20, 15), k_n=100.0, k_t=1.0, mu=2.0,
           calibration='gelsight_r15_data')
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
    """World position (N, 3), wxyz quaternion (N, 4), COM velocity (N, 6), COM in body (N, 3)."""
    transform = view.get_transforms()
    quat = transform[:, [6, 3, 4, 5]]
    return transform[:, :3], quat, view.get_velocities(), view.get_coms()[:, :3]


class TactileRecorder:
    def __init__(self, env, output, calibration_root):
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
                                         sensor=sensor, hull=(normals, offsets), hull_points=hull_points))

        self._build_taxel_grid()
        self._build_gel(calibration_root, GelsightRender, GelSightRenderCfg)
        self.rows, self.writer, self.closed = [], None, False
        self.meta = dict(schema='robodojo_tactile.v1', physics_dt=self.physics_dt, substeps=self.substeps,
            device=str(self.device), fingers=[dict(name=f['name'], side=f['side'], link=f['link'],
                prim_path=f['path'], gripping_face_normal_local=[0.0, f['sign'], 0.0]) for f in self.fingers],
            blocks=dict(zip(self.block_labels, self.block_paths)), block_colliders=self.block_pieces,
            taxel=dict(TAXEL, u_centres=self.taxel_u[:, 0].tolist(), v_centres=self.taxel_v[0].tolist()),
            gel=dict(GEL, u_range=list(self.gel_u_range), v_range=list(self.gel_v_range),
                     calibration_root=str(calibration_root), coverage=self.gel_coverage),
            observation_only=True, init_seconds=time.monotonic()-started)
        (self.output/'meta.json').write_text(json.dumps(self.meta, indent=2))

    # ------------------------------------------------------------------ setup
    def _build_taxel_grid(self):
        self.taxel_u, self.taxel_v = geo.pad_grid(TAXEL['u_range'], TAXEL['v_range'],
                                                  TAXEL['rows'], TAXEL['cols'])

    def _build_gel(self, calibration_root, GelsightRender, GelSightRenderCfg):
        rows, cols, pixel = GEL['rows'], GEL['cols'], GEL['mm_per_pixel'] / 1000
        self.gel_u_range = (GEL['tip_u'] - rows * pixel, GEL['tip_u'])
        self.gel_v_range = (GEL['v_center'] - cols * pixel / 2, GEL['v_center'] + cols * pixel / 2)
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
            finger['inward'] = torch.as_tensor(inward, dtype=torch.float32, device=self.device)
            fr, fc = GEL['field']
            step_r, step_c = rows // fr, cols // fc
            index = (np.arange(fr)[:, None] * step_r + step_r // 2) * cols + (np.arange(fc)[None] * step_c + step_c // 2)
            field_index = torch.as_tensor(index.reshape(-1), device=self.device)
            finger['field_points'] = finger['gel_base'][field_index] + GEL['thickness'] * finger['inward']
            finger['field_mask'] = finger['gel_mask'][field_index]  # no gel past the tapered tip
        self.gel_centre = torch.as_tensor([np.mean(self.gel_u_range), 0.0, GEL['v_center']],
                                          dtype=torch.float32, device=self.device)
        self.gel_reach = float(np.hypot(rows * pixel, cols * pixel) / 2 + GEL['thickness'] + GEL['margin'] + 0.03)
        cfg = GelSightRenderCfg(base_data_path=str(calibration_root), sensor_data_dir_name=GEL['calibration'],
            background_path='bg.jpg', calib_path='polycalib.npz', real_background='real_bg.npy',
            image_height=rows, image_width=cols, num_bins=120, mm_per_pixel=GEL['mm_per_pixel'])
        self.gel_renderer = GelsightRender(cfg, device=self.device)
        self.gel_nominal = _numpy(self.gel_renderer.render(torch.zeros(1, rows, cols, device=self.device))[0])

    # ------------------------------------------------------------------ recording
    def record(self, step_id):
        started = time.monotonic()
        finger_state = [_pose(f['sensor'].body_physx_view) for f in self.fingers]
        block_state = [_pose(view) for view in self.block_views]
        block_pos = torch.cat([s[0] for s in block_state]).to(self.device)
        block_quat = torch.cat([s[1] for s in block_state]).to(self.device)
        block_vel = torch.cat([s[2] for s in block_state]).to(self.device)
        block_com = torch.cat([s[3] for s in block_state]).to(self.device)
        row = dict(step=step_id, contact=[], taxel=[], gel_field_normal=[], gel_field_shear=[],
                   gel_depth=[], finger_pose=[], block_pose=_numpy(torch.cat([block_pos, block_quat], -1)))
        heights = []
        for finger, (pos, quat, vel, com) in zip(self.fingers, finger_state):
            pos, quat, vel, com = (x.to(self.device) for x in (pos, quat, vel, com))
            row['finger_pose'].append(_numpy(torch.cat([pos[0], quat[0]])))
            row['contact'].append(self._contact(finger, quat[0]))
            row['taxel'].append(self._taxel(finger, pos[0], quat[0]))
            height, normal, shear = self._gel(finger, pos[0], quat[0], vel[0], com[0],
                                             block_pos, block_quat, block_vel, block_com)
            heights.append(height)
            row['gel_field_normal'].append(normal)
            row['gel_field_shear'].append(shear)
            row['gel_depth'].append(float(height.max()))
        height = torch.stack(heights)
        rgb = _numpy(self.gel_renderer.render(height))
        self._write_gel_frame(rgb)
        pooled = torch.nn.functional.max_pool2d(height[:, None], HEIGHT_POOL)[:, 0]
        row['gel_height'] = _numpy(pooled).astype(np.float16)
        row['seconds'] = time.monotonic() - started
        self.rows.append(row)

    def _contact(self, finger, quat):
        """Option 1: IsaacLab ContactSensor readings for one finger."""
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

    def _taxel(self, finger, pos, quat):
        """Option 2: raw PhysX contact points of this finger vs each block, as pressure on its face."""
        view = finger['sensor'].contact_physx_view
        forces, points, _, _, count, start = view.get_contact_data(dt=self.physics_dt)
        friction, friction_points, f_count, f_start = view.get_friction_data(dt=self.physics_dt)
        count, start = count.reshape(-1).tolist(), start.reshape(-1).tolist()
        f_count, f_start = f_count.reshape(-1).tolist(), f_start.reshape(-1).tolist()
        sign = finger['sign']
        pressure = np.zeros(self.taxel_u.shape)
        shear = np.zeros((*self.taxel_u.shape, 2))
        kept_points, kept_forces, contacts = [], [], 0
        for block in range(len(self.block_paths)):
            if count[block] == 0:
                continue
            rows = slice(start[block], start[block] + count[block])
            local = _numpy(geo.quat_rotate_inverse(quat, points[rows].to(self.device) - pos))
            magnitude = _numpy(forces[rows, 0])
            contacts += len(local)
            inner = sign * local[:, 1] > TAXEL['inner_y']
            if not inner.any():
                continue
            patch = geo.patch_pressure(local[inner, 0], local[inner, 2], magnitude[inner],
                                       self.taxel_u, self.taxel_v, TAXEL['sigma'])
            pressure += patch
            kept_points.append(local[inner][:, [0, 2]]); kept_forces.append(magnitude[inner])
            if f_count[block] and patch.sum() > 0:
                f_rows = slice(f_start[block], f_start[block] + f_count[block])
                f_local = _numpy(geo.quat_rotate_inverse(quat, friction[f_rows].to(self.device)))
                f_side = sign * _numpy(geo.quat_rotate_inverse(quat, friction_points[f_rows].to(self.device) - pos))[:, 1]
                total = f_local[f_side > TAXEL['inner_y']][:, [0, 2]].sum(0)
                shear += patch[..., None] / patch.sum() * total  # patch's friction, spread like its pressure
        points_uv = np.concatenate(kept_points) if kept_points else np.zeros((0, 2))
        point_force = np.concatenate(kept_forces) if kept_forces else np.zeros(0)
        return dict(pressure=pressure.astype(np.float32), shear=shear.astype(np.float32),
                    contacts=contacts, inner_contacts=len(points_uv),
                    points_uv=points_uv.astype(np.float32), point_force=point_force.astype(np.float32))

    def _gel(self, finger, pos, quat, vel, com, block_pos, block_quat, block_vel, block_com):
        """Option 3: virtual-gel indentation (GelSight image input) and the TacSL force field."""
        rows, cols = GEL['rows'], GEL['cols']
        thickness, margin = GEL['thickness'], GEL['margin']
        centre = geo.quat_rotate(quat, self.gel_centre) + pos
        near = geo.near_pieces(self.blocks_convex, block_pos, centre, self.gel_reach)
        field_count = GEL['field'][0] * GEL['field'][1]
        height = torch.zeros(rows * cols, device=self.device)
        normal_force = torch.zeros(field_count, device=self.device)
        shear_force = torch.zeros(field_count, 2, device=self.device)
        if len(near):
            convex = geo.subset(self.blocks_convex, near)
            inward_w = geo.quat_rotate(quat, finger['inward'])
            # Height map: rays from `margin` inside the finger surface outward along the gel normal.
            origins = geo.quat_rotate(quat, finger['gel_base'] - margin * finger['inward']) + pos
            directions = inward_w.expand_as(origins)
            local_o, local_d = geo.to_piece_frames(convex, block_pos, block_quat, origins, directions)
            t, _ = geo.ray_entry(convex.normals, convex.offsets, local_o, local_d)
            gap = t.min(0).values - margin       # distance from the finger surface to the object
            height = torch.where(torch.isfinite(gap) & finger['gel_mask'],
                                 (thickness - gap).clamp(0, thickness + margin), height)
            # Force field (TacSL penalty model) at the undeformed gel surface points.
            points_w = geo.quat_rotate(quat, finger['field_points']) + pos
            local_p = geo.to_piece_frames(convex, block_pos, block_quat, points_w)
            sdf, grad = geo.convex_sdf(convex.normals, convex.offsets, local_p)
            depth, piece = (-sdf).clamp(min=0).max(0)
            depth = torch.where(finger['field_mask'], depth, torch.zeros_like(depth))
            if bool((depth > 0).any()):
                body = convex.body[piece]
                pts = torch.arange(len(piece), device=self.device)
                normal_w = geo.quat_rotate(block_quat[body], grad[piece, pts])
                fc = GEL['k_n'] * depth
                point_vel = vel[:3] + torch.cross(vel[3:].expand_as(points_w),
                                                  points_w - (geo.quat_rotate(quat, com) + pos), dim=-1)
                block_com_w = geo.quat_rotate(block_quat[body], block_com[body]) + block_pos[body]
                object_vel = block_vel[body, :3] + torch.cross(block_vel[body, 3:], points_w - block_com_w, dim=-1)
                relative = point_vel - object_vel
                vt = relative - normal_w * (normal_w * relative).sum(-1, keepdim=True)
                vt_norm = vt.norm(dim=-1)
                ft = torch.minimum(GEL['k_t'] * vt_norm, GEL['mu'] * fc)
                force_w = fc[:, None] * normal_w - ft[:, None] * vt / vt_norm.clamp(min=1e-9)[:, None]
                force_w = torch.where((depth > 0)[:, None], force_w, torch.zeros_like(force_w))
                u_axis = geo.quat_rotate(quat, torch.tensor([1.0, 0, 0], device=self.device))
                v_axis = geo.quat_rotate(quat, torch.tensor([0, 0, 1.0], device=self.device))
                normal_force = force_w @ (-inward_w)
                shear_force = torch.stack([force_w @ u_axis, force_w @ v_axis], -1)
        fr, fc_ = GEL['field']
        return (height.reshape(rows, cols), _numpy(normal_force.reshape(fr, fc_)),
                _numpy(shear_force.reshape(fr, fc_, 2)))

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
        taxel = lambda key: np.stack([[t[key] for t in r['taxel']] for r in self.rows])
        points = [[t['points_uv'] for t in r['taxel']] for r in self.rows]
        forces = [[t['point_force'] for t in r['taxel']] for r in self.rows]
        flat_points, flat_forces, offsets = [], [], [0]
        for step_points, step_forces in zip(points, forces):
            for p, f in zip(step_points, step_forces):
                flat_points.append(p); flat_forces.append(f); offsets.append(offsets[-1] + len(p))
        np.savez_compressed(self.output/'tactile.npz',
            step=np.array([r['step'] for r in self.rows]), seconds=np.array([r['seconds'] for r in self.rows]),
            finger_pose=stack('finger_pose'), block_pose=stack('block_pose'),
            contact_net_w=contact('net_w'), contact_net_peak=contact('net_peak'),
            contact_block_force_w=contact('block_force_w'), contact_block_peak=contact('block_peak'),
            contact_friction_w=contact('friction_w'), contact_pos_w=contact('contact_pos_w'),
            contact_normal=contact('normal'), contact_shear_uv=contact('shear_uv'),
            taxel_pressure=taxel('pressure'), taxel_shear=taxel('shear'),
            taxel_contacts=taxel('contacts'), taxel_inner_contacts=taxel('inner_contacts'),
            taxel_points_uv=np.concatenate(flat_points) if flat_points else np.zeros((0, 2), np.float32),
            taxel_point_force=np.concatenate(flat_forces) if flat_forces else np.zeros(0, np.float32),
            taxel_point_offsets=np.array(offsets),
            gel_height=stack('gel_height'), gel_depth=np.array([r['gel_depth'] for r in self.rows]),
            gel_field_normal=stack('gel_field_normal'), gel_field_shear=stack('gel_field_shear'),
            gel_nominal=self.gel_nominal,
            gel_mask=np.stack([_numpy(f['gel_mask'].reshape(GEL['rows'], GEL['cols'])) for f in self.fingers]))
        self.meta.update(frames=len(self.rows), mean_record_seconds=float(np.mean([r['seconds'] for r in self.rows])))
        (self.output/'meta.json').write_text(json.dumps(self.meta, indent=2))
