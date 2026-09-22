"""GelSight real: a flat GelSight pad physically mounted on each X5 fingertip (changes the contact physics).

Like a real GelSight, the gel is a flat pad: a 2 mm compliant slab over the R1.5 window, flush with
the fingertip, with PhysX compliant contact as TacSL uses for its elastomer. Objects therefore really
sink into the gel, deeper the harder the grip. A depth camera sits behind the gel inside the finger,
as in TacSL's GelSight finger. Its clipping range keeps only the gel layer: the finger body lies
nearer than the gel's back face and the scene beyond the gel surface lies past the far plane, so the
camera sees an object only where it has pressed into the gel. Indentation = gel surface distance -
depth, rendered like the real sensor.

The gel is invisible to the RGB cameras, so the policy's images are unchanged; its grasps are not,
because the contact is soft and the pad sits 2 mm proud of the fingertip.
"""
import numpy as np
import torch

# Flat gripping pad of the X5 finger hull (link frame, Assets/Robots/x5/ARX.usd): y = sign*24.494 mm,
# x = 56-71 mm, z = -10.4-8.0 mm; the face behind it slopes back. The gel is one flat slab over the
# GelSight R1.5 window, flush with the tip, so it spans the pad and bridges the slope behind it.
# Stiffness and damping are per PhysX contact point.
MOUNT = dict(face=0.024494, thickness=0.002, backing=0.003, camera_distance=0.020,
             stiffness=6000.0, damping=10.0, static_friction=0.5, dynamic_friction=0.5)
PAD, CAMERA = 'gelsight_pad', 'gelsight_camera'


def mount_gel_pads(gel, fingers):
    """Spawn every X5 with gel pads and tactile cameras on its fingers (before the environment exists).

    ``gel`` is the image window (tip_u, v_center, rows, cols, mm_per_pixel); ``fingers`` the
    (link, sign of the gripping-face normal along local y) pairs.
    """
    import env.robot_manager.robot_config.x5 as x5
    if getattr(x5.get_robot_config, 'tactile_gel_pads', False):
        return
    original = x5.get_robot_config

    def with_gel_pads():
        cfg = original()
        spawn = cfg.spawn.func

        def spawn_with_gel_pads(prim_path, spawn_cfg, *args, **kwargs):
            import isaaclab.sim as sim_utils
            prim = spawn(prim_path, spawn_cfg, *args, **kwargs)
            for root in sim_utils.find_matching_prim_paths(prim_path):
                for link, sign in fingers:
                    _author_pad(prim.GetStage(), f'{root}/{link}', sign, gel)
            return prim

        return cfg.replace(spawn=cfg.spawn.replace(func=spawn_with_gel_pads))

    with_gel_pads.tactile_gel_pads = True
    x5.get_robot_config = with_gel_pads


def _author_pad(stage, link_path, sign, gel):
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom, UsdPhysics
    length, width = gel['rows'] * gel['mm_per_pixel'] / 1000, gel['cols'] * gel['mm_per_pixel'] / 1000
    face, thickness, backing = MOUNT['face'], MOUNT['thickness'], MOUNT['backing']
    # Collider: from `backing` inside the tip pad to `thickness` in front of it. Guide purpose keeps
    # it out of every rendered image.
    pad = UsdGeom.Cube.Define(stage, f'{link_path}/{PAD}')
    pad.CreateSizeAttr(1.0)
    pad.CreatePurposeAttr(UsdGeom.Tokens.guide)
    pad.AddTranslateOp().Set(Gf.Vec3d(gel['tip_u'] - length / 2, sign * (face + (thickness - backing) / 2),
                                      gel['v_center']))
    pad.AddScaleOp().Set(Gf.Vec3f(length, thickness + backing, width))
    UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
    material, material_cfg = f'{link_path}/{PAD}_material', sim_utils.RigidBodyMaterialCfg(
        static_friction=MOUNT['static_friction'], dynamic_friction=MOUNT['dynamic_friction'],
        compliant_contact_stiffness=MOUNT['stiffness'], compliant_contact_damping=MOUNT['damping'])
    material_cfg.func(material, material_cfg)
    sim_utils.bind_physics_material(pad.GetPath().pathString, material)
    # Camera `camera_distance` behind the gel surface, looking out through it with the fingertip at
    # the top of the image. The frustum spans exactly the R1.5 window at the gel surface.
    distance = MOUNT['camera_distance']
    camera = UsdGeom.Camera.Define(stage, f'{link_path}/{CAMERA}')
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    focal = 20.0
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(focal * width / distance)
    camera.CreateVerticalApertureAttr(focal * length / distance)
    camera.CreateClippingRangeAttr(Gf.Vec2f(distance - thickness + 1e-4, distance + 1e-4))
    camera.AddTranslateOp().Set(Gf.Vec3d(gel['tip_u'] - length / 2, sign * (face + thickness - distance),
                                         gel['v_center']))
    camera.AddOrientOp().Set(Gf.Quatf(*_quat_wxyz(_camera_rotation(sign))))
    camera.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))  # IsaacLab views expect translate, orient, scale


def _camera_rotation(sign):
    """Camera axes in the link frame: it looks along -Z = the gripping-face normal, +Y = toward the tip."""
    up, back = np.array([1.0, 0, 0]), np.array([0, -sign, 0])
    return np.stack([np.cross(up, back), up, back], axis=1)


def _quat_wxyz(rotation):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return float(w), float(x), float(y), float(z)


class MountedGel:
    """Depth cameras of the mounted pads, one tile per finger, as indentation maps (meters)."""

    def __init__(self, finger_paths, gel, device, render):
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        root = finger_paths[0].rsplit('/', 2)[0]
        cfg = TiledCameraCfg(prim_path=f'{root}/robot[0-9]+/link[78]/{CAMERA}', spawn=None,
                             data_types=['distance_to_image_plane'], width=gel['cols'], height=gel['rows'],
                             update_period=0.0, update_latest_camera_pose=False)
        self.camera = TiledCamera(cfg)
        # Created after the timeline started playing: initialize by hand, as TacSL does.
        self.camera._initialize_impl()
        self.camera._is_initialized = True
        # Its render product has never been drawn: reading it now would block. Rendering does not
        # step physics.
        for _ in range(2):
            render()
        paths = [prim.GetPath().pathString.rsplit('/', 1)[0] for prim in self.camera._view.prims]
        self.order = torch.as_tensor([paths.index(path) for path in finger_paths], device=device)
        self.device = device

    def height(self):
        self.camera.update(0.0, force_recompute=True)
        depth = self.camera.data.output['distance_to_image_plane'][..., 0].to(self.device)[self.order]
        distance, thickness = MOUNT['camera_distance'], MOUNT['thickness']
        valid = torch.isfinite(depth) & (depth > 0)
        return torch.where(valid, (distance - depth).clamp(0, thickness), torch.zeros_like(depth))
