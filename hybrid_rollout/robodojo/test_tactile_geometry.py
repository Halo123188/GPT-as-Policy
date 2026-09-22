import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('trimesh')

from hybrid_rollout.robodojo.tactile import geometry as geo


def box(size=(0.04, 0.06, 0.02)):
    points = geo.box_points(size)
    normals, offsets = geo.hull_planes(points)
    return geo.ConvexSet.from_pieces([(0, normals, offsets, points)])


def identity_pose(position=(0.0, 0.0, 0.0)):
    return torch.tensor([position], dtype=torch.float32), torch.tensor([[1.0, 0, 0, 0]])


def test_box_hull_has_six_planes():
    normals, offsets = geo.hull_planes(geo.box_points((0.04, 0.06, 0.02)))
    assert len(normals) == 6
    assert sorted(np.round(offsets, 6)) == [0.01, 0.01, 0.02, 0.02, 0.03, 0.03]


def test_ray_entry_hits_miss_and_inside():
    convex = box()
    origins = torch.tensor([[[-0.1, 0, 0], [-0.1, 0.05, 0], [0, 0, 0]]])
    directions = torch.tensor([[[1.0, 0, 0]] * 3])
    t, normal = geo.ray_entry(convex.normals, convex.offsets, origins, directions)
    assert t[0, 0] == pytest.approx(0.08, abs=1e-6)
    assert torch.allclose(normal[0, 0], torch.tensor([-1.0, 0, 0]))
    assert torch.isinf(t[0, 1])            # passes beside the box
    assert t[0, 2] == pytest.approx(-0.02, abs=1e-6)  # origin inside: entry is behind it


def test_padding_planes_never_activate():
    small = geo.box_points((0.02, 0.02, 0.02))
    wedge = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]], float)
    pieces = [(0, *geo.hull_planes(small), small), (1, *geo.hull_planes(wedge), wedge)]
    convex = geo.ConvexSet.from_pieces(pieces)
    assert convex.normals.shape[1] == 6 and torch.isinf(convex.offsets[1, 4:]).all()
    origins = torch.tensor([[[-0.1, 0.002, 0.002], [-0.1, 0.019, 0.019]]]).expand(2, -1, -1)
    t, _ = geo.ray_entry(convex.normals, convex.offsets, origins, torch.tensor([1.0, 0, 0]).expand(2, 2, 3))
    assert t[0, 0] == pytest.approx(0.09, abs=1e-6) and t[1, 0] == pytest.approx(0.1, abs=1e-6)
    assert torch.isinf(t[1, 1])            # beyond the wedge's slanted face


def test_rotated_body_frames():
    convex = box((0.1, 0.02, 0.02))
    angle = np.pi / 2  # box long axis now along world y
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]], dtype=torch.float32)
    local_o, local_d = geo.to_piece_frames(convex, pos, quat, torch.tensor([[0.0, -0.2, 0]]),
                                           torch.tensor([[0.0, 1.0, 0]]))
    t, _ = geo.ray_entry(convex.normals, convex.offsets, local_o, local_d)
    assert t[0, 0] == pytest.approx(0.15, abs=1e-5)


def test_gel_height_from_ray_gap():
    """Indentation = thickness - gap, the quantity the recorder feeds TacSL's GelSight renderer."""
    thickness, margin = 0.001, 0.003
    convex = box((0.02, 0.02, 0.02))
    pos, quat = identity_pose((0.0, 0.0105, 0.0))   # block face 0.5 mm in front of the finger surface
    u, v = np.meshgrid(np.linspace(-0.0175, 0.0175, 8), [0.0], indexing='ij')
    base = torch.tensor(np.stack([u.ravel(), np.zeros(u.size), v.ravel()], -1), dtype=torch.float32)
    inward = torch.tensor([0.0, 1.0, 0.0])
    origins = base - margin * inward
    local_o, local_d = geo.to_piece_frames(convex, pos, quat, origins, inward.expand_as(origins))
    t, _ = geo.ray_entry(convex.normals, convex.offsets, local_o, local_d)
    gap = t.min(0).values - margin
    height = torch.where(torch.isfinite(gap), (thickness - gap).clamp(0, thickness + margin), torch.zeros_like(gap))
    inside = np.abs(u.ravel()) < 0.01
    assert np.allclose(height.numpy()[inside], 0.0005, atol=1e-6)
    assert np.allclose(height.numpy()[~inside], 0.0)


def test_pad_grid_puts_the_fingertip_on_top():
    u, v = geo.pad_grid((0.0, 0.03), (-0.01, 0.01), 3, 2)
    assert np.allclose(u[:, 0], [0.025, 0.015, 0.005]) and np.allclose(v[0], [-0.005, 0.005])


def test_quaternion_round_trip():
    quat = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]])
    vectors = torch.tensor([[0.1, -0.2, 0.3]])
    back = geo.quat_rotate_inverse(quat, geo.quat_rotate(quat, vectors))
    assert torch.allclose(back, vectors, atol=1e-6)
    assert torch.allclose(geo.quat_rotate(quat, torch.tensor([[1.0, 0, 0]])),
                          torch.tensor([[np.cos(np.pi / 4), 0, -np.sin(np.pi / 4)]], dtype=torch.float32), atol=1e-6)


def test_mounted_camera_looks_out_through_the_gel_with_the_tip_up():
    pytest.importorskip('scipy')
    from scipy.spatial.transform import Rotation
    from hybrid_rollout.robodojo.tactile.mounted import _camera_rotation, _quat_wxyz
    for sign in (-1.0, 1.0):
        rotation = _camera_rotation(sign)
        w, x, y, z = _quat_wxyz(rotation)
        assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), rotation, atol=1e-6)
        assert np.allclose(rotation @ [0, 0, -1], [0, sign, 0])   # looks along the gripping-face normal
        assert np.allclose(rotation @ [0, 1, 0], [1, 0, 0])       # image up = toward the fingertip
