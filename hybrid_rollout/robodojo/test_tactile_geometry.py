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
    sdf, _ = geo.convex_sdf(convex.normals, convex.offsets, torch.tensor([[[0.0, 0, 0]], [[0.002, 0.002, 0.002]]]))
    assert sdf[0, 0] == pytest.approx(-0.01, abs=1e-6)
    assert sdf[1, 0] == pytest.approx(-0.002, abs=1e-6)


def test_sdf_depth_and_gradient_in_world():
    convex = box()
    pos, quat = identity_pose((1.0, 2.0, 3.0))
    points = torch.tensor([[1.0, 2.0, 3.009], [1.0, 2.0, 3.05]])
    local = geo.to_piece_frames(convex, pos, quat, points)
    sdf, grad = geo.convex_sdf(convex.normals, convex.offsets, local)
    assert sdf[0, 0] == pytest.approx(-0.001, abs=1e-6)
    assert torch.allclose(grad[0, 0], torch.tensor([0, 0, 1.0]))
    assert sdf[0, 1] > 0


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


def test_splat_preserves_total_force():
    u, v = geo.pad_grid((-0.015, 0.072), (-0.031, 0.031), 24, 16)
    assert u[0, 0] > u[-1, 0]  # fingertip on the top row
    grid = geo.splat([0.03, 0.0], [0.0, 0.01], [2.0, 1.0], u, v, 0.002)
    assert grid.sum() == pytest.approx(3.0, rel=0.02)
    row, col = np.unravel_index(grid.argmax(), grid.shape)
    assert abs(u[row, col] - 0.03) < 0.004 and abs(v[row, col]) < 0.004
    vectors = geo.splat([0.03], [0.0], [[1.0, -2.0]], u, v, 0.002)
    assert vectors.shape == (24, 16, 2)
    assert vectors[..., 1].sum() == pytest.approx(-2.0, rel=0.02)
    assert geo.splat([], [], np.zeros(0), u, v, 0.002).shape == (24, 16)


def test_quaternion_round_trip():
    quat = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]])
    vectors = torch.tensor([[0.1, -0.2, 0.3]])
    back = geo.quat_rotate_inverse(quat, geo.quat_rotate(quat, vectors))
    assert torch.allclose(back, vectors, atol=1e-6)
    assert torch.allclose(geo.quat_rotate(quat, torch.tensor([[1.0, 0, 0]])),
                          torch.tensor([[np.cos(np.pi / 4), 0, -np.sin(np.pi / 4)]], dtype=torch.float32), atol=1e-6)


def test_patch_pressure_fills_the_face_contact_polygon():
    u, v = geo.pad_grid((0.020, 0.080), (-0.024, 0.024), 24, 20)
    corners_u, corners_v = [0.056, 0.056, 0.071, 0.071], [-0.010, 0.008, -0.010, 0.008]
    grid = geo.patch_pressure(corners_u, corners_v, [5.0, 5.0, 5.0, 5.0], u, v, 0.0015)
    assert grid.sum() == pytest.approx(20.0)
    inside = (u > 0.056) & (u < 0.071) & (v > -0.010) & (v < 0.008)
    assert grid[~inside].sum() == 0 and (grid[inside] > 0).all()
    assert np.ptp(grid[inside]) < 1e-9          # equal corner forces: uniform pressure
    tilted = geo.patch_pressure(corners_u, corners_v, [1.0, 1.0, 9.0, 9.0], u, v, 0.0015)
    assert tilted[inside & (u > 0.066)].mean() > 3 * tilted[inside & (u < 0.061)].mean()


def test_patch_pressure_falls_back_for_edge_and_point_contacts():
    u, v = geo.pad_grid((0.020, 0.080), (-0.024, 0.024), 24, 20)
    edge = geo.patch_pressure([0.05, 0.06, 0.07], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], u, v, 0.0015)
    point = geo.patch_pressure([0.05], [0.0], [2.0], u, v, 0.0015)
    assert edge.sum() == pytest.approx(3.0, rel=0.03) and point.sum() == pytest.approx(2.0, rel=0.03)
    assert geo.patch_pressure([0.05], [0.0], [0.0], u, v, 0.0015).sum() == 0
