"""Simulator-free geometry for the tactile models: convex pieces, ray casts, SDFs and grids.

Every contact body in build_tower (the blocks and the X5 finger collision hulls) is convex, so a
piece is stored as half-spaces ``normals @ p <= offsets`` in its rigid-body frame. That gives an
exact inside distance (for TacSL's penetration depth) and exact ray entry points (for the gel
height map) on CPU or GPU, for any number of objects, without PhysX SDF meshes.
"""
from dataclasses import dataclass

import numpy as np
import torch


def hull_planes(points):
    """Outward unit normals (F, 3) and offsets (F,) of the convex hull of ``points`` (N, 3)."""
    import trimesh
    hull = trimesh.convex.convex_hull(np.asarray(points, np.float64))
    normals = np.asarray(hull.face_normals, np.float64)
    offsets = np.einsum('fi,fi->f', normals, hull.vertices[hull.faces[:, 0]])
    _, keep = np.unique(np.round(np.c_[normals, offsets], 6), axis=0, return_index=True)
    return normals[np.sort(keep)], offsets[np.sort(keep)]


def box_points(size, transform=None):
    """Eight corners of an axis-aligned box of edge lengths ``size``, optionally transformed (4x4)."""
    half = np.asarray(size, np.float64) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    return corners if transform is None else apply_transform(transform, corners)


def apply_transform(matrix, points):
    """Column-convention 4x4 transform of (N, 3) points."""
    matrix = np.asarray(matrix, np.float64)
    return np.asarray(points, np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass
class ConvexSet:
    """Padded half-space stack for P convex pieces, each attached to one of B rigid bodies."""
    normals: torch.Tensor   # (P, F, 3); padding rows are zero with offset +inf (never active)
    offsets: torch.Tensor   # (P, F)
    body: torch.Tensor      # (P,) index of the owning body
    radius: torch.Tensor    # (P,) bounding radius about the body origin, for culling

    @classmethod
    def from_pieces(cls, pieces, device='cpu'):
        """``pieces`` is a list of (body_index, normals (F, 3), offsets (F,), points (N, 3))."""
        faces = max(len(n) for _, n, _, _ in pieces)
        normals = np.zeros((len(pieces), faces, 3))
        offsets = np.full((len(pieces), faces), np.inf)
        for i, (_, n, d, _) in enumerate(pieces):
            normals[i, :len(n)], offsets[i, :len(d)] = n, d
        radius = [np.linalg.norm(p, axis=1).max() for *_, p in pieces]
        as_tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        return cls(as_tensor(normals), as_tensor(offsets), as_tensor([b for b, *_ in pieces], torch.long),
                   as_tensor(radius))


def quat_rotate(quat_wxyz, vectors):
    """Rotate (..., 3) vectors by (..., 4) wxyz quaternions (broadcasting)."""
    w, xyz = quat_wxyz[..., :1], quat_wxyz[..., 1:]
    t = 2 * torch.cross(xyz.expand_as(vectors), vectors, dim=-1)
    return vectors + w * t + torch.cross(xyz.expand_as(vectors), t, dim=-1)


def quat_rotate_inverse(quat_wxyz, vectors):
    conj = torch.cat([quat_wxyz[..., :1], -quat_wxyz[..., 1:]], dim=-1)
    return quat_rotate(conj, vectors)


def ray_entry(normals, offsets, origins, directions):
    """Cyrus-Beck ray/convex entry for rays already expressed in each piece's frame.

    normals (P, F, 3), offsets (P, F), origins/directions (P, R, 3).
    Returns t_enter (P, R) (+inf on a miss; negative when the origin is inside), and the entry
    face normal (P, R, 3) in the piece frame.
    """
    denom = torch.einsum('pfk,prk->prf', normals, directions)
    num = offsets[:, None, :] - torch.einsum('pfk,prk->prf', normals, origins)
    ratio = num / torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    entering, exiting = denom < -1e-12, denom > 1e-12
    parallel_outside = (~entering & ~exiting & (num < 0)).any(-1)
    t_in = torch.where(entering, ratio, torch.full_like(ratio, -torch.inf))
    t_out = torch.where(exiting, ratio, torch.full_like(ratio, torch.inf))
    t_enter, face = t_in.max(-1)
    t_exit = t_out.min(-1).values
    hit = (t_enter <= t_exit) & (t_exit >= 0) & ~parallel_outside
    entry_normal = torch.gather(normals, 1, face[..., None].expand(-1, -1, 3))
    return torch.where(hit, t_enter, torch.full_like(t_enter, torch.inf)), entry_normal


def convex_sdf(normals, offsets, points):
    """Signed distance of (P, R, 3) points to each convex piece: exact inside, a lower bound outside.

    Returns (sdf (P, R), gradient (P, R, 3) in the piece frame).
    """
    signed = torch.einsum('pfk,prk->prf', normals, points) - offsets[:, None, :]
    sdf, face = signed.max(-1)
    grad = torch.gather(normals, 1, face[..., None].expand(-1, -1, 3))
    return sdf, grad


def to_piece_frames(convex, body_pos, body_quat, points, directions=None):
    """Express world (R, 3) points/directions in every piece frame -> (P, R, 3)."""
    pos, quat = body_pos[convex.body], body_quat[convex.body]
    local = quat_rotate_inverse(quat[:, None, :], points[None] - pos[:, None, :])
    if directions is None:
        return local
    return local, quat_rotate_inverse(quat[:, None, :], directions[None].expand(len(pos), -1, -1))


def near_pieces(convex, body_pos, center, reach):
    """Indices of pieces whose bounding sphere comes within ``reach`` of ``center``."""
    distance = torch.linalg.norm(body_pos[convex.body] - center, dim=-1)
    return torch.nonzero(distance <= convex.radius + reach).flatten()


def subset(convex, index):
    return ConvexSet(convex.normals[index], convex.offsets[index], convex.body[index], convex.radius[index])


def pad_grid(u_range, v_range, rows, cols):
    """Cell-centre coordinates of a rows x cols grid; row 0 is the u_range maximum (the fingertip)."""
    u_edges = np.linspace(u_range[1], u_range[0], rows + 1)
    v_edges = np.linspace(v_range[0], v_range[1], cols + 1)
    u = (u_edges[:-1] + u_edges[1:]) / 2
    v = (v_edges[:-1] + v_edges[1:]) / 2
    return np.meshgrid(u, v, indexing='ij')


def splat(u, v, values, u_centres, v_centres, sigma):
    """Spread point values (N,) or (N, C) onto a grid with a normalised Gaussian kernel.

    The kernel integrates to one over the plane, so the grid sum equals the point sum for points
    well inside the grid. Returns (rows, cols) or (rows, cols, C).
    """
    values = np.asarray(values, np.float64)
    squeeze = values.ndim == 1
    values = values[:, None] if squeeze else values
    rows, cols = u_centres.shape
    out = np.zeros((rows, cols, values.shape[1]))
    if len(values):
        cell = abs(u_centres[0, 0] - u_centres[min(1, rows - 1), 0]) or 1.0
        cell_v = abs(v_centres[0, min(1, cols - 1)] - v_centres[0, 0]) or 1.0
        du = u_centres[None] - np.asarray(u)[:, None, None]
        dv = v_centres[None] - np.asarray(v)[:, None, None]
        kernel = np.exp(-(du ** 2 + dv ** 2) / (2 * sigma ** 2)) * cell * cell_v / (2 * np.pi * sigma ** 2)
        out = np.einsum('nrc,nk->rck', kernel, values)
    return out[..., 0] if squeeze else out


def patch_pressure(u, v, forces, u_centres, v_centres, sigma):
    """Pressure grid for one finger-object contact patch from its PhysX manifold points.

    PhysX reports a flat face-on-face contact by the corners of the patch polygon, each with its
    share of the force. Inside that polygon the pressure is interpolated linearly between the
    corner forces (so a tilted grasp shows a pressure gradient), rescaled to the patch's total
    force. Point, edge or sub-cell patches fall back to the Gaussian splat. Returns (rows, cols).
    """
    points, forces = np.c_[u, v], np.asarray(forces, np.float64)
    total = forces.sum()
    if total <= 0:
        return np.zeros(u_centres.shape)
    if len(points) >= 3:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay, QhullError
        try:
            triangulation = Delaunay(points)
        except QhullError:  # collinear: an edge contact
            triangulation = None
        if triangulation is not None:
            cells = np.c_[u_centres.ravel(), v_centres.ravel()]
            inside = triangulation.find_simplex(cells) >= 0
            if inside.any():
                values = np.clip(np.nan_to_num(LinearNDInterpolator(triangulation, forces)(cells[inside])), 0, None)
                if values.sum() > 0:
                    grid = np.zeros(cells.shape[0])
                    grid[inside] = values * total / values.sum()
                    return grid.reshape(u_centres.shape)
    return splat(u, v, forces, u_centres, v_centres, sigma)
