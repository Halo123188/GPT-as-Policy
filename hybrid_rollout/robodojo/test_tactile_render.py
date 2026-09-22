import json

import numpy as np
import pytest

pytest.importorskip('matplotlib')
imageio = pytest.importorskip('imageio.v2')
pytest.importorskip('imageio_ffmpeg')
pytest.importorskip('cv2')
pytest.importorskip('torch')

from hybrid_rollout.robodojo.tactile import geometry as geo
from hybrid_rollout.robodojo.tactile import render_video
from hybrid_rollout.robodojo.tactile.recorder import GEL, TAXEL

STEPS, FINGERS = 4, ('left_link7', 'left_link8', 'right_link7', 'right_link8')


def write_video(path, frames):
    writer = imageio.get_writer(str(path), fps=25, codec='libx264', macro_block_size=2)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def fake_archive(root):
    tactile = root/'sim'/'tactile'
    tactile.mkdir(parents=True)
    write_video(root/'sim'/'sensors.mp4', [np.full((360, 1920, 3), 40 * k, np.uint8) for k in range(STEPS)])
    write_video(tactile/'gelsight.mp4', [np.full((320, 960, 3), 90, np.uint8) for _ in range(STEPS)])
    u, v = geo.pad_grid(TAXEL['u_range'], TAXEL['v_range'], TAXEL['rows'], TAXEL['cols'])
    (tactile/'meta.json').write_text(json.dumps(dict(
        fingers=[dict(name=n) for n in FINGERS], blocks={f'block{i}': f'/b{i}' for i in range(8)},
        taxel=dict(TAXEL, u_centres=u[:, 0].tolist(), v_centres=v[0].tolist()), gel=dict(GEL))))
    rng = np.random.default_rng(0)
    points = [rng.random((2, 2), np.float32) * 0.03 for _ in range(STEPS * len(FINGERS))]
    np.savez(tactile/'tactile.npz', step=np.arange(STEPS),
             contact_normal=rng.random((STEPS, 4)) * 5, contact_shear_uv=rng.random((STEPS, 4, 2)),
             contact_friction_w=rng.random((STEPS, 4, 8, 3)),
             contact_block_force_w=rng.random((STEPS, 4, 8, 3)),
             taxel_pressure=rng.random((STEPS, 4, TAXEL['rows'], TAXEL['cols'])),
             taxel_shear=rng.random((STEPS, 4, TAXEL['rows'], TAXEL['cols'], 2)) * 0.1,
             taxel_points_uv=np.concatenate(points), taxel_point_offsets=np.r_[0, np.cumsum([len(p) for p in points])],
             gel_field_normal=rng.random((STEPS, 4, *GEL['field'])) * 0.1,
             gel_field_shear=rng.random((STEPS, 4, *GEL['field'], 2)) * 0.01,
             gel_height=(rng.random((STEPS, 4, GEL['rows'] // 4, GEL['cols'] // 4)) * 1e-3).astype(np.float16),
             gel_mask=np.ones((4, GEL['rows'], GEL['cols']), bool))


def test_renders_three_side_by_side_videos(tmp_path):
    fake_archive(tmp_path)
    written = render_video.render(tmp_path, tmp_path/'videos')
    assert [p.rsplit('/', 1)[-1] for p in written] == ['contact.mp4', 'taxel.mp4', 'gelsight.mp4']
    for path in written:
        reader = imageio.get_reader(path)
        frames = [f for f in reader]
        reader.close()
        assert len(frames) == STEPS
        assert frames[0].shape == (render_video.PANEL[1], render_video.LEFT_W + render_video.PANEL[0], 3)
