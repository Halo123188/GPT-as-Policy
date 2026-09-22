import json

import numpy as np
import pytest

pytest.importorskip('matplotlib')
imageio = pytest.importorskip('imageio.v2')
pytest.importorskip('imageio_ffmpeg')
pytest.importorskip('cv2')
pytest.importorskip('torch')

from hybrid_rollout.robodojo.tactile import render_video
from hybrid_rollout.robodojo.tactile.recorder import GEL, VIRTUAL

STEPS, FINGERS = 4, ('left_link7', 'left_link8', 'right_link7', 'right_link8')


def write_video(path, frames):
    writer = imageio.get_writer(str(path), fps=25, codec='libx264', macro_block_size=2)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def fake_archive(root, kind='virtual'):
    tactile = root/'sim'/'tactile'
    tactile.mkdir(parents=True)
    write_video(root/'sim'/'sensors.mp4', [np.full((360, 1920, 3), 40 * k, np.uint8) for k in range(STEPS)])
    write_video(tactile/'gelsight.mp4', [np.full((320, 960, 3), 90, np.uint8) for _ in range(STEPS)])
    (tactile/'meta.json').write_text(json.dumps(dict(
        fingers=[dict(name=n) for n in FINGERS], blocks={f'block{i}': f'/b{i}' for i in range(8)},
        gel={**GEL, **VIRTUAL, 'kind': kind})))
    rng = np.random.default_rng(0)
    np.savez(tactile/'tactile.npz', step=np.arange(STEPS),
             contact_normal=rng.random((STEPS, 4)) * 5, contact_shear_uv=rng.random((STEPS, 4, 2)),
             contact_friction_w=rng.random((STEPS, 4, 8, 3)),
             contact_block_force_w=rng.random((STEPS, 4, 8, 3)),
             gel_height=(rng.random((STEPS, 4, GEL['rows'] // 4, GEL['cols'] // 4)) * 1e-3).astype(np.float16),
             gel_nominal=np.full((GEL['rows'], GEL['cols'], 3), 90, np.uint8))


def test_renders_side_by_side_videos(tmp_path):
    fake_archive(tmp_path)
    written = render_video.render(tmp_path, tmp_path/'videos')
    assert [p.rsplit('/', 1)[-1] for p in written] == ['contact.mp4', 'gelsight.mp4']
    for path in written:
        reader = imageio.get_reader(path)
        frames = [f for f in reader]
        reader.close()
        assert len(frames) == STEPS
        assert frames[0].shape == (render_video.PANEL[1], render_video.LEFT_W + render_video.PANEL[0], 3)


def test_mounted_archive_renders_gelsight_real(tmp_path):
    fake_archive(tmp_path, kind='mounted')
    written = render_video.render(tmp_path, tmp_path/'videos', limit=2)
    assert [p.rsplit('/', 1)[-1] for p in written] == ['contact.mp4', 'gelsight_real.mp4']
    with pytest.raises(ValueError):
        render_video.render(tmp_path, tmp_path/'videos', models=['gelsight'])
