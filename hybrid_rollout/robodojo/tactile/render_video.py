"""Side-by-side videos: rollout cameras (left) and one tactile model over time (right).

Runs without Isaac Sim, from a finished archive (``<archive>/sim/sensors.mp4`` and
``<archive>/sim/tactile/``). Writes ``contact.mp4``, ``taxel.mp4`` and ``gelsight.mp4``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Dark chart theme (reference palette, dark mode) and categorical slots 1-4 for the four fingers.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#1a1a19', '#ffffff', '#c3c2b7', '#898781', '#2c2c2a', '#383835'
SERIES = ('#3987e5', '#d95926', '#199e70', '#c98500')
SEQUENTIAL = ('#1a1a19', '#104281', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb')        # force (blue)
SEQUENTIAL_2 = ('#1a1a19', '#5a2a14', '#9a3f1c', '#d95926', '#eb8a5f', '#f6c3a8')      # indentation (orange)
PANEL = (960, 810)          # right panel, px
LEFT_W = 960                # left column: head camera 960x540 over two wrist cameras 480x270
# Effective finger-block friction coefficient: in every demo episode the per-block friction/normal
# ratio saturates at exactly this value (finger has the default 0.5 material).
FRICTION_MU = 0.5
WINDOW_SECONDS = 5.0        # time-series plots scroll over the last 5 s
LABEL_ROOM = 0.08           # fraction of the window kept right of the cursor for value labels


def finger_label(name):
    side, link = name.split('_')
    return f"{side.capitalize()} · finger {'A' if link == 'link7' else 'B'}"


def load(archive):
    sim = Path(archive)/'sim'
    meta = json.loads((sim/'tactile'/'meta.json').read_text())
    data = dict(np.load(sim/'tactile'/'tactile.npz'))
    return sim, meta, data


def camera_frames(path):
    import imageio.v2 as imageio
    import cv2
    reader = imageio.get_reader(str(path))
    for frame in reader:
        head, left, right = frame[:, :640], frame[:, 640:1280], frame[:, 1280:1920]
        top = cv2.resize(head, (LEFT_W, 540), interpolation=cv2.INTER_AREA)
        bottom = np.concatenate([cv2.resize(x, (LEFT_W // 2, 270), interpolation=cv2.INTER_AREA)
                                 for x in (left, right)], axis=1)
        yield np.concatenate([top, bottom], axis=0)
    reader.close()


def label_cameras(frame):
    import cv2
    for text, (x, y) in (('head camera', (12, 28)), ('left wrist', (12, 568)), ('right wrist', (492, 568))):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def image_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.set_title(title, color=INK, fontsize=9, pad=4)


def nice_ceiling(x):
    """Smallest 1/2/2.5/5 x 10^k at or above x."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    power = 10 ** np.floor(np.log10(x))
    return float(min(m for m in (1, 2, 2.5, 5, 10) if m * power >= x * (1 - 1e-9)) * power)


class TimeSeries:
    """Four finger lines over a scrolling window that ends at now, like an oscilloscope.

    The y-axis fits the typical in-contact readings (90th percentile of nonzero values): impacts
    last one or two frames but are 5-10x the grip force, so they are clipped and marked with ▲
    at the top edge instead of flattening everything else.
    """
    def __init__(self, ax, t, values, names, ylabel, title, direct_labels=True, legend=True,
                 top=None, fmt='{:.1f}', reference=None):
        self.t, self.values, self.fmt = t, values, fmt
        style(ax, 'time (s)', ylabel)
        ax.set_title(title, color=INK, fontsize=10, loc='left', pad=18 if legend else 6)
        if top is None:
            active = values[np.isfinite(values) & (values > 1e-3)]
            top = nice_ceiling(1.4 * float(np.percentile(active, 90))) if active.size else 1.0
        self.top = top
        self.ax = ax
        ax.set_ylim(0, top)
        over = np.nan_to_num(values, nan=0.0) > top
        if over.any():
            unit = f' {ylabel}' if ylabel else ''
            # On the title line, clear of the ▲ marks at the top edge.
            ax.set_title(f'▲ above scale, up to {np.nanmax(values):.{0 if top > 5 else 2}f}{unit}',
                         loc='right', pad=18 if legend else 6, fontsize=7.5, color=MUTED)
        if reference is not None:
            value, text = reference
            ax.axhline(value, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(0.005, value + 0.02 * top, text, transform=ax.get_yaxis_transform(), fontsize=7.5,
                    color=INK_2, va='bottom')
        self.solid, self.marks = [], []
        for i, name in enumerate(names):
            self.solid.append(ax.plot([], [], color=SERIES[i], linewidth=2, label=finger_label(name))[0])
            # Clipped above the axes, but kept inside the window by hand (clip_on=False).
            self.marks.append((np.flatnonzero(over[:, i]),
                               ax.plot([], [], '^', color=SERIES[i], markersize=6, clip_on=False)[0]))
        self.cursor = ax.axvline(0, color=INK_2, linewidth=1)
        if legend:
            ax.legend(handles=self.solid, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8,
                      frameon=False, handlelength=1.4, columnspacing=1.0, labelcolor=INK_2, borderaxespad=0.1)
        self.direct = [ax.text(0, 0, '', fontsize=8, color=INK_2, va='center', clip_on=True)
                       for _ in names] if direct_labels else []

    def update(self, k):
        now = self.t[k]
        start = max(0.0, now - WINDOW_SECONDS)
        # A little room right of the cursor for the direct labels; early frames show the first window.
        self.ax.set_xlim(start, max(now, WINDOW_SECONDS) + LABEL_ROOM * WINDOW_SECONDS)
        first = int(np.searchsorted(self.t, start - 1 / 25))
        for i, line in enumerate(self.solid):
            line.set_data(self.t[first:k + 1], self.values[first:k + 1, i])
        for hits, mark in self.marks:
            shown = hits[(hits >= first) & (hits <= k)]
            mark.set_data(self.t[shown], np.full(len(shown), self.top))
        self.cursor.set_xdata([now, now])
        # Direct labels at the cursor, nudged apart so near-equal readings stay legible.
        top = self.top
        shown = [i for i in range(len(self.direct)) if self.values[k, i] > 0.02 * top]
        placed = {}
        for i in sorted(shown, key=lambda i: self.values[k, i]):
            y = min(self.values[k, i], 0.9 * top)
            if placed:
                y = max(y, max(placed.values()) + 0.075 * top)
            placed[i] = y
        for i, text in enumerate(self.direct):
            text.set_position((now + 0.01 * WINDOW_SECONDS, placed.get(i, 0)))
            text.set_text(self.fmt.format(self.values[k, i]) if i in placed else '')


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(PANEL[0] / 100, PANEL[1] / 100), dpi=100, facecolor=SURFACE)
    return fig, plt


def canvas_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def header(fig, title, subtitle):
    fig.text(0.03, 0.972, title, color=INK, fontsize=14, fontweight='bold', va='top')
    fig.text(0.03, 0.937, subtitle, color=INK_2, fontsize=9, va='top')
    return fig.text(0.97, 0.972, '', color=INK_2, fontsize=11, va='top', ha='right', family='monospace')


def clock(text, t, k):
    text.set_text(f't = {t[k]:5.2f} s')


def sequential_cmap(ramp=SEQUENTIAL):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', ramp)
    cmap.set_bad(SURFACE)
    return cmap


# ---------------------------------------------------------------------------- option 1
def contact_panel(meta, data, t):
    fig, plt = figure()
    names = [f['name'] for f in meta['fingers']]
    now = header(fig, 'Option 1 · Contact sensor',
                 'IsaacLab ContactSensor on each finger link, filtered against the 8 blocks (PhysX contact forces)')
    # The sensor covers the whole link: side or back hits have no gripping-face component but still
    # carry normal and friction force, so plot magnitudes over all faces.
    normal = np.linalg.norm(data['contact_block_force_w'].sum(-2), axis=-1)
    # PhysX friction, in the contact plane; NaN for blocks that are not touching.
    shear = np.linalg.norm(np.nan_to_num(data['contact_friction_w']).sum(-2), axis=-1)
    # Coulomb friction: |F_t| <= mu |F_n| while the block sticks; at mu it slides.
    ratio = np.where(normal > 1.0, shear / np.maximum(normal, 1e-9), np.nan)
    grid = fig.add_gridspec(4, 1, left=0.09, right=0.95, top=0.865, bottom=0.035, hspace=0.72,
                            height_ratios=[1, 1, 0.9, 0.62])
    series = [TimeSeries(fig.add_subplot(grid[0]), t, normal, names, 'N', 'Normal force on the finger (any face; grip = gripping face)'),
              TimeSeries(fig.add_subplot(grid[1]), t, shear, names, 'N', 'Tangential (friction) force on the finger',
                         legend=False),
              TimeSeries(fig.add_subplot(grid[2]), t, ratio, names, '', 'Friction / normal (shown when normal > 1 N)',
                         legend=False, top=0.6, fmt='{:.2f}',
                         reference=(FRICTION_MU, f'μ = {FRICTION_MU} · on the line = sliding, below = sticking'))]
    table = fig.add_subplot(grid[3]); table.axis('off')
    columns = (0.0, 0.25, 0.39, 0.53, 0.66)
    for x, text in zip(columns, ('finger', 'normal', 'friction', 'F/N', 'touching (N)')):
        table.text(x, 1.02, text, color=MUTED, fontsize=8, transform=table.transAxes)
    cells = []
    for i, name in enumerate(names):
        y = 0.76 - i * 0.25
        table.plot([0.005], [y + 0.03], 's', color=SERIES[i], markersize=8, transform=table.transAxes)
        table.text(0.03, y, finger_label(name), color=INK, fontsize=9, transform=table.transAxes)
        cells.append([table.text(x, y, '', color=INK_2, fontsize=9, transform=table.transAxes,
                                 family='monospace') for x in columns[1:]])
    labels = list(meta['blocks'])
    per_block = np.linalg.norm(data['contact_block_force_w'], axis=-1)   # (T, 4, B)

    def update(k):
        clock(now, t, k)
        for s in series:
            s.update(k)
        for i, row in enumerate(cells):
            touching = [f'{labels[b]} {per_block[k, i, b]:.1f}' for b in np.argsort(-per_block[k, i])
                        if per_block[k, i, b] > 0.05][:2]
            row[0].set_text(f'{normal[k, i]:6.2f} N')
            row[1].set_text(f'{shear[k, i]:6.2f} N')
            r = ratio[k, i]
            row[2].set_text('—' if np.isnan(r) else f"{r:4.2f}{' slip' if r >= 0.96 * FRICTION_MU else ''}")
            row[3].set_text(', '.join(touching) or '—')
        return canvas_rgb(fig)
    return update, fig


# ---------------------------------------------------------------------------- option 2
def taxel_panel(meta, data, t):
    fig, plt = figure()
    names = [f['name'] for f in meta['fingers']]
    tax = meta['taxel']
    u = np.array(tax['u_centres']) * 1000; v = np.array(tax['v_centres']) * 1000
    du, dv = abs(u[0] - u[1]), abs(v[1] - v[0])
    now = header(fig, 'Option 2 · Taxel grid',
                 f"PhysX contact patches, distal {u[0]-u[-1]+du:.0f} mm of each finger face "
                 f"({tax['rows']}×{tax['cols']} taxels, {du:.1f}×{dv:.1f} mm) · dots = contacts · arrows = friction")
    pressure, shear = data['taxel_pressure'], data['taxel_shear']
    active = pressure[pressure > 0]
    vmax = max(float(np.percentile(active, 99)) if active.size else 1.0, 1e-3)
    extent = (v[0] - dv / 2, v[-1] + dv / 2, u[-1] - du / 2, u[0] + du / 2)
    grid = fig.add_gridspec(2, 1, left=0.07, right=0.95, top=0.95, bottom=0.07, hspace=0.22, height_ratios=[1.45, 1])
    maps = grid[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.07], wspace=0.28)
    cmap = sequential_cmap()
    images, arrows, dots = [], [], []
    V, U = np.meshgrid(v, u)
    norms = np.linalg.norm(shear, axis=-1)[pressure > 0.05 * vmax]
    shear_scale = max(float(np.percentile(norms, 99)) if norms.size else 1e-3, 1e-4)
    for i, name in enumerate(names):
        ax = fig.add_subplot(maps[0, i])
        image_axes(ax, finger_label(name))
        images.append(ax.imshow(pressure[0, i], cmap=cmap, vmin=0, vmax=vmax, extent=extent,
                                origin='upper', interpolation='nearest', aspect='equal'))
        arrows.append(ax.quiver(V, U, np.zeros_like(V), np.zeros_like(U), color=INK, angles='xy',
                                scale_units='xy', scale=shear_scale / (2.5 * du), width=0.014, headwidth=3.5))
        dots.append(ax.plot([], [], 'o', color=SERIES[i], markersize=4, markeredgecolor=INK, markeredgewidth=0.6)[0])
        ax.plot([extent[0], extent[1]], [extent[3], extent[3]], color=SERIES[i], linewidth=4, clip_on=False)
        ax.set_xlabel('width (mm)', color=MUTED, fontsize=7); ax.set_xticks([-20, 0, 20])
        ax.set_yticks([30, 50, 70]); ax.tick_params(colors=MUTED, labelsize=7)
        if i == 0:
            ax.set_ylabel('from finger root (mm) · tip at top', color=MUTED, fontsize=7)
        else:
            ax.set_yticklabels([])
    cax = fig.add_subplot(maps[0, 4])
    bar = fig.colorbar(images[0], cax=cax, extend='max')
    bar.set_label('normal force per taxel (N)', color=INK_2, fontsize=8)
    bar.ax.tick_params(colors=MUTED, labelsize=7); bar.outline.set_edgecolor(AXIS)
    total = pressure.sum(axis=(2, 3))
    series = TimeSeries(fig.add_subplot(grid[1]), t, total, names, 'N', 'Total normal force on the taxels')
    offsets = data['taxel_point_offsets']; pts = data['taxel_points_uv'] * 1000

    def update(k):
        clock(now, t, k)
        for i in range(len(names)):
            images[i].set_data(pressure[k, i])
            quiet = pressure[k, i] < 0.05 * vmax
            arrows[i].set_UVC(np.ma.masked_where(quiet, shear[k, i, ..., 1]),
                              np.ma.masked_where(quiet, shear[k, i, ..., 0]))
            j = k * len(names) + i
            p = pts[offsets[j]:offsets[j + 1]]
            dots[i].set_data(p[:, 1], p[:, 0])
        series.update(k)
        return canvas_rgb(fig)
    return update, fig


# ---------------------------------------------------------------------------- option 3
def pool_mask(mask, factor):
    n, h, w = mask.shape
    return mask[:, :h // factor * factor, :w // factor * factor].reshape(n, h // factor, factor, w // factor, factor).all((2, 4))


def gelsight_panel(meta, data, t, gel_frames):
    fig, plt = figure()
    names = [f['name'] for f in meta['fingers']]
    gel = meta['gel']
    now = header(fig, 'Option 3 · TacSL GelSight',
                 f"Virtual {gel['thickness']*1000:.0f} mm gel on each fingertip ({gel['rows']*gel['mm_per_pixel']:.0f}×"
                 f"{gel['cols']*gel['mm_per_pixel']:.0f} mm): TacSL R1.5 image, indentation, TacSL force field")
    fr, fc = gel['field']
    mask = data['gel_mask']
    step_r, step_c = mask.shape[1] // fr, mask.shape[2] // fc
    field_mask = mask[:, step_r // 2::step_r, step_c // 2::step_c][:, :fr, :fc]  # gel present at each point
    field_n = data['gel_field_normal'] * field_mask[None]
    field_s = data['gel_field_shear'] * field_mask[None, ..., None]
    height = data['gel_height'].astype(np.float32) * 1000        # mm, 4x4 max-pooled
    pooled = pool_mask(mask, mask.shape[1] // height.shape[2])
    height = np.where(pooled[None], height, np.nan)
    depth_max = max(float(np.nanpercentile(height[height > 0], 99)) if (height > 0).any() else 1.0, 0.05)
    vmax = max(float(np.percentile(field_n[field_n > 0], 99.5)) if (field_n > 0).any() else 0.1, 1e-4)
    norms = np.linalg.norm(field_s, axis=-1)[field_n > 0]
    shear_scale = max(float(np.percentile(norms, 99)) if norms.size else 1e-4, 1e-5)
    grid = fig.add_gridspec(2, 1, left=0.03, right=0.95, top=0.855, bottom=0.07, hspace=0.25, height_ratios=[2.5, 1])
    cells = grid[0].subgridspec(2, 8, width_ratios=[1, 1, 1, 0.12, 1, 1, 1, 0.09], wspace=0.06, hspace=0.20)
    depth_cmap, force_cmap = sequential_cmap(SEQUENTIAL_2), sequential_cmap()
    rgb_images, depths, fields, arrows = [], [], [], []
    X, Y = np.meshgrid(np.arange(fc), np.arange(fr))
    for i, name in enumerate(names):
        r, c = divmod(i, 2)
        col = c * 4
        ax = fig.add_subplot(cells[r, col])
        image_axes(ax, 'TacSL image')
        ax.text(0.0, 1.16, finger_label(name), transform=ax.transAxes, color=INK, fontsize=9, fontweight='bold')
        rgb_images.append(ax.imshow(np.zeros((gel['rows'], gel['cols'], 3), np.uint8), aspect='equal'))
        ax.plot([0, gel['cols'] - 1], [0, 0], color=SERIES[i], linewidth=4, clip_on=False)
        ax = fig.add_subplot(cells[r, col + 1])
        image_axes(ax, 'indentation')
        depths.append(ax.imshow(height[0, i], cmap=depth_cmap, vmin=0, vmax=depth_max, aspect='equal',
                                interpolation='nearest'))
        ax = fig.add_subplot(cells[r, col + 2])
        image_axes(ax, 'force field')
        fields.append(ax.imshow(field_n[0, i], cmap=force_cmap, vmin=0, vmax=vmax, aspect='equal', interpolation='nearest'))
        arrows.append(ax.quiver(X, Y, np.zeros_like(X, float), np.zeros_like(Y, float), color=INK, angles='xy',
                                scale_units='xy', scale=shear_scale / 1.5, width=0.02, headwidth=3.5))
    bars = cells[:, 7].subgridspec(2, 1, hspace=0.35)
    for index, (image, label) in enumerate(((depths[0], 'indentation (mm)'), (fields[0], 'normal force per point (N)'))):
        bar = fig.colorbar(image, cax=fig.add_subplot(bars[index]))
        bar.set_label(label, color=INK_2, fontsize=8)
        bar.ax.tick_params(colors=MUTED, labelsize=7); bar.outline.set_edgecolor(AXIS)
    total = np.clip(field_n, 0, None).sum(axis=(2, 3))
    series = TimeSeries(fig.add_subplot(grid[1]), t, total, names, 'N',
                        'TacSL force-field total · tracks contact area (rigid fingers fill the 1 mm gel)')
    dim = np.where(mask[..., None], 1.0, 0.25)

    def update(k):
        clock(now, t, k)
        frame = next(gel_frames)
        tiles = np.split(frame[:gel['rows'], :gel['cols'] * len(names)], len(names), axis=1)
        for i in range(len(names)):
            rgb_images[i].set_data((tiles[i] * dim[i]).astype(np.uint8))
            depths[i].set_data(height[k, i])
            fields[i].set_data(field_n[k, i])
            quiet = field_n[k, i] <= 0
            # Field rows run tip -> root like the image; image x = width (v), y = length (u).
            arrows[i].set_UVC(np.ma.masked_where(quiet, field_s[k, i, ..., 1]),
                              np.ma.masked_where(quiet, -field_s[k, i, ..., 0]))
        series.update(k)
        return canvas_rgb(fig)
    return update, fig


def gel_frame_iter(path):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path))
    for frame in reader:
        yield frame
    reader.close()


def render(archive, output, models=('contact', 'taxel', 'gelsight'), limit=None):
    import imageio.v2 as imageio
    sim, meta, data = load(archive)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dt = json.loads((sim/'reset.json').read_text())['metadata']['control_dt'] if (sim/'reset.json').exists() else 0.04
    total = len(data['step'])
    steps = total if limit is None else min(limit, total)
    if steps < total:  # Preview: every per-frame array, including the flattened contact-point index.
        fingers = len(meta['fingers'])
        data = {key: value[:steps] if value.ndim and len(value) == total else value for key, value in data.items()}
        data['taxel_point_offsets'] = data['taxel_point_offsets'][:steps * fingers + 1]
    t = data['step'] * dt
    written = []
    for model in models:
        if model == 'gelsight':
            update, fig = gelsight_panel(meta, data, t, gel_frame_iter(sim/'tactile'/'gelsight.mp4'))
        else:
            update, fig = (contact_panel if model == 'contact' else taxel_panel)(meta, data, t)
        path = output/f'{model}.mp4'
        writer = imageio.get_writer(str(path), fps=round(1 / dt), codec='libx264', pixelformat='yuv420p',
                                    quality=None, macro_block_size=2, output_params=['-crf', '18', '-movflags', '+faststart'])
        cameras = camera_frames(sim/'sensors.mp4')
        for k in range(steps):
            left = label_cameras(next(cameras))
            right = update(k)
            if right.shape[:2] != (PANEL[1], PANEL[0]):
                import cv2
                right = cv2.resize(right, PANEL, interpolation=cv2.INTER_AREA)
            writer.append_data(np.concatenate([left, right], axis=1))
        writer.close()
        import matplotlib.pyplot as plt
        plt.close(fig)
        written.append(str(path))
        print(json.dumps(dict(event='video', model=model, path=str(path), frames=steps)), flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='results/<experiment>/<case>/attempt_N')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['contact', 'taxel', 'gelsight'])
    parser.add_argument('--limit', type=int, help='Render only the first N frames (preview)')
    args = parser.parse_args()
    render(args.archive, args.output, args.models, args.limit)


if __name__ == '__main__':
    main()
