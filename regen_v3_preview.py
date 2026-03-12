"""Regenerate v3 preview with improved RGB visibility."""
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.ndimage import zoom
from pathlib import Path

OUT_DIR = Path('output/v3')
PREPPED_DIR = Path('prepped_cosmicwebb')

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
SW_BANDS = {'F115W', 'F150W'}

with open(PREPPED_DIR / 'band_info.json') as f:
    BAND_INFO = json.load(f)

sum_to_flux = 6.501853565914121
BAND_CONFIG = {}
for band in BANDS:
    info = BAND_INFO[band]
    pixar_sr = info['pixar_sr']
    mjysr_to_sim = pixar_sr * 1e15 * sum_to_flux
    BAND_CONFIG[band] = {
        'pixels': 125 if band in SW_BANDS else 63,
        'mjysr_to_sim': mjysr_to_sim,
    }

# Load saved data
images = {band: np.load(str(OUT_DIR / f'images_{band}.npy')) for band in BANDS}
sources = {band: np.load(str(OUT_DIR / f'sources_{band}.npy')) for band in BANDS}
labels = np.load(str(OUT_DIR / 'lensed.npy'))
theta_Es = np.load(str(OUT_DIR / 'theta_Es.npy'))
z_lenses = np.load(str(OUT_DIR / 'z_lens.npy'))
z_sources = np.load(str(OUT_DIR / 'z_source.npy'))

order_l = np.where(labels == 1)[0]
order_nl = np.where(labels == 0)[0]
n_show = min(5, len(order_l), len(order_nl))


def make_rgb(r, g, b, stretch_q=10):
    target_shape = r.shape
    if g.shape != target_shape:
        g = zoom(g, np.array(target_shape) / np.array(g.shape), order=1)
    if b.shape != target_shape:
        b = zoom(b, np.array(target_shape) / np.array(b.shape), order=1)
    r = np.nan_to_num(np.clip(r, 0, None), nan=0.0)
    g = np.nan_to_num(np.clip(g, 0, None), nan=0.0)
    b = np.nan_to_num(np.clip(b, 0, None), nan=0.0)
    lum = r + g + b
    lum_max = np.max(lum)
    if lum_max <= 0:
        return np.zeros((*target_shape, 3))
    lum_stretched = np.arcsinh(lum * stretch_q)
    lum_stretched /= np.max(lum_stretched)
    rgb = np.stack([r, g, b], axis=-1)
    lum_safe = np.where(lum > 0, lum, 1.0)
    for ch in range(3):
        rgb[:, :, ch] = rgb[:, :, ch] / lum_safe * lum_stretched
    return np.clip(rgb, 0, 1)


def show(ax, img, title, cmap='gray', stretch='asinh', lw_frac=0.001):
    d = np.nan_to_num(np.clip(img, 0, None), nan=0.0)
    vmin = float(np.percentile(d, 0.5))
    vmax = float(np.percentile(d, 99.9))
    if vmax <= vmin:
        vmax = vmin + 1.0
    if stretch == 'asinh':
        lw = max((vmax - vmin) * lw_frac, 1e-6)
        norm = AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)
        im = ax.imshow(d, norm=norm, origin='lower', cmap=cmap)
    else:
        im = ax.imshow(d, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=7)
    ax.axis('off')


def show_rgb(ax, rgb, title):
    ax.imshow(rgb, origin='lower')
    ax.set_title(title, fontsize=7)
    ax.axis('off')


# Normalization factors for pixel solid angle correction
norm = {band: BAND_CONFIG[band]['mjysr_to_sim'] for band in BANDS}

# 10 rows: 4 bands + RGB lensed + RGB arcs + RGB lens-only + lens-sub + arc-only(asinh) + arc-only(inferno)
fig, axs = plt.subplots(10, n_show, figsize=(4*n_show, 42), dpi=100)

# Rows 0-3: individual bands (asinh)
for row, band in enumerate(BANDS):
    for col in range(n_show):
        idx = order_l[col]
        show(axs[row, col], images[band][idx],
             f'{band} θE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f} zs={z_sources[idx]:.2f}')
    axs[row, 0].set_ylabel(f'{band}\n(asinh)', fontsize=10)

# Row 4: RGB lensed (full image with background)
for col in range(n_show):
    idx = order_l[col]
    rgb = make_rgb(
        images['F277W'][idx] / norm['F277W'],
        images['F150W'][idx] / norm['F150W'],
        images['F115W'][idx] / norm['F115W'],
        stretch_q=20,
    )
    show_rgb(axs[4, col], rgb,
             f'RGB lensed θE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f}')
axs[4, 0].set_ylabel('RGB lensed\n(F277W/F150W/F115W)', fontsize=10)

# Row 5: RGB arcs only (no lens, no background)
for col in range(n_show):
    idx = order_l[col]
    rgb = make_rgb(
        sources['F277W'][idx] / norm['F277W'],
        sources['F150W'][idx] / norm['F150W'],
        sources['F115W'][idx] / norm['F115W'],
        stretch_q=50,
    )
    show_rgb(axs[5, col], rgb,
             f'RGB arcs θE={theta_Es[idx]:.2f}" zs={z_sources[idx]:.2f}')
axs[5, 0].set_ylabel('RGB arcs\n(source only)', fontsize=10)

# Row 6: RGB lens-only (full image minus arcs, includes background)
for col in range(n_show):
    idx = order_l[col]
    lens_r = np.clip(images['F277W'][idx] - sources['F277W'][idx], 0, None) / norm['F277W']
    lens_g = np.clip(images['F150W'][idx] - sources['F150W'][idx], 0, None) / norm['F150W']
    lens_b = np.clip(images['F115W'][idx] - sources['F115W'][idx], 0, None) / norm['F115W']
    rgb = make_rgb(lens_r, lens_g, lens_b, stretch_q=20)
    show_rgb(axs[6, col], rgb,
             f'RGB lens-only zl={z_lenses[idx]:.2f}')
axs[6, 0].set_ylabel('RGB lens\n(no arcs)', fontsize=10)

# Row 7: RGB arcs boosted (arc peak matched to lens peak)
for col in range(n_show):
    idx = order_l[col]
    arc_r = sources['F277W'][idx] / norm['F277W']
    arc_g = sources['F150W'][idx] / norm['F150W']
    arc_b = sources['F115W'][idx] / norm['F115W']
    arc_peak = max(np.max(np.abs(arc_r)), np.max(np.abs(arc_g)), np.max(np.abs(arc_b)), 1e-10)
    lens_r = np.clip(images['F277W'][idx] - sources['F277W'][idx], 0, None) / norm['F277W']
    lens_g = np.clip(images['F150W'][idx] - sources['F150W'][idx], 0, None) / norm['F150W']
    lens_b = np.clip(images['F115W'][idx] - sources['F115W'][idx], 0, None) / norm['F115W']
    lens_peak = max(np.max(np.abs(lens_r)), np.max(np.abs(lens_g)), np.max(np.abs(lens_b)), 1e-10)
    boost = lens_peak / arc_peak
    rgb = make_rgb(arc_r * boost, arc_g * boost, arc_b * boost, stretch_q=50)
    show_rgb(axs[7, col], rgb,
             f'RGB arcs boosted {boost:.0f}x')
axs[7, 0].set_ylabel('RGB arcs\n(boosted)', fontsize=10)

# Row 8: lens-subtracted F115W (asinh)
for col in range(n_show):
    idx = order_l[col]
    show(axs[8, col],
         sources['F115W'][idx] + np.median(images['F115W'][idx]),
         f'F115W lens-sub θE={theta_Es[idx]:.2f}"',
         stretch='asinh', lw_frac=0.01)
axs[8, 0].set_ylabel('F115W\nlens-sub', fontsize=10)

# Row 9: arc-only F115W (inferno)
for col in range(n_show):
    idx = order_l[col]
    show(axs[9, col], sources['F115W'][idx],
         f'arcs θE={theta_Es[idx]:.2f}"', cmap='inferno', stretch='linear')
axs[9, 0].set_ylabel('F115W\narc-only', fontsize=10)

plt.suptitle('COSMOS-Web v3 simulation — COWLS-calibrated (4 bands, 10 images)', fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'preview.png'), dpi=120, bbox_inches='tight')
print(f'Saved -> {OUT_DIR}/preview.png')
