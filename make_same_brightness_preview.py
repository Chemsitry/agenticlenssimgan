"""Generate preview_same_brightness.png with proper pixel solid angle normalization."""
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

OUT_DIR = Path('output/multiband')
PREPPED_DIR = Path('prepped_cosmicwebb')

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
SW_BANDS = {'F115W', 'F150W'}

with open(PREPPED_DIR / 'band_info.json') as f:
    BAND_INFO = json.load(f)

sum_to_flux = 6.501853565914121
BAND_CONFIG = {}
for band in BANDS:
    info = BAND_INFO[band]
    is_sw = band in SW_BANDS
    pixar_sr = info['pixar_sr']
    mjysr_to_sim = pixar_sr * 1e15 * sum_to_flux
    BAND_CONFIG[band] = {'mjysr_to_sim': mjysr_to_sim}

# Load saved data
sources = {band: np.load(str(OUT_DIR / f'sources_{band}.npy')) for band in BANDS}
images = {band: np.load(str(OUT_DIR / f'images_{band}.npy')) for band in BANDS}
labels = np.load(str(OUT_DIR / 'lensed.npy'))
theta_Es = np.load(str(OUT_DIR / 'theta_Es.npy'))
z_lenses = np.load(str(OUT_DIR / 'z_lens.npy'))
z_sources = np.load(str(OUT_DIR / 'z_source.npy'))

order_l = np.where(labels == 1)[0]
n_show = min(5, len(order_l))

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

fig, axs = plt.subplots(4, n_show, figsize=(4*n_show, 16), dpi=120)

for col in range(n_show):
    idx = order_l[col]
    tE = theta_Es[idx]
    zl = z_lenses[idx]
    zs = z_sources[idx]

    # Normalize by mjysr_to_sim to convert to MJy/sr surface brightness
    norm = {band: BAND_CONFIG[band]['mjysr_to_sim'] for band in BANDS}

    # Get arc and lens images in MJy/sr
    arc_r = sources['F277W'][idx] / norm['F277W']
    arc_g = sources['F150W'][idx] / norm['F150W']
    arc_b = sources['F115W'][idx] / norm['F115W']

    lens_r = (images['F277W'][idx] - sources['F277W'][idx]) / norm['F277W']
    lens_g = (images['F150W'][idx] - sources['F150W'][idx]) / norm['F150W']
    lens_b = (images['F115W'][idx] - sources['F115W'][idx]) / norm['F115W']

    # Row 0: RGB lensed (original relative brightness)
    rgb_full = make_rgb(
        images['F277W'][idx] / norm['F277W'],
        images['F150W'][idx] / norm['F150W'],
        images['F115W'][idx] / norm['F115W'],
    )
    axs[0, col].imshow(rgb_full, origin='lower')
    axs[0, col].set_title(f'RGB lensed\nθE={tE:.2f}" zl={zl:.2f} zs={zs:.2f}', fontsize=7)
    axs[0, col].axis('off')

    # Row 1: RGB arcs boosted so arc peak ≈ lens peak
    arc_peak = max(np.max(np.abs(arc_r)), np.max(np.abs(arc_g)), np.max(np.abs(arc_b)), 1e-10)
    lens_peak = max(np.max(np.abs(lens_r)), np.max(np.abs(lens_g)), np.max(np.abs(lens_b)), 1e-10)
    boost = lens_peak / arc_peak
    rgb_boosted = make_rgb(arc_r * boost, arc_g * boost, arc_b * boost, stretch_q=50)
    axs[1, col].imshow(rgb_boosted, origin='lower')
    axs[1, col].set_title(f'RGB arcs boosted {boost:.0f}x', fontsize=7)
    axs[1, col].axis('off')

    # Row 2: RGB arcs original
    rgb_arcs = make_rgb(arc_r, arc_g, arc_b, stretch_q=50)
    axs[2, col].imshow(rgb_arcs, origin='lower')
    axs[2, col].set_title(f'RGB arcs original', fontsize=7)
    axs[2, col].axis('off')

    # Row 3: RGB lens-only
    rgb_lens = make_rgb(
        np.clip(lens_r, 0, None),
        np.clip(lens_g, 0, None),
        np.clip(lens_b, 0, None),
    )
    axs[3, col].imshow(rgb_lens, origin='lower')
    axs[3, col].set_title(f'RGB lens-only', fontsize=7)
    axs[3, col].axis('off')

axs[0, 0].set_ylabel('RGB lensed', fontsize=10)
axs[1, 0].set_ylabel('RGB arcs\n(boosted)', fontsize=10)
axs[2, 0].set_ylabel('RGB arcs\n(original)', fontsize=10)
axs[3, 0].set_ylabel('RGB lens\nonly', fontsize=10)

plt.suptitle('Same-brightness sanity check — arcs should be blue, lens should be red', fontsize=11)
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'preview_same_brightness.png'), dpi=120, bbox_inches='tight')
print(f'Saved -> {OUT_DIR}/preview_same_brightness.png')
