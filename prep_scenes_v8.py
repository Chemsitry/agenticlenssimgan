"""
prep_scenes_v8.py — Extract 630x630 real JWST scenes, each centered on a
detected elliptical galaxy from the DR0.5 mosaic.

Architectural change: we no longer stitch a lens-stamp onto a separate
background-cutout. Each image is ONE real JWST observation with a galaxy
already in it. During simulation we just add a lensed source on top.

Output: prepped_mosaic_630/scenes_v8/ (one .npy per band, shape (N, 630, 630))
Positions come from prepped_mosaic_630/lenses_v7/lens_info.json — galaxies
that already passed the concentration/isolation cuts.
"""

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
MOSAIC_DIR = Path('raw_data/1727_mosaic')
PREPPED_DIR = Path('prepped_mosaic_630')
SRC_INFO = PREPPED_DIR / 'lenses_v7' / 'lens_info.json'
OUT_DIR = PREPPED_DIR / 'scenes_v8'

SCENE_SIZE = 630  # matches simulate_v8 IMAGE_SIZE
PEAK_MIN = 1000   # F444W peak (sim units) to keep — ensures visible lens
COMPACTNESS_MAX = 0.35  # rejects stars — stars have diffraction spikes
                        # and concentrate most flux within r<3 px of center;
                        # galaxies spread flux out over larger radii.
                        # (measured as sum(r<3) / sum(r<15) in F444W)

sum_to_flux = 6.501853565914121


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SRC_INFO) as f:
        src_info = json.load(f)
    with open(PREPPED_DIR / 'band_info.json') as f:
        band_info = json.load(f)

    positions = [(int(l['y']), int(l['x'])) for l in src_info['lenses']]
    n_total = len(positions)
    print(f'Extracting {n_total} candidate scenes, size={SCENE_SIZE}')

    # Pre-allocate
    raw = {b: np.zeros((n_total, SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
           for b in BANDS}
    valid = np.ones(n_total, dtype=bool)
    half = SCENE_SIZE // 2

    for band in BANDS:
        info = band_info[band]
        m2s = info['pixar_sr'] * 1e15 * sum_to_flux
        bg_median = info['bg_median']
        fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
        print(f'  {band}: bg_median={bg_median:.4e}  m2s={m2s:.2f}')
        with fits.open(str(fits_files[0]), memmap=True) as hdul:
            sci = hdul[1].data
            ny, nx = sci.shape
            for i, (y, x) in enumerate(positions):
                y0, x0 = y - half, x - half
                y1, x1 = y0 + SCENE_SIZE, x0 + SCENE_SIZE
                if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                    valid[i] = False
                    continue
                s = sci[y0:y1, x0:x1].astype(np.float32)
                if not np.all(np.isfinite(s)):
                    bad = ~np.isfinite(s)
                    if bad.mean() > 0.02:   # > 2% bad pixels → skip
                        valid[i] = False
                        continue
                    s[bad] = 0.0
                s = (s - bg_median) * m2s
                raw[band][i] = s

                if (i + 1) % 50 == 0 and band == BANDS[0]:
                    print(f'    {i + 1}/{n_total}')

    # Brightness filter on F444W peak at the scene center (a 40x40 box)
    c = half
    f444 = raw['F444W']
    center_peaks = f444[:, c - 20:c + 20, c - 20:c + 20].max(axis=(1, 2))
    bright = center_peaks > PEAK_MIN

    # Compactness filter — rejects stars (with diffraction spikes).
    # Stars have most flux concentrated within the PSF core (r<3 px); galaxies
    # spread flux over larger radii. Compactness = sum(r<3) / sum(r<15).
    yy, xx = np.mgrid[:SCENE_SIZE, :SCENE_SIZE]
    rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    mask_small = rr < 3
    mask_medium = rr < 15
    compactness = np.zeros(n_total, dtype=np.float32)
    for i in range(n_total):
        if not valid[i]:
            continue
        s = f444[i]
        f_small = s[mask_small].sum()
        f_medium = s[mask_medium].sum()
        compactness[i] = f_small / f_medium if f_medium > 0 else 1.0
    not_star = compactness < COMPACTNESS_MAX

    keep = valid & bright & not_star
    n_keep = int(keep.sum())
    print(f'\n  bright: {int(bright.sum())}/{n_total}   not_star: {int(not_star.sum())}/{n_total}')
    print(f'  final: {n_keep}/{n_total} scenes pass all filters')

    for b in BANDS:
        arr = raw[b][keep]
        path = OUT_DIR / f'scenes_{b}.npy'
        np.save(str(path), arr)
        print(f'  {b}: {arr.shape}  {arr.nbytes/1e6:.1f} MB -> {path}')

    kept_positions = [p for p, k in zip(src_info['lenses'], keep) if k]
    info_out = {
        'n_scenes': n_keep,
        'scene_size': SCENE_SIZE,
        'pixel_scale': 0.03,
        'fov_arcsec': SCENE_SIZE * 0.03,
        'brightness_filter': f'central F444W peak > {PEAK_MIN} sim units',
        'compactness_filter': f'F444W sum(r<3)/sum(r<15) < {COMPACTNESS_MAX} (rejects stars with diffraction spikes)',
        'derived_from': 'DR0.5 mosaic cutouts centered on lenses_v7 positions',
        'notes': 'One-image-per-sample approach: scene IS the real JWST cutout '
                 'with a real galaxy near center. Simulation just adds a '
                 'lensed source on top — single noise realization, no stitching.',
        'positions': kept_positions,
    }
    with open(OUT_DIR / 'scene_info.json', 'w') as f:
        json.dump(info_out, f, indent=2)
    print(f'\nDone — {n_keep} real-scene cutouts in {OUT_DIR}/')


if __name__ == '__main__':
    main()
