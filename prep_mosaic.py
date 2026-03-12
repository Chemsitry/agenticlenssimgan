"""
prep_mosaic.py — Extract backgrounds and PSFs from full COSMOS-Web DR0.5 mosaics.

These are the deep coadded mosaics (36000x30000, 30mas/pix, ~6184s exposure)
covering the full 0.54 deg² COSMOS-Web survey. All 4 bands are resampled to
a common 30mas pixel grid.

Background patches are spatially matched across all bands — the same sky
coordinates are used for all 4 bands, so background galaxies are correlated.

Output: prepped_mosaic/<band>/backgrounds.npy, psf_median.npy
        prepped_mosaic/band_info.json

Usage:
    .venv/bin/python3 prep_mosaic.py
    .venv/bin/python3 prep_mosaic.py --size 224
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=125, help='Patch size in pixels (default: 125)')
args = parser.parse_args()

MOSAIC_DIR = Path('raw_data/1727_mosaic')
OUT_DIR = Path('prepped_mosaic') if args.size == 125 else Path(f'prepped_mosaic_{args.size}')
OUT_DIR.mkdir(exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']

# All bands are at 30mas pixel scale in the DR0.5 mosaics
PIXEL_SCALE = 0.03  # arcsec/pix
PATCH_SIZE = args.size
PSF_STAMP_SIZE = 63 # pixels
N_BACKGROUNDS = 2000
N_PSF_MAX = 200
VALID_FRAC = 0.98   # require 98% valid pixels per patch


# ── Phase 1: Open all mosaics, compute background stats ─────────────────

print('Phase 1: Opening mosaics and computing background statistics...')

band_data = {}
band_info = {}

for band in BANDS:
    print(f'\n  {band}:')
    band_dir = OUT_DIR / band
    band_dir.mkdir(exist_ok=True)

    fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
    if not fits_files:
        print(f'    ERROR: No FITS file found in {MOSAIC_DIR / band}')
        continue
    fits_path = fits_files[0]
    print(f'    File: {fits_path.name}')

    hdul = fits.open(str(fits_path), memmap=True)
    sci = hdul[1].data
    h = hdul[1].header

    ny, nx = sci.shape
    print(f'    Shape: {ny} x {nx}')

    pixar_sr = h.get('PIXAR_SR', 2.1154e-14)
    photmjsr = h.get('PHOTMJSR')
    xposure = hdul[0].header.get('EFFEXPTM', h.get('XPOSURE', 6184.0))
    print(f'    PIXAR_SR: {pixar_sr:.4e}  PHOTMJSR: {photmjsr}  XPOSURE: {xposure:.1f}s')

    rng = np.random.default_rng(42)
    n_sample = 500000
    ys = rng.integers(500, ny - 500, size=n_sample)
    xs = rng.integers(500, nx - 500, size=n_sample)
    sample = sci[ys, xs]
    valid = np.isfinite(sample) & (sample != 0)
    sample = sample[valid]
    bg_mean, bg_median, bg_std = sigma_clipped_stats(sample, sigma=3, maxiters=5)
    print(f'    Background: mean={bg_mean:.6f}  median={bg_median:.6f}  std={bg_std:.6f} MJy/sr')

    band_data[band] = {
        'sci': sci, 'hdul': hdul,
        'ny': ny, 'nx': nx,
        'pixar_sr': pixar_sr, 'photmjsr': photmjsr, 'xposure': xposure,
        'bg_mean': bg_mean, 'bg_median': bg_median, 'bg_std': bg_std,
    }


# ── Phase 2: Extract spatially matched background patches ───────────────
# Same pixel coordinates across all bands = same sky position (common 30mas grid)

ref = band_data[BANDS[0]]
ny, nx = ref['ny'], ref['nx']
half = PATCH_SIZE // 2

print(f'\nPhase 2: Extracting {N_BACKGROUNDS} spatially matched {PATCH_SIZE}x{PATCH_SIZE} patches...')
print(f'  (same sky position across all {len(BANDS)} bands)')

patches = {band: [] for band in BANDS}
rng = np.random.default_rng(42)
attempts = 0
max_attempts = N_BACKGROUNDS * 20

while len(patches[BANDS[0]]) < N_BACKGROUNDS and attempts < max_attempts:
    attempts += 1
    cy = int(rng.integers(half + 100, ny - half - 100))
    cx = int(rng.integers(half + 100, nx - half - 100))
    y0, x0 = cy - half, cx - half

    # Check validity in ALL bands at this location
    all_valid = True
    for band in BANDS:
        patch = band_data[band]['sci'][y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            all_valid = False
            break
        valid_mask = np.isfinite(patch) & (patch != 0)
        if valid_mask.mean() < VALID_FRAC:
            all_valid = False
            break

    if not all_valid:
        continue

    # Extract from all bands at the same sky location
    for band in BANDS:
        patch = band_data[band]['sci'][y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
        valid_mask = np.isfinite(patch) & (patch != 0)
        patch_clean = patch - band_data[band]['bg_median']
        patch_clean = np.where(valid_mask, patch_clean, 0.0)
        patches[band].append(patch_clean.astype(np.float32))

    n_done = len(patches[BANDS[0]])
    if n_done % 500 == 0:
        print(f'    {n_done}/{N_BACKGROUNDS} patches ({attempts} attempts)')

n_patches = len(patches[BANDS[0]])
print(f'  Got {n_patches} matched patches in {attempts} attempts')

for band in BANDS:
    arr = np.array(patches[band])
    band_dir = OUT_DIR / band
    np.save(str(band_dir / 'backgrounds.npy'), arr)
    print(f'  {band}: {arr.shape} -> {band_dir}/backgrounds.npy  {arr.nbytes/1e6:.1f} MB')


# ── Phase 3: Extract PSF stars per band ─────────────────────────────────
# PSFs are band-dependent, so extracted independently per band.

print(f'\nPhase 3: Extracting PSF stars per band...')
half_psf = PSF_STAMP_SIZE // 2

for band in BANDS:
    print(f'\n  {band}:')
    sci = band_data[band]['sci']
    ny_b, nx_b = band_data[band]['ny'], band_data[band]['nx']
    bg_median = band_data[band]['bg_median']
    bg_std = band_data[band]['bg_std']
    band_dir = OUT_DIR / band

    bright_threshold = bg_median + 20 * bg_std
    star_candidates = []

    tile_size = 4000
    for ty in range(500, ny_b - 500, tile_size):
        for tx in range(500, nx_b - 500, tile_size):
            ty_end = min(ty + tile_size, ny_b - 500)
            tx_end = min(tx + tile_size, nx_b - 500)
            tile = sci[ty:ty_end, tx:tx_end]

            bright = np.where(tile > bright_threshold)
            if len(bright[0]) == 0:
                continue

            for i in range(len(bright[0])):
                py = bright[0][i] + ty
                px = bright[1][i] + tx

                region = sci[py-2:py+3, px-2:px+3]
                if region.shape != (5, 5):
                    continue
                if not np.isfinite(region).all():
                    continue
                if sci[py, px] != np.max(region):
                    continue

                star_candidates.append((py, px, float(sci[py, px])))

    print(f'    Found {len(star_candidates)} bright local maxima')

    star_candidates.sort(key=lambda x: -x[2])
    stars_used = []
    psf_stamps = []

    for cy, cx, peak in star_candidates:
        if len(psf_stamps) >= N_PSF_MAX:
            break

        too_close = False
        for sy, sx in stars_used:
            if abs(cy - sy) < 50 and abs(cx - sx) < 50:
                too_close = True
                break
        if too_close:
            continue

        stamp = sci[cy - half_psf:cy + half_psf + 1,
                     cx - half_psf:cx + half_psf + 1].copy()
        if stamp.shape != (PSF_STAMP_SIZE, PSF_STAMP_SIZE):
            continue
        if not np.isfinite(stamp).all():
            continue

        yy, xx = np.mgrid[:PSF_STAMP_SIZE, :PSF_STAMP_SIZE]
        r = np.sqrt((yy - half_psf)**2 + (xx - half_psf)**2)
        stamp_sub = stamp - bg_median
        total_flux = np.sum(stamp_sub)
        core_flux = np.sum(stamp_sub[r < 10])
        if total_flux <= 0 or core_flux / total_flux < 0.5:
            continue

        stamp_norm = stamp_sub / total_flux
        psf_stamps.append(stamp_norm.astype(np.float32))
        stars_used.append((cy, cx))

    print(f'    Selected {len(psf_stamps)} PSF stars')

    if len(psf_stamps) >= 3:
        psf_stamps = np.array(psf_stamps)
        np.save(str(band_dir / 'psf_stars.npy'), psf_stamps)

        psf_median = np.median(psf_stamps, axis=0).astype(np.float64)
        psf_median = np.clip(psf_median, 0, None)
        psf_median /= psf_median.sum()
        np.save(str(band_dir / 'psf_median.npy'), psf_median)
        print(f'    Saved -> {band_dir}/psf_stars.npy ({len(psf_stamps)} stamps)')
        print(f'    Saved -> {band_dir}/psf_median.npy (sum={psf_median.sum():.4f})')
    else:
        print(f'    WARNING: Only {len(psf_stamps)} stars found, skipping PSF')

    band_info[band] = {
        'pixar_sr': band_data[band]['pixar_sr'],
        'photmjsr': band_data[band]['photmjsr'],
        'xposure': band_data[band]['xposure'],
        'pixel_scale': PIXEL_SCALE,
        'bg_median': float(band_data[band]['bg_median']),
        'bg_std': float(band_data[band]['bg_std']),
        'n_backgrounds': n_patches,
        'n_psf_stars': len(psf_stamps),
        'mosaic_shape': [band_data[band]['ny'], band_data[band]['nx']],
    }


# ── Cleanup ─────────────────────────────────────────────────────────────

for band in BANDS:
    band_data[band]['hdul'].close()

with open(OUT_DIR / 'band_info.json', 'w') as f:
    json.dump(band_info, f, indent=2)
print(f'\nSaved -> {OUT_DIR}/band_info.json')

print('\nDone! Summary:')
for band in BANDS:
    info = band_info.get(band, {})
    print(f'  {band}: {info.get("n_backgrounds", 0)} backgrounds, '
          f'{info.get("n_psf_stars", 0)} PSF stars, '
          f'XPOSURE={info.get("xposure", 0):.0f}s')
print(f'  Background patches are spatially matched across all bands.')
