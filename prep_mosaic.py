"""
prep_mosaic.py — Extract backgrounds and PSFs from full COSMOS-Web DR0.5 mosaics.

These are the deep coadded mosaics (36000x30000, 30mas/pix, ~6184s exposure)
covering the full 0.54 deg² COSMOS-Web survey. All 4 bands are resampled to
a common 30mas pixel grid.

Output: prepped_mosaic/<band>/backgrounds.npy, psf_median.npy
        prepped_mosaic/band_info.json

Usage:
    .venv/bin/python3 prep_mosaic.py
"""

import json
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

MOSAIC_DIR = Path('raw_data/1727_mosaic')
OUT_DIR = Path('prepped_mosaic')
OUT_DIR.mkdir(exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']

# All bands are at 30mas pixel scale in the DR0.5 mosaics
PIXEL_SCALE = 0.03  # arcsec/pix
PATCH_SIZE = 125    # pixels (~3.75" FoV)
PSF_STAMP_SIZE = 63 # pixels
N_BACKGROUNDS = 2000
N_PSF_MAX = 200
VALID_FRAC = 0.98   # require 98% valid pixels per patch

# ── Process each band ────────────────────────────────────────────────────

band_info = {}

for band in BANDS:
    print(f'\n{"="*60}')
    print(f'Processing {band}')
    print(f'{"="*60}')

    band_dir = OUT_DIR / band
    band_dir.mkdir(exist_ok=True)

    # Find the FITS file
    fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
    if not fits_files:
        print(f'  ERROR: No FITS file found in {MOSAIC_DIR / band}')
        continue
    fits_path = fits_files[0]
    print(f'  File: {fits_path.name}')

    # Open with memmap to avoid loading entire 48GB into RAM
    hdul = fits.open(str(fits_path), memmap=True)
    sci = hdul[1].data   # (36000, 30000) float32
    wht = hdul[4].data   # weight map
    h = hdul[1].header

    ny, nx = sci.shape
    print(f'  Shape: {ny} x {nx}')

    pixar_sr = h.get('PIXAR_SR', 2.1154e-14)
    photmjsr = h.get('PHOTMJSR')
    xposure = hdul[0].header.get('EFFEXPTM', h.get('XPOSURE', 6184.0))
    print(f'  PIXAR_SR: {pixar_sr:.4e}  PHOTMJSR: {photmjsr}  XPOSURE: {xposure:.1f}s')

    # ── Background statistics ────────────────────────────────────────
    print('  Computing background stats (sigma-clipped on subsample)...')
    rng = np.random.default_rng(42)
    # Sample random pixels (avoid edges)
    n_sample = 500000
    ys = rng.integers(500, ny - 500, size=n_sample)
    xs = rng.integers(500, nx - 500, size=n_sample)
    sample = sci[ys, xs]
    valid = np.isfinite(sample) & (sample != 0)
    sample = sample[valid]
    bg_mean, bg_median, bg_std = sigma_clipped_stats(sample, sigma=3, maxiters=5)
    print(f'  Background: mean={bg_mean:.6f}  median={bg_median:.6f}  std={bg_std:.6f} MJy/sr')

    # ── Extract background patches ───────────────────────────────────
    print(f'  Extracting {N_BACKGROUNDS} background patches ({PATCH_SIZE}x{PATCH_SIZE})...')
    half = PATCH_SIZE // 2
    patches = []
    attempts = 0
    max_attempts = N_BACKGROUNDS * 20

    while len(patches) < N_BACKGROUNDS and attempts < max_attempts:
        attempts += 1
        cy = int(rng.integers(half + 100, ny - half - 100))
        cx = int(rng.integers(half + 100, nx - half - 100))

        patch = sci[cy - half:cy + half + 1, cx - half:cx + half + 1]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        # Check validity
        valid_mask = np.isfinite(patch) & (patch != 0)
        if valid_mask.mean() < VALID_FRAC:
            continue

        # Subtract background
        patch_clean = patch - bg_median
        patch_clean = np.where(valid_mask, patch_clean, 0.0)
        patches.append(patch_clean.astype(np.float32))

        if len(patches) % 500 == 0:
            print(f'    {len(patches)}/{N_BACKGROUNDS} patches ({attempts} attempts)')

    patches = np.array(patches)
    print(f'  Got {len(patches)} patches in {attempts} attempts')
    np.save(str(band_dir / 'backgrounds.npy'), patches)
    print(f'  Saved -> {band_dir}/backgrounds.npy  {patches.nbytes/1e6:.1f} MB')

    # ── Extract PSF stars ────────────────────────────────────────────
    print(f'  Extracting PSF stars...')
    half_psf = PSF_STAMP_SIZE // 2

    # Find bright pixels (potential stars) using a subsample approach
    # Work in tiles to avoid loading everything at once
    bright_threshold = bg_median + 20 * bg_std
    star_candidates = []

    tile_size = 4000
    for ty in range(500, ny - 500, tile_size):
        for tx in range(500, nx - 500, tile_size):
            ty_end = min(ty + tile_size, ny - 500)
            tx_end = min(tx + tile_size, nx - 500)
            tile = sci[ty:ty_end, tx:tx_end]

            # Find bright pixels
            bright = np.where(tile > bright_threshold)
            if len(bright[0]) == 0:
                continue

            for i in range(len(bright[0])):
                py = bright[0][i] + ty
                px = bright[1][i] + tx

                # Check if local maximum in 5x5
                region = sci[py-2:py+3, px-2:px+3]
                if region.shape != (5, 5):
                    continue
                if not np.isfinite(region).all():
                    continue
                if sci[py, px] != np.max(region):
                    continue

                star_candidates.append((py, px, float(sci[py, px])))

    print(f'  Found {len(star_candidates)} bright local maxima')

    # Sort by brightness and enforce isolation
    star_candidates.sort(key=lambda x: -x[2])
    stars_used = []
    psf_stamps = []

    for cy, cx, peak in star_candidates:
        if len(psf_stamps) >= N_PSF_MAX:
            break

        # Check isolation (no other selected star within 50 pixels)
        too_close = False
        for sy, sx in stars_used:
            if abs(cy - sy) < 50 and abs(cx - sx) < 50:
                too_close = True
                break
        if too_close:
            continue

        # Extract stamp
        stamp = sci[cy - half_psf:cy + half_psf + 1,
                     cx - half_psf:cx + half_psf + 1].copy()
        if stamp.shape != (PSF_STAMP_SIZE, PSF_STAMP_SIZE):
            continue
        if not np.isfinite(stamp).all():
            continue

        # Compactness check: >50% of flux within r < 10 px
        yy, xx = np.mgrid[:PSF_STAMP_SIZE, :PSF_STAMP_SIZE]
        r = np.sqrt((yy - half_psf)**2 + (xx - half_psf)**2)
        stamp_sub = stamp - bg_median
        total_flux = np.sum(stamp_sub)
        core_flux = np.sum(stamp_sub[r < 10])
        if total_flux <= 0 or core_flux / total_flux < 0.5:
            continue

        # Normalize
        stamp_norm = stamp_sub / total_flux
        psf_stamps.append(stamp_norm.astype(np.float32))
        stars_used.append((cy, cx))

    print(f'  Selected {len(psf_stamps)} PSF stars')

    if len(psf_stamps) >= 3:
        psf_stamps = np.array(psf_stamps)
        np.save(str(band_dir / 'psf_stars.npy'), psf_stamps)

        # Median stack
        psf_median = np.median(psf_stamps, axis=0).astype(np.float64)
        psf_median = np.clip(psf_median, 0, None)
        psf_median /= psf_median.sum()
        np.save(str(band_dir / 'psf_median.npy'), psf_median)
        print(f'  Saved -> {band_dir}/psf_stars.npy ({len(psf_stamps)} stamps)')
        print(f'  Saved -> {band_dir}/psf_median.npy (sum={psf_median.sum():.4f})')
    else:
        print(f'  WARNING: Only {len(psf_stamps)} stars found, skipping PSF')

    # Save band info
    band_info[band] = {
        'pixar_sr': pixar_sr,
        'photmjsr': photmjsr,
        'xposure': xposure,
        'pixel_scale': PIXEL_SCALE,
        'bg_median': float(bg_median),
        'bg_std': float(bg_std),
        'n_backgrounds': len(patches),
        'n_psf_stars': len(psf_stamps),
        'mosaic_shape': [ny, nx],
    }

    hdul.close()

# ── Save band info ───────────────────────────────────────────────────────

with open(OUT_DIR / 'band_info.json', 'w') as f:
    json.dump(band_info, f, indent=2)
print(f'\nSaved -> {OUT_DIR}/band_info.json')

print('\nDone! Summary:')
for band in BANDS:
    info = band_info.get(band, {})
    print(f'  {band}: {info.get("n_backgrounds", 0)} backgrounds, '
          f'{info.get("n_psf_stars", 0)} PSF stars, '
          f'XPOSURE={info.get("xposure", 0):.0f}s')
