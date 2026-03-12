"""
prep_sources.py — Extract real galaxy stamps from COSMOS-Web DR0.5 mosaics.

These stamps replace smooth Sersic profiles as source galaxies in the
lens simulation, producing arcs with realistic complex morphology
(clumpy star-forming regions, spiral arms, irregular structure).

Detection: extended sources in F277W (15-sigma, concentration < 0.45)
Extraction: matched multi-band stamps, background-subtracted, normalized to sum=1

Note: Stamps include the JWST PSF convolution. Since lenstronomy reconvolves
the lensed image with the PSF, arcs will be mildly broadened (~sqrt(2) in PSF
sigma). This is acceptable for training data; future improvement could apply
PSF deconvolution before normalization.

Output: prepped_mosaic{_size}/sources/
    stamps_F115W.npy  (N, stamp, stamp) — per-band galaxy stamps (sum=1)
    stamps_F150W.npy
    stamps_F277W.npy
    stamps_F444W.npy
    source_info.json  — catalog with positions, fluxes, concentration

Usage:
    .venv/bin/python3 prep_sources.py
    .venv/bin/python3 prep_sources.py --size 224 --n 500
"""

import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=125, help='Image size (matches prepped dir)')
parser.add_argument('--n', type=int, default=500, help='Number of source galaxies to extract')
parser.add_argument('--stamp', type=int, default=65, help='Stamp size in pixels')
args = parser.parse_args()

MOSAIC_DIR = Path('raw_data/1727_mosaic')
PREPPED_DIR = Path('prepped_mosaic') if args.size == 125 else Path(f'prepped_mosaic_{args.size}')
OUT_DIR = PREPPED_DIR / 'sources'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
DETECT_BAND = 'F277W'   # good depth, traces rest-frame optical at z~2
STAMP_SIZE = args.stamp
HALF = STAMP_SIZE // 2
N_SOURCES = args.n
VALID_FRAC = 0.95

# Load band info for background stats
with open(PREPPED_DIR / 'band_info.json') as f:
    band_info = json.load(f)


# ── Phase 1: Open all mosaics ──────────────────────────────────────────

print('Phase 1: Opening mosaics...')
band_data = {}

for band in BANDS:
    fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
    if not fits_files:
        raise FileNotFoundError(f'No FITS file in {MOSAIC_DIR / band}')
    fits_path = fits_files[0]

    hdul = fits.open(str(fits_path), memmap=True)
    sci = hdul[1].data

    band_data[band] = {
        'sci': sci, 'hdul': hdul,
        'ny': sci.shape[0], 'nx': sci.shape[1],
        'bg_median': band_info[band]['bg_median'],
        'bg_std': band_info[band]['bg_std'],
    }
    print(f'  {band}: {sci.shape[0]}x{sci.shape[1]}  bg_std={band_info[band]["bg_std"]:.6f}')


# ── Phase 2: Detect extended sources ───────────────────────────────────

print(f'\nPhase 2: Detecting extended sources in {DETECT_BAND}...')

det = band_data[DETECT_BAND]
sci_det = det['sci']
ny, nx = det['ny'], det['nx']
bg_median_det = det['bg_median']
bg_std_det = det['bg_std']

# 15-sigma detection threshold
detect_thresh = bg_median_det + 15 * bg_std_det
print(f'  Detection threshold: {detect_thresh:.6f} MJy/sr (15-sigma above bg)')

candidates = []
tile_size = 4000
margin = HALF + 100  # buffer from mosaic edges

for ty in range(margin, ny - margin, tile_size):
    for tx in range(margin, nx - margin, tile_size):
        ty_end = min(ty + tile_size, ny - margin)
        tx_end = min(tx + tile_size, nx - margin)
        tile = sci_det[ty:ty_end, tx:tx_end]

        bright = np.where(tile > detect_thresh)
        if len(bright[0]) == 0:
            continue

        for i in range(len(bright[0])):
            py = bright[0][i] + ty
            px = bright[1][i] + tx

            # Local maximum check (5x5 window)
            region = sci_det[py-2:py+3, px-2:px+3]
            if region.shape != (5, 5):
                continue
            if not np.isfinite(region).all():
                continue
            if sci_det[py, px] != np.max(region):
                continue

            # Measure concentration in stamp-sized aperture
            y0, x0 = py - HALF, px - HALF
            stamp = sci_det[y0:y0+STAMP_SIZE, x0:x0+STAMP_SIZE]
            if stamp.shape != (STAMP_SIZE, STAMP_SIZE):
                continue
            if not np.all(np.isfinite(stamp)):
                continue

            stamp_sub = stamp - bg_median_det
            yy, xx = np.mgrid[:STAMP_SIZE, :STAMP_SIZE]
            r = np.sqrt((yy - HALF)**2 + (xx - HALF)**2)

            core_flux = np.sum(stamp_sub[r < 5])
            total_flux = np.sum(stamp_sub[r < HALF])

            if total_flux <= 0:
                continue

            concentration = core_flux / total_flux

            # EXTENDED sources: low concentration (galaxies, not stars)
            # Stars have concentration > 0.5 (compact PSF)
            # Galaxies: light is spread out, concentration < 0.45
            if concentration > 0.45:
                continue

            # Require minimum S/N (reject faint noise peaks)
            if total_flux < 50 * bg_std_det * STAMP_SIZE:
                continue

            candidates.append((py, px, float(total_flux), float(concentration)))

print(f'  Found {len(candidates)} extended source candidates')


# ── Phase 3: Extract multi-band stamps ─────────────────────────────────

print(f'\nPhase 3: Selecting and extracting {N_SOURCES} source galaxies...')

# Sort by flux (brightest first — best morphology signal)
candidates.sort(key=lambda x: -x[2])

selected = []
stamps_per_band = {band: [] for band in BANDS}
catalog = []

for cy, cx, flux_det, conc in candidates:
    if len(selected) >= N_SOURCES:
        break

    # Isolation: no other selected source within STAMP_SIZE pixels
    too_close = False
    for sy, sx in selected:
        if abs(cy - sy) < STAMP_SIZE and abs(cx - sx) < STAMP_SIZE:
            too_close = True
            break
    if too_close:
        continue

    # Extract and validate in ALL bands at same position
    y0, x0 = cy - HALF, cx - HALF
    band_stamps = {}
    all_valid = True

    for band in BANDS:
        sci = band_data[band]['sci']
        stamp = sci[y0:y0+STAMP_SIZE, x0:x0+STAMP_SIZE]

        if stamp.shape != (STAMP_SIZE, STAMP_SIZE):
            all_valid = False
            break
        valid = np.isfinite(stamp) & (stamp != 0)
        if valid.mean() < VALID_FRAC:
            all_valid = False
            break
        band_stamps[band] = stamp.copy()

    if not all_valid:
        continue

    # Background-subtract, sigma-threshold, and normalize
    # Key: zero out pixels below 3*bg_std to remove noise floor.
    # Without this, noise pixels eat >60% of normalized flux, making
    # the source diffuse/blobby when used as an INTERPOL profile.
    yy, xx = np.mgrid[:STAMP_SIZE, :STAMP_SIZE]
    r = np.sqrt((yy - HALF)**2 + (xx - HALF)**2)
    processed_stamps = {}
    stamp_ok = True

    for band in BANDS:
        stamp = band_stamps[band].astype(np.float64)
        stamp -= band_data[band]['bg_median']
        stamp = np.nan_to_num(stamp, nan=0.0)

        # 2.5-sigma threshold: zero out noise pixels
        noise_thresh = 2.5 * band_data[band]['bg_std']
        stamp[stamp < noise_thresh] = 0.0

        total = stamp.sum()
        if total <= 0:
            stamp_ok = False
            break

        # Check galaxy is centered: >25% of flux within r<10 pixels
        frac_core = stamp[r < 10].sum() / total
        if frac_core < 0.25:
            stamp_ok = False
            break

        processed_stamps[band] = stamp

    if not stamp_ok:
        continue

    # Compute band flux ratios (real galaxy SED) before per-band normalization.
    ref_total = processed_stamps[DETECT_BAND].sum()
    if ref_total <= 0:
        continue

    # Verify detection band has meaningful pixel count
    n_signal = np.sum(processed_stamps[DETECT_BAND] > 0)
    if n_signal < 20:
        continue

    # Normalize each band to sum=1 (clean spatial morphology per band)
    all_norm_ok = True
    for band in BANDS:
        total = processed_stamps[band].sum()
        if total <= 0:
            all_norm_ok = False
            break
        processed_stamps[band] = (processed_stamps[band] / total).astype(np.float32)
    if not all_norm_ok:
        continue

    for band in BANDS:
        stamps_per_band[band].append(processed_stamps[band])

    selected.append((cy, cx))
    catalog.append({
        'y': int(cy), 'x': int(cx),
        'flux_detect': float(flux_det),
        'concentration': float(conc),
    })

    if len(selected) % 100 == 0:
        print(f'  {len(selected)}/{N_SOURCES} sources')

print(f'\n  Extracted {len(selected)} source galaxies')

# Save stamps per band
for band in BANDS:
    arr = np.array(stamps_per_band[band])
    np.save(str(OUT_DIR / f'stamps_{band}.npy'), arr)
    print(f'  {band}: {arr.shape} -> {OUT_DIR}/stamps_{band}.npy  {arr.nbytes/1e6:.1f} MB')

# Save catalog
info = {
    'n_sources': len(selected),
    'stamp_size': STAMP_SIZE,
    'pixel_scale': 0.03,
    'detect_band': DETECT_BAND,
    'detect_sigma': 15,
    'concentration_max': 0.45,
    'sources': catalog,
}
with open(OUT_DIR / 'source_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'  Catalog -> {OUT_DIR}/source_info.json')

# Cleanup
for band in BANDS:
    band_data[band]['hdul'].close()

print(f'\nDone! {len(selected)} galaxy stamps ready for INTERPOL sources.')
