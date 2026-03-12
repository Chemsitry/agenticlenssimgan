"""
prep_lenses.py — Extract real elliptical galaxy stamps for lens light profiles.

Selects compact/elliptical galaxies (high concentration) from the DR0.5 mosaics
to replace smooth Sersic lens light with realistic morphology: color gradients,
isophote structure, companions, and natural substructure.

Uses larger stamps (101x101 = 3.03") than source galaxies (65x65) because lens
galaxies at z~0.3-1.0 have larger angular extent.

Output: prepped_mosaic{_size}/lenses/
    stamps_F115W.npy  (N, 101, 101) — per-band galaxy stamps (sum=1)
    stamps_F150W.npy
    stamps_F277W.npy
    stamps_F444W.npy
    lens_info.json

Usage:
    .venv/bin/python3 prep_lenses.py --size 224
    .venv/bin/python3 prep_lenses.py --size 630 --n 300
"""

import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=224, help='Image size (matches prepped dir)')
parser.add_argument('--n', type=int, default=300, help='Number of lens galaxies to extract')
parser.add_argument('--stamp', type=int, default=101, help='Stamp size in pixels')
args = parser.parse_args()

MOSAIC_DIR = Path('raw_data/1727_mosaic')
PREPPED_DIR = Path('prepped_mosaic') if args.size == 125 else Path(f'prepped_mosaic_{args.size}')
OUT_DIR = PREPPED_DIR / 'lenses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
DETECT_BAND = 'F444W'   # ellipticals are brightest in red/NIR
STAMP_SIZE = args.stamp
HALF = STAMP_SIZE // 2
N_LENSES = args.n
VALID_FRAC = 0.95

# Load band info
with open(PREPPED_DIR / 'band_info.json') as f:
    band_info = json.load(f)


# ── Phase 1: Open mosaics ──────────────────────────────────────────────

print('Phase 1: Opening mosaics...')
band_data = {}

for band in BANDS:
    fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
    if not fits_files:
        raise FileNotFoundError(f'No FITS file in {MOSAIC_DIR / band}')
    hdul = fits.open(str(fits_files[0]), memmap=True)
    sci = hdul[1].data
    band_data[band] = {
        'sci': sci, 'hdul': hdul,
        'ny': sci.shape[0], 'nx': sci.shape[1],
        'bg_median': band_info[band]['bg_median'],
        'bg_std': band_info[band]['bg_std'],
    }
    print(f'  {band}: {sci.shape[0]}x{sci.shape[1]}  bg_std={band_info[band]["bg_std"]:.6f}')


# ── Phase 2: Detect compact/elliptical galaxies ───────────────────────

print(f'\nPhase 2: Detecting compact galaxies in {DETECT_BAND}...')

det = band_data[DETECT_BAND]
sci_det = det['sci']
ny, nx = det['ny'], det['nx']
bg_median_det = det['bg_median']
bg_std_det = det['bg_std']

# Higher detection threshold — we want bright ellipticals
detect_thresh = bg_median_det + 25 * bg_std_det
print(f'  Detection threshold: {detect_thresh:.6f} MJy/sr (25-sigma)')

candidates = []
tile_size = 4000
margin = HALF + 100

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

            # Local maximum in 5x5
            region = sci_det[py-2:py+3, px-2:px+3]
            if region.shape != (5, 5):
                continue
            if not np.isfinite(region).all():
                continue
            if sci_det[py, px] != np.max(region):
                continue

            # Measure concentration
            y0, x0 = py - HALF, px - HALF
            stamp = sci_det[y0:y0+STAMP_SIZE, x0:x0+STAMP_SIZE]
            if stamp.shape != (STAMP_SIZE, STAMP_SIZE):
                continue
            if not np.all(np.isfinite(stamp)):
                continue

            stamp_sub = stamp - bg_median_det
            yy, xx = np.mgrid[:STAMP_SIZE, :STAMP_SIZE]
            r = np.sqrt((yy - HALF)**2 + (xx - HALF)**2)

            core_flux = np.sum(stamp_sub[r < 8])
            total_flux = np.sum(stamp_sub[r < HALF])

            if total_flux <= 0:
                continue

            concentration = core_flux / total_flux

            # COMPACT/ELLIPTICAL: high concentration
            # But not TOO compact (reject stars): require some resolved extent
            # Stars: concentration > 0.75 at r<8 (very tight PSF)
            # Ellipticals: 0.45 < concentration < 0.75 (resolved but concentrated)
            if concentration < 0.45 or concentration > 0.75:
                continue

            # Require the source to be resolved: check flux at intermediate radii
            mid_flux = np.sum(stamp_sub[(r >= 5) & (r < 15)])
            if mid_flux < 0.05 * total_flux:
                continue  # too point-like (star)

            candidates.append((py, px, float(total_flux), float(concentration)))

print(f'  Found {len(candidates)} compact galaxy candidates')


# ── Phase 3: Extract multi-band stamps ─────────────────────────────────

print(f'\nPhase 3: Extracting {N_LENSES} lens galaxies...')

candidates.sort(key=lambda x: -x[2])

selected = []
stamps_per_band = {band: [] for band in BANDS}
catalog = []

for cy, cx, flux_det, conc in candidates:
    if len(selected) >= N_LENSES:
        break

    # Isolation: no other selected source within 1.5*STAMP_SIZE
    too_close = False
    for sy, sx in selected:
        if abs(cy - sy) < int(1.5 * STAMP_SIZE) and abs(cx - sx) < int(1.5 * STAMP_SIZE):
            too_close = True
            break
    if too_close:
        continue

    # Extract all bands
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

    # Background-subtract, gentle noise clip, smooth taper, normalize
    yy, xx = np.mgrid[:STAMP_SIZE, :STAMP_SIZE]
    r = np.sqrt((yy - HALF)**2 + (xx - HALF)**2)
    processed = {}
    ok = True

    for band in BANDS:
        stamp = band_stamps[band].astype(np.float64)
        stamp -= band_data[band]['bg_median']
        stamp = np.nan_to_num(stamp, nan=0.0)

        # 2.5-sigma threshold: zero out noise pixels
        noise_thresh = 2.5 * band_data[band]['bg_std']
        stamp[stamp < noise_thresh] = 0.0

        # Cosine taper over outer 20 pixels (smooth to zero at stamp edge)
        edge = 20
        taper = np.ones((STAMP_SIZE, STAMP_SIZE))
        for d in range(edge):
            w = 0.5 * (1 - np.cos(np.pi * d / edge))
            taper[d, :] *= w
            taper[-(d+1), :] *= w
            taper[:, d] *= w
            taper[:, -(d+1)] *= w
        stamp *= taper

        total = stamp.sum()
        if total <= 0:
            ok = False
            break

        # Check centering: galaxy should be centered
        frac_core = stamp[r < 15].sum() / total
        if frac_core < 0.3:
            ok = False
            break

        processed[band] = (stamp / total).astype(np.float32)

    if not ok:
        continue

    for band in BANDS:
        stamps_per_band[band].append(processed[band])

    selected.append((cy, cx))
    catalog.append({
        'y': int(cy), 'x': int(cx),
        'flux_detect': float(flux_det),
        'concentration': float(conc),
    })

    if len(selected) % 50 == 0:
        print(f'  {len(selected)}/{N_LENSES} lenses')

print(f'\n  Extracted {len(selected)} lens galaxies')

# Save
for band in BANDS:
    arr = np.array(stamps_per_band[band])
    np.save(str(OUT_DIR / f'stamps_{band}.npy'), arr)
    print(f'  {band}: {arr.shape} -> {OUT_DIR}/stamps_{band}.npy  {arr.nbytes/1e6:.1f} MB')

info = {
    'n_lenses': len(selected),
    'stamp_size': STAMP_SIZE,
    'pixel_scale': 0.03,
    'detect_band': DETECT_BAND,
    'detect_sigma': 25,
    'concentration_range': [0.45, 0.75],
    'bands': BANDS,
    'lenses': catalog,
}
with open(OUT_DIR / 'lens_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'  Catalog -> {OUT_DIR}/lens_info.json')

for band in BANDS:
    band_data[band]['hdul'].close()

print(f'\nDone! {len(selected)} lens galaxy stamps ready for INTERPOL lens light.')
