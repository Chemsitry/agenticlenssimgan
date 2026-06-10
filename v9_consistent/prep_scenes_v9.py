"""
prep_scenes_v9.py — Cut 630x630 JWST scenes only at positions of confirmed
DEV-morphology elliptical galaxies with known Legacy Survey photo-z.

Same per-cutout processing as prep_scenes_v8 (NaN check, brightness, anti-star
compactness), but the candidate list comes from Legacy Survey DR10 (TYPE='DEV'
with valid Z_PHOT_MEAN) instead of an unguided brightness-detection scan.

Inputs (read from ~/Desktop/data prep/):
  - raw_data/1727_mosaic/<band>/mosaic_*.fits   (COSMOS-Web NIRCam mosaic)
  - prepped_mosaic_630/band_info.json           (bg_median, pixar_sr per band)
  - desi_jwst_dev/cache/*sweep*.fits / *-pz.fits   (Legacy Survey + photo-z VAC)

Outputs (written to ~/Desktop/data prep v9_consistent/):
  prepped_scenes_v9/
    scenes_F115W.npy / F150W / F277W / F444W   (N, 630, 630)
    manifest.parquet   one row per kept scene: RA, Dec, z_phot, z_phot_err, TYPE,
                       LS RELEASE/BRICKID/OBJID, source x/y/peak/compactness
    scene_info.json    summary
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import argparse

# ── Paths ───────────────────────────────────────────────────────────────
DATA_PREP_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep')
DESI_CACHE     = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache')
V9_ROOT        = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')

MOSAIC_DIR = DATA_PREP_ROOT / 'raw_data' / '1727_mosaic'
BAND_INFO  = DATA_PREP_ROOT / 'prepped_mosaic_630' / 'band_info.json'
OUT_DIR    = V9_ROOT / 'prepped_scenes_v9'

# COSMOS-Web COSMOS-field sweeps cached by desi_jwst_dev:
SWEEP_FILES = [
    DESI_CACHE / '604133c95f__sweep-145p000-150p005.fits',
    DESI_CACHE / '4c7cd8831a__sweep-150p000-155p005.fits',
]
PZ_FILES = [
    DESI_CACHE / 'da3f342a9a__sweep-145p000-150p005-pz.fits',
    DESI_CACHE / 'd74e4b30f9__sweep-150p000-155p005-pz.fits',
]

# ── Config (same defaults as prep_scenes_v8) ────────────────────────────
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
SCENE_SIZE = 630
PEAK_MIN = 100              # F444W central peak (sim units); lowered from v8's 1000 to admit higher-z ellipticals
COMPACTNESS_MAX = 0.35      # rejects stars/diffraction spikes
EDGE_MARGIN = 250           # allow positions slightly closer to mosaic edge (gaps zero-padded by NaN handler)

SUM_TO_FLUX = 6.501853565914121

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Build v9 scenes from known-z DEV galaxies.')
parser.add_argument('--zmax', type=float, default=0.8, help='Max Z_PHOT_MEAN (default 0.8)')
parser.add_argument('--zstd-max', type=float, default=0.1, help='Max Z_PHOT_STD (default 0.1)')
parser.add_argument('--max-candidates', type=int, default=2000,
                    help='Cap on how many DEV positions to try (default 2000)')
parser.add_argument('--max-scenes', type=int, default=500,
                    help='Stop once this many scenes pass QA (default 500)')
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'output -> {OUT_DIR}')


# ── 1. Build the candidate list: DEV + valid photo-z ────────────────────
def load_legacy_survey_dev():
    print('loading Legacy Survey + photo-z for COSMOS field...')
    ls_pieces = []
    pz_pieces = []
    for sf, pf in zip(SWEEP_FILES, PZ_FILES):
        with fits.open(sf, memmap=True) as h:
            t = h[1].data
            tstr = np.array([s.strip() if isinstance(s, str) else s.decode().strip()
                             for s in t['TYPE']])
            dev = tstr == 'DEV'
            ls_pieces.append(pd.DataFrame({
                'RA': np.asarray(t['RA'])[dev],
                'DEC': np.asarray(t['DEC'])[dev],
                'TYPE': tstr[dev],
                'RELEASE': np.asarray(t['RELEASE'])[dev],
                'BRICKID': np.asarray(t['BRICKID'])[dev],
                'OBJID':   np.asarray(t['OBJID'])[dev],
                'MASKBITS': np.asarray(t['MASKBITS'])[dev],
            }))
        with fits.open(pf, memmap=True) as h:
            t = h[1].data
            pz_pieces.append(pd.DataFrame({
                'RELEASE': np.asarray(t['RELEASE']),
                'BRICKID': np.asarray(t['BRICKID']),
                'OBJID':   np.asarray(t['OBJID']),
                'Z_PHOT_MEAN': np.asarray(t['Z_PHOT_MEAN']),
                'Z_PHOT_STD':  np.asarray(t['Z_PHOT_STD']),
            }))
    ls = pd.concat(ls_pieces, ignore_index=True)
    pz = pd.concat(pz_pieces, ignore_index=True)
    df = ls.merge(pz, on=['RELEASE', 'BRICKID', 'OBJID'], how='inner')

    print(f'  {len(df):,} DEV galaxies with photo-z VAC join')

    cut = ((df['Z_PHOT_MEAN'] > 0.0) &
           (df['Z_PHOT_MEAN'] < args.zmax) &
           (df['Z_PHOT_STD'] < args.zstd_max) &
           (df['MASKBITS'] == 0))
    df = df[cut].reset_index(drop=True)
    print(f'  after z_phot<{args.zmax} + z_std<{args.zstd_max} + maskbits=0: {len(df):,}')
    return df


candidates = load_legacy_survey_dev()

# ── 2. Find which candidates fall inside the COSMOS-Web mosaic ──────────
print('\nopening F444W mosaic for WCS / footprint check...')
f444_mosaic = list((MOSAIC_DIR / 'F444W').glob('mosaic*.fits'))[0]
with fits.open(f444_mosaic, memmap=True) as h:
    wcs = WCS(h[1].header)
    ny, nx = h[1].data.shape
xs, ys = wcs.all_world2pix(candidates['RA'].to_numpy(),
                            candidates['DEC'].to_numpy(), 0)

inside = ((xs >= EDGE_MARGIN) & (xs < nx - EDGE_MARGIN) &
          (ys >= EDGE_MARGIN) & (ys < ny - EDGE_MARGIN))
candidates = candidates.loc[inside].copy().reset_index(drop=True)
candidates['x_pix'] = xs[inside].astype(int)
candidates['y_pix'] = ys[inside].astype(int)
print(f'  inside mosaic with {EDGE_MARGIN}-px edge margin: {len(candidates):,}')

if len(candidates) == 0:
    sys.exit('No DEV candidates inside the COSMOS-Web mosaic. Check WCS / paths.')

# Optionally cap the number we try (keep brightest? for now just random shuffle)
if len(candidates) > args.max_candidates:
    candidates = candidates.sample(args.max_candidates, random_state=42).reset_index(drop=True)
    print(f'  capped to {args.max_candidates} for this run')


# ── 3. Cut stamps per band + apply per-cutout QA ────────────────────────
band_info = json.loads(BAND_INFO.read_text())
half = SCENE_SIZE // 2
n_try = len(candidates)

# Pre-allocate per-band raw stamps
raw = {b: np.zeros((n_try, SCENE_SIZE, SCENE_SIZE), dtype=np.float32) for b in BANDS}
valid = np.ones(n_try, dtype=bool)

for band in BANDS:
    info = band_info[band]
    m2s = info['pixar_sr'] * 1e15 * SUM_TO_FLUX
    bg_median = info['bg_median']
    fits_file = list((MOSAIC_DIR / band).glob('mosaic*.fits'))[0]
    print(f'\n{band}: bg_median={bg_median:.4e}  m2s={m2s:.2f}')
    with fits.open(str(fits_file), memmap=True) as hdul:
        sci = hdul[1].data
        for i, row in candidates.iterrows():
            if not valid[i]:
                continue
            x, y = int(row['x_pix']), int(row['y_pix'])
            y0, x0 = y - half, x - half
            y1, x1 = y0 + SCENE_SIZE, x0 + SCENE_SIZE
            if y0 < 0 or x0 < 0 or y1 > sci.shape[0] or x1 > sci.shape[1]:
                valid[i] = False
                continue
            s = sci[y0:y1, x0:x1].astype(np.float32)
            if not np.all(np.isfinite(s)):
                # The COSMOS-Web mosaic has internal gaps. Tolerate up to 25%
                # NaN by zeroing them — the central galaxy is intact since
                # gaps are concentrated near tile edges away from the galaxy.
                # Beyond 25% the cutout is too holey to use.
                if (~np.isfinite(s)).mean() > 0.25:
                    valid[i] = False
                    continue
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            raw[band][i] = (s - bg_median) * m2s
            if (i + 1) % 100 == 0 and band == BANDS[0]:
                print(f'  {i+1}/{n_try} cut')

# Brightness filter on F444W central peak
c = half
f444 = raw['F444W']
center_peaks = f444[:, c-20:c+20, c-20:c+20].max(axis=(1, 2))
bright = center_peaks > PEAK_MIN

# Compactness filter (anti-star)
yy, xx = np.mgrid[:SCENE_SIZE, :SCENE_SIZE]
rr = np.sqrt((yy - c)**2 + (xx - c)**2)
mask_small = rr < 3
mask_medium = rr < 15
compactness = np.full(n_try, 1.0, dtype=np.float32)
for i in range(n_try):
    if not valid[i]:
        continue
    s = f444[i]
    fm = s[mask_medium].sum()
    if fm > 0:
        compactness[i] = float(s[mask_small].sum() / fm)
not_star = compactness < COMPACTNESS_MAX

keep = valid & bright & not_star
print(f'\nQA filter results:')
print(f'  in-bounds + finite: {int(valid.sum())}/{n_try}')
print(f'  central F444W peak > {PEAK_MIN}: {int(bright.sum())}/{n_try}')
print(f'  compactness < {COMPACTNESS_MAX} (anti-star): {int(not_star.sum())}/{n_try}')
print(f'  passing all filters: {int(keep.sum())}/{n_try}')

# Cap at --max-scenes
kept_idx = np.where(keep)[0]
if len(kept_idx) > args.max_scenes:
    kept_idx = kept_idx[:args.max_scenes]
    print(f'  capped to first {args.max_scenes} for this run')

# ── 4. Save scenes + manifest ───────────────────────────────────────────
n_keep = len(kept_idx)
print(f'\nSaving {n_keep} scenes to {OUT_DIR}/')
for b in BANDS:
    arr = raw[b][kept_idx]
    path = OUT_DIR / f'scenes_{b}.npy'
    np.save(str(path), arr)
    print(f'  {b}: {arr.shape}  {arr.nbytes/1e6:.1f} MB')

manifest = candidates.iloc[kept_idx].copy().reset_index(drop=True)
manifest['scene_idx']      = np.arange(n_keep)
manifest['F444W_peak']     = center_peaks[kept_idx]
manifest['compactness']    = compactness[kept_idx]
manifest.to_parquet(OUT_DIR / 'manifest.parquet', index=False)
print(f'  manifest.parquet: {len(manifest)} rows')

info_out = {
    'n_scenes':           int(n_keep),
    'scene_size':         SCENE_SIZE,
    'pixel_scale':        0.03,
    'fov_arcsec':         SCENE_SIZE * 0.03,
    'source':             'Legacy Survey DR10 DEV-morphology + Zhou photo-z VAC',
    'z_phot_cuts':        f'Z_PHOT_MEAN < {args.zmax}, Z_PHOT_STD < {args.zstd_max}, MASKBITS=0',
    'brightness_filter':  f'central F444W peak > {PEAK_MIN} sim units',
    'compactness_filter': f'F444W sum(r<3)/sum(r<15) < {COMPACTNESS_MAX}',
    'mosaic':             'COSMOS-Web DR0.5 (raw_data/1727_mosaic/)',
    'notes':              'Each scene is centered on a Legacy Survey DEV galaxy '
                          'with a known photometric redshift. Use manifest.parquet '
                          'to map scene_idx → (RA, Dec, z_phot, ...).',
}
(OUT_DIR / 'scene_info.json').write_text(json.dumps(info_out, indent=2))

print(f'\nz_phot distribution of kept scenes:')
print(manifest['Z_PHOT_MEAN'].describe())
print(f'\nFirst 5 kept scenes:')
print(manifest.head(5)[['scene_idx','RA','DEC','Z_PHOT_MEAN','Z_PHOT_STD','F444W_peak','compactness']].to_string(index=False))
