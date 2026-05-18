"""
extend_fj_with_mosaic_photometry.py — for FJ-calibration candidates that have
a DESI σ_v but fell outside the published JWST catalog footprints, measure
F115W/F150W/F277W magnitudes directly from the COSMOS-Web mosaic.

Inputs:
  - desi_jwst_dev/cache/dev_vdisp.parquet  (DEV galaxies with quality DESI σ_v)
  - desi_jwst_dev/cache/sample_final.parquet  (subset already with catalog mags)
  - data prep/raw_data/1727_mosaic/<band>/mosaic_*.fits  (COSMOS-Web NIRCam mosaic)

Output:
  - data prep v9_consistent/sample_final_extended.parquet
    (sample_final rows merged with mosaic-measured mags for primer_cosmos galaxies
     that the JWST catalog cross-match missed)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

V9_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
sys.path.insert(0, str(V9_ROOT))
import cutout_photometry as cphot

DESI_CACHE = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache')
MOSAIC_DIR = Path('/Users/nathankvinnesland/Desktop/data prep/raw_data/1727_mosaic')
BAND_INFO  = Path('/Users/nathankvinnesland/Desktop/data prep/prepped_mosaic_630/band_info.json')

# Bands the calibration uses
BANDS = ['F115W', 'F150W', 'F277W']
SUM_TO_FLUX = 6.501853565914121
STAMP_SIZE = 201  # large enough for a 1.8" aperture + sky annulus

# Load DESI dev_vdisp (quality σ_v sample) and existing sample_final
dv = pd.read_parquet(DESI_CACHE / 'dev_vdisp.parquet')
sf = pd.read_parquet(DESI_CACHE / 'sample_final.parquet')
print(f'dev_vdisp: {len(dv)}  ·  sample_final (catalog mags): {len(sf)}')

# Existing sample_final TARGETIDs — don't double-count these
have_targets = set(sf['TARGETID'].tolist())

# Restrict candidates to galaxies in the COSMOS-Web mosaic footprint
# (RA 149.85–150.05, Dec 2.28–2.47 approximately — primer_cosmos field is here)
cand = dv[~dv['TARGETID'].isin(have_targets)].copy()
print(f'candidates not yet in sample_final: {len(cand)}')

# Open the mosaic WCS to filter by footprint and convert RA/Dec → pixel
print('\nopening F444W mosaic WCS to find COSMOS-Web-resident candidates...')
mosaic_paths = {b: list((MOSAIC_DIR / b).glob('mosaic*.fits'))[0] for b in BANDS}
with fits.open(mosaic_paths[BANDS[0]], memmap=True) as h:
    wcs = WCS(h[1].header)
    ny, nx = h[1].data.shape

xs, ys = wcs.all_world2pix(cand['RA'].to_numpy(), cand['DEC'].to_numpy(), 0)
half = STAMP_SIZE // 2
in_mosaic = (xs >= half) & (xs < nx - half) & (ys >= half) & (ys < ny - half)
cand = cand.loc[in_mosaic].copy()
cand['x_pix'] = xs[in_mosaic].astype(int)
cand['y_pix'] = ys[in_mosaic].astype(int)
print(f'  candidates inside COSMOS-Web mosaic: {len(cand)}')

if len(cand) == 0:
    print('Nothing to add — all dev_vdisp galaxies were either already in sample_final or outside COSMOS-Web.')
    sys.exit(0)

# Measure F115W/F150W/F277W AB mag at each candidate's position
band_info = json.loads(BAND_INFO.read_text())
new_rows = []
print('\nmeasuring per-band photometry from mosaic...')
for band in BANDS:
    info = band_info[band]
    m2s = info['pixar_sr'] * 1e15 * SUM_TO_FLUX
    bg_median = info['bg_median']
    print(f'  {band}: bg_median={bg_median:.4e}  m2s={m2s:.2f}')
    with fits.open(mosaic_paths[band], memmap=True) as h:
        sci = h[1].data
        for i, row in cand.iterrows():
            x, y = int(row['x_pix']), int(row['y_pix'])
            stamp = sci[y-half:y+half+1, x-half:x+half+1].astype(np.float32)
            stamp = np.nan_to_num(stamp, nan=0.0, posinf=0.0, neginf=0.0)
            stamp = (stamp - bg_median) * m2s
            mag = cphot.cutout_ab_mag(stamp,
                                       aperture_radius_pix=30,   # tighter aperture for cleaner mag
                                       sky_annulus=(40, 60))
            cand.loc[i, f'm_{band}'] = mag

# Drop candidates whose photometry came back NaN in any band
ok = (cand['m_F115W'].notna() & cand['m_F150W'].notna() & cand['m_F277W'].notna() &
      np.isfinite(cand['m_F115W']) & np.isfinite(cand['m_F150W']) & np.isfinite(cand['m_F277W']))
cand = cand[ok].copy()
print(f'\n  with finite mags in all 3 bands: {len(cand)}')

# Reshape into sample_final's column layout
new_rows = pd.DataFrame({
    'RA': cand['RA'], 'DEC': cand['DEC'],
    'Z':  cand['Z'],  'Z_PHOT_MEAN': cand.get('Z_PHOT_MEAN', cand['Z']),
    'VDISP': cand['VDISP'], 'VDISP_IVAR': cand['VDISP_IVAR'],
    'm115': cand['m_F115W'], 'm150': cand['m_F150W'], 'm277': cand['m_F277W'],
    'm115_err': np.nan, 'm150_err': np.nan, 'm277_err': np.nan,  # mosaic photometry doesn't give a per-band err here
    'TARGETID': cand['TARGETID'],
    'field': 'cosmos_web_mosaic',  # tag so we know where mags came from
    'jwst_match_arcsec': 0.0,
    'match_arcsec': cand.get('match_arcsec', 0.0),
})

# Combine with original sample_final
combined = pd.concat([sf, new_rows], ignore_index=True)
out_path = V9_ROOT / 'sample_final_extended.parquet'
combined.to_parquet(out_path, index=False)
print(f'\nwrote {out_path}: {len(combined)} galaxies ({len(sf)} catalog + {len(new_rows)} mosaic-measured)')
print()
print('Combined sample:')
print(combined[['RA','DEC','Z','VDISP','m115','m150','m277','field']].to_string(index=False))
print()
print(f'z range: {combined["Z"].min():.3f} – {combined["Z"].max():.3f}')
print(f'σ_v range: {combined["VDISP"].min():.0f} – {combined["VDISP"].max():.0f} km/s')
