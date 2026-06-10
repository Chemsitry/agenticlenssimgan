"""
prep_scenes_v12.py — unified lens-scene catalog combining:
  - v11 JADES DR5 confirmed ellipticals (GOODS-N + GOODS-S)
  - Pool 1 calibration galaxies that have DESI-measured σ_v
deduplicated by sky position.

For galaxies in Pool 1 (DESI σ_v), tag has_measured_sigma_v=True so the
simulator can use the actual σ_v instead of FJ-predicted σ_v.

Outputs:
  prepped_scenes_v12/
    scenes_F115W/F150W/F277W/F444W.npy
    manifest.parquet — superset of v11 manifest + has_measured_sigma_v + sigma_v_measured
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

V9_ROOT      = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
DATA_PREP_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep')
MOSAIC_CACHE = V9_ROOT / 'cache_mosaics'
COSMOS_MOSAIC_DIR = DATA_PREP_ROOT / 'raw_data' / '1727_mosaic'

V11_DIR = V9_ROOT / 'prepped_scenes_v11'
OUT_DIR = V9_ROOT / 'prepped_scenes_v12'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
SCENE_SIZE = 630
SUM_TO_FLUX = 6.501853565914121
NAN_TOLERANCE = 0.25
DEDUP_ARCSEC = 3.0

COSMOS_BAND_INFO = json.loads(
    (DATA_PREP_ROOT / 'prepped_mosaic_630' / 'band_info.json').read_text())

# Field → mosaic paths for the 4 calibration fields' galaxies
def calib_mosaic_paths(field_initial):
    if field_initial == 'primer_cosmos' or field_initial == 'cosmos_web_mosaic':
        return {b: list((COSMOS_MOSAIC_DIR / b).glob('mosaic*.fits'))[0] for b in BANDS}
    if field_initial == 'primer_uds':
        # Include F444W now that we've downloaded it
        return {b: MOSAIC_CACHE / f'primer-uds_{b.lower()}_drc_sci.fits.gz' for b in BANDS}
    if field_initial == 'jades_gdn':
        return {b: MOSAIC_CACHE / f'jades-gdn_{b.lower()}_drz.fits' for b in BANDS}
    if field_initial == 'ceers_egs':
        ceers_dir = MOSAIC_CACHE / 'ceers_pointings'
        per = {}
        for b in BANDS[:3]:
            fs = sorted(ceers_dir.rglob(f'hlsp_ceers*{b.lower()}*_sci.fits*'))
            fs = [f for f in fs if 'bkgsub' not in f.name] or fs
            if fs: per[b] = fs[0]
        return per
    return {}


def open_mosaic_for_field(path):
    h = fits.open(str(path), memmap=True)
    for hdu in h:
        if hdu.data is not None and hdu.data.ndim == 2:
            bunit = str(hdu.header.get('BUNIT', '')).strip()
            pixar_sr = float(hdu.header.get('PIXAR_SR', 0.0))
            if 'mjy/sr' in bunit.lower():
                m2s = pixar_sr * 1e15 * SUM_TO_FLUX
            elif 'nanojansky' in bunit.lower() or 'njy' in bunit.lower():
                scale = 10.0 if '10' in bunit else 1.0
                m2s = scale * 1e-9 * 1e-6 * 1e15 * SUM_TO_FLUX
            else:
                m2s = pixar_sr * 1e15 * SUM_TO_FLUX if pixar_sr > 0 else 1.0
            return h, hdu, WCS(hdu.header), m2s
    h.close()
    return None


def cut_stamp(sci_hdu, wcs, ra, dec, scene_size=SCENE_SIZE, bg_median=0.0, m2s=1.0):
    try:
        x, y = wcs.all_world2pix(ra, dec, 0)
        x = int(x); y = int(y)
    except Exception:
        return None
    half = scene_size // 2
    y0, x0 = y - half, x - half
    y1, x1 = y0 + scene_size, x0 + scene_size
    ny, nx = sci_hdu.data.shape
    if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
        return None
    try:
        s = sci_hdu.data[y0:y1, x0:x1].astype(np.float32)
    except (OSError, TypeError, ValueError):
        return None
    if not np.isfinite(s).any():
        return None
    if (~np.isfinite(s)).mean() > NAN_TOLERANCE:
        return None
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return (s - bg_median) * m2s


def main():
    # 1. Load v11 manifest + arrays
    v11_manifest = pd.read_parquet(V11_DIR / 'manifest.parquet')
    print(f'v11 baseline: {len(v11_manifest)} scenes')
    v11_arrays = {b: np.load(str(V11_DIR / f'scenes_{b}.npy')) for b in BANDS}

    # 2. Load DESI calibration sample (49)
    cal = pd.read_parquet(V9_ROOT / 'sample_final_extended.parquet')
    cal = cal.dropna(subset=['m115','m150','m277','VDISP','Z']).reset_index(drop=True)
    # Map sample_final field strings to the field_initial values used elsewhere
    field_map = {'jades_gdn': 'jades_gdn', 'ceers_egs': 'ceers_egs',
                 'primer_uds': 'primer_uds', 'primer_cosmos': 'primer_cosmos',
                 'cosmos_web_mosaic': 'primer_cosmos',
                 'primer_cosmos_mosaic': 'primer_cosmos'}
    cal['field_initial'] = cal['field'].map(field_map).fillna(cal['field'])
    print(f'\ncalibration sample: {len(cal)} galaxies (DESI σ_v measured)')
    print(cal['field_initial'].value_counts().to_string())

    # 3. Deduplicate against v11 (some jades_gdn calib galaxies may already be in v11)
    v11_coords = SkyCoord(v11_manifest['RA'].to_numpy()*u.deg,
                          v11_manifest['DEC'].to_numpy()*u.deg)
    cal_coords = SkyCoord(cal['RA'].to_numpy()*u.deg,
                          cal['DEC'].to_numpy()*u.deg)
    idx, sep, _ = cal_coords.match_to_catalog_sky(v11_coords)
    is_new = sep.arcsec > DEDUP_ARCSEC
    print(f'\nof {len(cal)} calibration galaxies, {int(is_new.sum())} are NOT already in v11')
    new_cal = cal.loc[is_new].reset_index(drop=True)

    # 4. For each new calibration galaxy, cut JWST stamps from its field's mosaic
    new_stamps = {b: [] for b in BANDS}
    new_rows = []
    by_field = new_cal.groupby('field_initial')
    for field, sub in by_field:
        paths = calib_mosaic_paths(field)
        if not paths:
            print(f'  field {field}: no mosaic paths available, skip {len(sub)} candidates')
            continue
        print(f'  field {field}: cutting {len(sub)} stamps...')
        # Open once per band
        opened = {}
        for b in BANDS:
            if b not in paths: continue
            res = open_mosaic_for_field(paths[b])
            if res is not None:
                opened[b] = res
        try:
            for _, row in sub.iterrows():
                stamps = {}
                ok = True
                for b in BANDS:
                    if b not in opened:
                        # Zero-pad missing band (PRIMER-UDS has no F444W on disk)
                        stamps[b] = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
                        continue
                    h, sci_hdu, wcs, m2s = opened[b]
                    bg = COSMOS_BAND_INFO[b]['bg_median'] if field == 'primer_cosmos' else 0.0
                    s = cut_stamp(sci_hdu, wcs, row['RA'], row['DEC'],
                                  bg_median=bg, m2s=m2s)
                    if s is None:
                        ok = False; break
                    stamps[b] = s
                if not ok:
                    continue
                for b in BANDS:
                    new_stamps[b].append(stamps[b])
                new_rows.append({
                    'jades_id':    -1,
                    'RA':          float(row['RA']),
                    'DEC':         float(row['DEC']),
                    'Z_PHOT_MEAN': float(row.get('Z', row['Z_PHOT_MEAN'])),
                    'Z_PHOT_STD':  0.01,
                    'TYPE':        'DEV',
                    'class_label': 'DESI-confirmed',
                    'class_prob':  1.0,
                    'field':       field,
                    'F444W_peak':  float(stamps['F444W'][SCENE_SIZE//2-20:SCENE_SIZE//2+20,
                                                         SCENE_SIZE//2-20:SCENE_SIZE//2+20].max()),
                    'has_F444W':   bool('F444W' in opened),
                    'has_measured_sigma_v': True,
                    'sigma_v_measured': float(row['VDISP']),
                    'desi_targetid': int(row['TARGETID']) if 'TARGETID' in row.index else -1,
                })
        finally:
            for h, _, _, _ in opened.values():
                try: h.close()
                except Exception: pass

    print(f'\n{len(new_rows)} new calibration scenes cut successfully')

    # 5. Combine arrays
    if new_rows:
        combined = {}
        for b in BANDS:
            new_arr = np.stack(new_stamps[b], axis=0)
            combined[b] = np.concatenate([v11_arrays[b], new_arr], axis=0)
    else:
        combined = v11_arrays

    # 6. Combine manifests
    v11_manifest['has_measured_sigma_v'] = False
    v11_manifest['sigma_v_measured'] = np.nan
    v11_manifest['desi_targetid'] = -1
    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    # Align columns
    for col in v11_manifest.columns:
        if col not in new_df.columns:
            new_df[col] = None
    for col in new_df.columns:
        if col not in v11_manifest.columns:
            v11_manifest[col] = None
    new_df = new_df[v11_manifest.columns]
    manifest = pd.concat([v11_manifest, new_df], ignore_index=True)
    manifest['scene_idx'] = np.arange(len(manifest))

    # 6b. Filter out scenes with empty central data (mosaic coverage gaps)
    # Use F277W as the reference band — every kept scene must have a real galaxy
    # at center, not just NaN/zero from a mosaic gap.
    final_arr = {b: combined[b] for b in BANDS}
    c = SCENE_SIZE // 2
    cen_peak_277 = final_arr['F277W'][:, c-20:c+20, c-20:c+20].max(axis=(1,2))
    cen_peak_115 = final_arr['F115W'][:, c-20:c+20, c-20:c+20].max(axis=(1,2))
    cen_peak_150 = final_arr['F150W'][:, c-20:c+20, c-20:c+20].max(axis=(1,2))
    keep_mask = (cen_peak_115 > 5) & (cen_peak_150 > 5) & (cen_peak_277 > 5)
    n_dropped = int((~keep_mask).sum())
    print(f'\nDropping {n_dropped} scenes with empty central data (mosaic gaps)')
    for b in BANDS:
        combined[b] = combined[b][keep_mask]
    manifest = manifest[keep_mask].reset_index(drop=True)
    manifest['scene_idx'] = np.arange(len(manifest))
    print(f'  kept: {len(manifest)} clean scenes')

    # 7. Save
    for b in BANDS:
        np.save(str(OUT_DIR / f'scenes_{b}.npy'), combined[b])
        print(f'  {b}: {combined[b].shape}  {combined[b].nbytes/1e6:.1f} MB')
    manifest.to_parquet(OUT_DIR / 'manifest.parquet', index=False)
    print(f'\nmanifest: {len(manifest)} scenes')
    print(f'  field breakdown: {manifest["field"].value_counts().to_dict()}')
    print(f'  with measured σ_v: {int(manifest["has_measured_sigma_v"].sum())}')

    info = {
        'n_scenes':       int(len(manifest)),
        'scene_size':     SCENE_SIZE,
        'pixel_scale':    0.03,
        'source':         'v12 = v11 JADES DR5 ellipticals (494) + DESI σ_v calibration galaxies (deduped)',
        'with_measured_sigma_v': int(manifest['has_measured_sigma_v'].sum()),
    }
    (OUT_DIR / 'scene_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
