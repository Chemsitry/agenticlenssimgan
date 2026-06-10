"""
prep_scenes_v11.py — cut JWST lens scenes using ONLY JADES DR5 confirmed
ellipticals (classified by the JADES team using JWST imaging) as the parent.

Combines GOODS-N + GOODS-S. Each entry needs:
  1. A JADES DR5 classification: Round Smooth / In-between Round Smooth / Cigar Shaped Smooth
  2. An EAZY photo-z from the JADES DR3 (GOODS-N) or DR2 (GOODS-S) photometry catalog
  3. Position inside our downloaded JADES mosaic with edge margin for a 630-cutout

This is the "cleanest" elliptical sample we can build with public JWST data —
JADES is the deepest NIRCam survey and their morphology classifications use
the actual JWST imaging, not ground-based DR9/DR10 photometry.

Outputs:
  prepped_scenes_v11/
    scenes_F115W.npy / F150W / F277W / F444W   (N, 630, 630)
    manifest.parquet  — RA, Dec, z_phot, class, field, etc.
    scene_info.json
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

V9_ROOT       = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
DESI_CACHE    = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache')
MOSAIC_CACHE  = V9_ROOT / 'cache_mosaics'

OUT_DIR  = V9_ROOT / 'prepped_scenes_v11'
BANDS    = ['F115W', 'F150W', 'F277W', 'F444W']
SCENE_SIZE = 630
SUM_TO_FLUX = 6.501853565914121
EDGE_MARGIN = 315  # half of SCENE_SIZE
NAN_TOLERANCE = 0.25

# Mosaic paths per field
MOSAIC_PATHS = {
    'goods-n': {b: MOSAIC_CACHE / f'jades-gdn_{b.lower()}_drz.fits' for b in BANDS},
    'goods-s': {b: MOSAIC_CACHE / f'jades-gds_{b.lower()}_drz.fits' for b in BANDS},
}


def nat(a):
    a = np.asarray(a)
    return a.astype(a.dtype.newbyteorder('=')) if a.dtype.byteorder == '>' else a


def load_goods_n_with_z():
    """GOODS-N: already prepared in jades_dr5_ellipticals_with_z.parquet."""
    p = V9_ROOT / 'jades_dr5_ellipticals_with_z.parquet'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df['field'] = 'goods-n'
    df['z_phot'] = df['EAZY_z_a']
    return df[['jades_id','ra','dec','z_phot','class_label','probability','field']]


def load_goods_s_with_z():
    """GOODS-S: cross-match DR5 elliptical IDs to DR2 photometry catalog."""
    cat_path = MOSAIC_CACHE / 'jades-gds_dr2_catalog.fits'
    if not cat_path.exists():
        print(f'  GOODS-S DR2 catalog not present yet at {cat_path.name}; skipping.')
        return pd.DataFrame()
    df5 = pd.read_csv(V9_ROOT / 'jades_dr5_elliptical_galaxies.csv')
    gs = df5[df5['field'] == 'goods-s'].copy()
    gs['jades_id'] = gs['jades_id'].astype(int)
    print(f'  GOODS-S DR5 entries: {len(gs)}')

    with fits.open(str(cat_path), memmap=True) as h:
        pz_hdu = None
        flag_hdu = None
        for hdu in h:
            if hdu.name == 'PHOTOZ' and hdu.columns is not None:
                pz_hdu = hdu
            elif hdu.name == 'FLAG' and hdu.columns is not None:
                flag_hdu = hdu
        if pz_hdu is None or flag_hdu is None:
            print('  WARN: DR2 catalog missing PHOTOZ/FLAG HDU — falling back to RA/Dec match')
            return pd.DataFrame()
        pz_df = pd.DataFrame({
            'ID':       nat(pz_hdu.data['ID']),
            'EAZY_z_a': nat(pz_hdu.data['EAZY_z_a']),
        })
        flag_df = pd.DataFrame({
            'ID':  nat(flag_hdu.data['ID']),
            'RA':  nat(flag_hdu.data['RA']),
            'DEC': nat(flag_hdu.data['DEC']),
        })
    dr2 = pz_df.merge(flag_df, on='ID', how='inner')

    matched = gs.merge(dr2, left_on='jades_id', right_on='ID', how='inner')
    matched = matched[(matched['EAZY_z_a'] > 0.05) & (matched['EAZY_z_a'] < 5)]
    matched['z_phot'] = matched['EAZY_z_a']
    matched['field'] = 'goods-s'
    print(f'  GOODS-S matched + valid photo-z: {len(matched)}')
    return matched[['jades_id','ra','dec','z_phot','class_label','probability','field']]


def cut_stamp(sci_hdu, wcs, ra, dec, scene_size=SCENE_SIZE):
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
    return s


def open_mosaic(path):
    h = fits.open(str(path), memmap=True)
    for hdu in h:
        if hdu.data is None: continue
        if hdu.name == 'SCI' or hdu.data.ndim == 2:
            try:
                wcs = WCS(hdu.header)
                bunit = str(hdu.header.get('BUNIT', '')).strip()
                pixar_sr = float(hdu.header.get('PIXAR_SR', 0.0))
                # m2s scaling so output matches v8 "sim units" convention
                if 'mjy/sr' in bunit.lower():
                    m2s = pixar_sr * 1e15 * SUM_TO_FLUX
                elif 'nanojansky' in bunit.lower() or 'njy' in bunit.lower():
                    scale = 10.0 if '10' in bunit else 1.0
                    m2s = scale * 1e-9 * 1e-6 * 1e15 * SUM_TO_FLUX
                else:
                    m2s = pixar_sr * 1e15 * SUM_TO_FLUX if pixar_sr > 0 else 1.0
                return h, hdu, wcs, m2s
            except Exception:
                continue
    h.close()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields', nargs='+', default=['goods-n', 'goods-s'])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('loading parents...')
    gn = load_goods_n_with_z()
    print(f'  GOODS-N ready: {len(gn)}')
    gs = load_goods_s_with_z()
    print(f'  GOODS-S ready: {len(gs)}')

    parents = pd.concat([gn, gs], ignore_index=True) if len(gs) else gn
    if not args.fields:
        args.fields = parents['field'].unique().tolist()

    raw = {b: [] for b in BANDS}
    manifest_rows = []

    for field in args.fields:
        sub = parents[parents['field'] == field]
        if not len(sub):
            print(f'\n  field {field}: no entries — skipping')
            continue
        if not all(MOSAIC_PATHS[field][b].exists() for b in BANDS):
            missing = [b for b in BANDS if not MOSAIC_PATHS[field][b].exists()]
            print(f'\n  field {field}: missing mosaic bands {missing} — skipping')
            continue

        print(f'\n=== {field}: {len(sub)} candidates ===')
        # Open all 4 mosaics for this field once
        opened = {}
        for band in BANDS:
            res = open_mosaic(MOSAIC_PATHS[field][band])
            if res is not None:
                opened[band] = res
        try:
            kept = 0
            for _, row in sub.iterrows():
                stamps = {}
                ok = True
                for band in BANDS:
                    if band not in opened:
                        ok = False; break
                    h, sci_hdu, wcs, m2s = opened[band]
                    s = cut_stamp(sci_hdu, wcs, row['ra'], row['dec'])
                    if s is None:
                        ok = False; break
                    stamps[band] = s * m2s
                if not ok:
                    continue
                for band in BANDS:
                    raw[band].append(stamps[band])
                manifest_rows.append({
                    'jades_id':    int(row['jades_id']),
                    'RA':          float(row['ra']),
                    'DEC':         float(row['dec']),
                    'Z_PHOT_MEAN': float(row['z_phot']),
                    'Z_PHOT_STD':  0.05,   # rough — EAZY 68% intervals available later
                    'TYPE':        'DEV',  # JADES-confirmed elliptical
                    'class_label': str(row['class_label']),
                    'class_prob':  float(row['probability']),
                    'field':       field,
                    'F444W_peak':  float(stamps['F444W'][SCENE_SIZE//2-20:SCENE_SIZE//2+20,
                                                         SCENE_SIZE//2-20:SCENE_SIZE//2+20].max()),
                    'has_F444W':   True,
                })
                kept += 1
            print(f'  passed: {kept}')
        finally:
            for h, _, _, _ in opened.values():
                try: h.close()
                except Exception: pass

    n = len(manifest_rows)
    print(f'\nTotal scenes: {n}')
    if n == 0: return

    for band in BANDS:
        arr = np.stack(raw[band], axis=0)
        np.save(str(OUT_DIR / f'scenes_{band}.npy'), arr)
        print(f'  {band}: {arr.shape}  {arr.nbytes/1e6:.1f} MB')

    manifest = pd.DataFrame(manifest_rows)
    manifest['scene_idx'] = np.arange(len(manifest))
    manifest.to_parquet(OUT_DIR / 'manifest.parquet', index=False)
    print(f'\nfield counts: {manifest["field"].value_counts().to_dict()}')
    print(f'z range: {manifest["Z_PHOT_MEAN"].min():.2f} – {manifest["Z_PHOT_MEAN"].max():.2f}')
    print(f'class breakdown: {manifest["class_label"].value_counts().to_dict()}')

    info = {
        'n_scenes': int(n),
        'scene_size': SCENE_SIZE,
        'pixel_scale': 0.03,
        'source': 'JADES DR5 morphology-classified ellipticals (Round Smooth / In-between / Cigar Shaped)',
        'photo_z_source': 'JADES DR3 EAZY (GOODS-N) / DR2 EAZY (GOODS-S)',
        'fields': args.fields,
    }
    (OUT_DIR / 'scene_info.json').write_text(json.dumps(info, indent=2))
    print(f'\nwrote {OUT_DIR}')


if __name__ == '__main__':
    main()
