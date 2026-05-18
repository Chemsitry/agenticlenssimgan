"""
prep_scenes_v10.py — build a lens-scene catalog from MULTIPLE JWST mosaics
(COSMOS-Web + PRIMER-UDS + CEERS + JADES GOODS-N) using ALL galaxies in those
footprints with a known photo-z. No morphology filter at the parent stage —
just rely on brightness/compactness QA to keep things lens-like.

Inputs:
  - desi_jwst_dev/cache/dev_in_jwst.parquet  (5,008 galaxies with photo-z in JWST fields)
  - data prep/raw_data/1727_mosaic/<band>/         COSMOS-Web mosaic
  - data prep v9_consistent/cache_mosaics/primer-uds_<band>_drc_sci.fits.gz
  - data prep v9_consistent/cache_mosaics/ceers_pointings/*.fits.gz (extracted)
  - data prep v9_consistent/cache_mosaics/jades-gdn_<band>_drz.fits

Outputs:
  data prep v9_consistent/prepped_scenes_v10/
    scenes_F115W.npy / F150W / F277W / F444W   (N, 630, 630)
    manifest.parquet  one row per kept scene with RA, Dec, z_phot, field, etc.
    scene_info.json   summary
"""
from __future__ import annotations
import json, tarfile, sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

DATA_PREP_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep')
DESI_CACHE     = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache')
V9_ROOT        = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
COSMOS_MOSAIC_DIR = DATA_PREP_ROOT / 'raw_data' / '1727_mosaic'
MOSAIC_CACHE   = V9_ROOT / 'cache_mosaics'
CEERS_POINTINGS_DIR = MOSAIC_CACHE / 'ceers_pointings'

OUT_DIR  = V9_ROOT / 'prepped_scenes_v10'
BANDS    = ['F115W', 'F150W', 'F277W', 'F444W']

SCENE_SIZE = 630
PEAK_MIN = 100              # F444W central peak (sim units)
COMPACTNESS_MAX = 0.35
NAN_TOLERANCE = 0.25
EDGE_MARGIN = 250

# Per-field calibration constants from COSMOS-Web (we re-use the band_info.json
# bg_median + sum_to_flux scaling for all mosaics; this gives the same "sim units"
# convention as the v8 pipeline so simulate_v9_consistent.py just works).
SUM_TO_FLUX = 6.501853565914121
COSMOS_BAND_INFO = json.loads((DATA_PREP_ROOT / 'prepped_mosaic_630' / 'band_info.json').read_text())

# Generic JWST i2d/drz/drc reader: returns (sci_array, wcs, m2s_factor) where
# m2s_factor scales pixel values to sim units consistent with COSMOS-Web.
# - COSMOS-Web mosaic has BUNIT=MJy/sr and PIXAR_SR; we apply pixar_sr*1e15*sum_to_flux
# - JADES drz files: also MJy/sr, same factor
# - PRIMER drc files: BUNIT='10.0*nanoJansky' (per pixel!); need conversion
# - CEERS i2d files: MJy/sr
def open_mosaic(path):
    h = fits.open(str(path), memmap=True)
    sci_hdu = None
    for hdu in h:
        if hdu.data is None: continue
        if hdu.name == 'SCI' or hdu.data.ndim == 2:
            sci_hdu = hdu
            if hdu.name == 'SCI': break
    if sci_hdu is None:
        h.close(); return None
    header = sci_hdu.header
    bunit = str(header.get('BUNIT', '')).strip()
    pixar_sr = float(header.get('PIXAR_SR', 0.0))
    if 'mjy/sr' in bunit.lower():
        m2s = pixar_sr * 1e15 * SUM_TO_FLUX
    elif 'nanojansky' in bunit.lower() or 'njy' in bunit.lower():
        # PRIMER DRC: pixel value × scaling = nJy. Convert nJy → MJy → sim units:
        # flux_MJy = nJy × 1e-9 × 1e-6.  Then sim = flux_MJy × 1e15 × sum_to_flux.
        scale = 1.0
        if '10.0' in bunit or '10*' in bunit:
            scale = 10.0
        m2s = scale * 1e-9 * 1e-6 * 1e15 * SUM_TO_FLUX  # per nJy → sim
    else:
        # default to MJy/sr
        m2s = pixar_sr * 1e15 * SUM_TO_FLUX if pixar_sr > 0 else 1.0
    return h, sci_hdu, m2s


# Per-field mosaic finder
def get_mosaic_paths():
    out = {}
    # COSMOS-Web (primer_cosmos field): single mosaic per band
    if (COSMOS_MOSAIC_DIR / 'F115W').exists():
        try:
            out['primer_cosmos'] = {
                b: list((COSMOS_MOSAIC_DIR / b).glob('mosaic*.fits'))[0] for b in BANDS
            }
        except Exception:
            pass
    # PRIMER-UDS
    if all((MOSAIC_CACHE / f'primer-uds_{b.lower()}_drc_sci.fits.gz').exists() for b in BANDS[:3]):
        out['primer_uds'] = {
            b: MOSAIC_CACHE / f'primer-uds_{b.lower()}_drc_sci.fits.gz' for b in BANDS[:3]
        }
        # F444W: optional — if missing, drop the F444W stamp for this field
        f444 = MOSAIC_CACHE / 'primer-uds_f444w_drc_sci.fits.gz'
        if f444.exists():
            out['primer_uds']['F444W'] = f444
    # JADES GOODS-N
    if all((MOSAIC_CACHE / f'jades-gdn_{b.lower()}_drz.fits').exists() for b in BANDS[:3]):
        out['jades_gdn'] = {b: MOSAIC_CACHE / f'jades-gdn_{b.lower()}_drz.fits' for b in BANDS[:3]}
        f444 = MOSAIC_CACHE / 'jades-gdn_f444w_drz.fits'
        if f444.exists():
            out['jades_gdn']['F444W'] = f444
    # CEERS — single merged mosaic per band, inside ceers_pointings/fullceers_<band>/
    if CEERS_POINTINGS_DIR.exists():
        per_band = {}
        for b in BANDS[:3]:
            pat = b.lower()
            files = sorted(CEERS_POINTINGS_DIR.rglob(f'hlsp_ceers*{pat}*_sci.fits*'))
            # prefer non-bkgsub
            files = [f for f in files if 'bkgsub' not in f.name] or files
            if files:
                per_band[b] = files
        if per_band:
            out['ceers_egs'] = per_band
    return out


def maybe_extract_ceers():
    CEERS_POINTINGS_DIR.mkdir(exist_ok=True)
    for band in BANDS[:3]:
        tarball = MOSAIC_CACHE / f'fullceers_{band.lower()}.tar.gz'
        if not tarball.exists():
            continue
        if list(CEERS_POINTINGS_DIR.glob(f'*{band.lower()}*_i2d.fits*')):
            continue
        try:
            print(f'  extracting {tarball.name}...')
            with tarfile.open(tarball, 'r:gz') as tf:
                tf.extractall(CEERS_POINTINGS_DIR, filter='data')
        except (tarfile.ReadError, EOFError, OSError) as e:
            print(f'    skipping (incomplete tarball: {e})')


def cut_stamp(sci_hdu, wcs, ra, dec, scene_size, m2s, bg_median=0.0):
    """Return scene_size x scene_size cutout in sim units, or None if out-of-bounds/too-NaN."""
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
    nan_frac = (~np.isfinite(s)).mean()
    if nan_frac > NAN_TOLERANCE:
        return None
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = (s - bg_median) * m2s
    return s, float(nan_frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-field', type=int, default=200,
                        help='Max scenes per JWST field (default 200)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('loading dev_in_jwst (5k galaxies with photo-z in JWST footprints)...')
    cand = pd.read_parquet(DESI_CACHE / 'dev_in_jwst.parquet')
    print(f'  total candidates: {len(cand):,}')
    print(f'  by field: {cand["field_initial"].value_counts().to_dict()}')

    maybe_extract_ceers()
    mosaics_by_field = get_mosaic_paths()
    print(f'\navailable mosaic fields: {list(mosaics_by_field.keys())}')

    raw = {b: [] for b in BANDS}
    manifest_rows = []
    half = SCENE_SIZE // 2

    # Compactness mask templates
    yy, xx = np.mgrid[:SCENE_SIZE, :SCENE_SIZE]
    rr = np.sqrt((yy - half)**2 + (xx - half)**2)
    mask_small = rr < 3
    mask_medium = rr < 15

    for field, paths in mosaics_by_field.items():
        sub = cand[cand['field_initial'] == field].copy()
        if args.max_per_field:
            sub = sub.head(args.max_per_field)
        print(f'\n=== {field}: trying {len(sub)} candidates ===')

        # Pre-open every mosaic file for this field, ONCE.
        # Structure: opened[band] = list of (handle, sci_hdu, m2s, wcs)
        opened_per_band = {}
        for band in BANDS:
            if band not in paths:
                continue
            pths = paths[band] if isinstance(paths[band], list) else [paths[band]]
            opened_list = []
            for p in pths:
                op = open_mosaic(p)
                if op is None:
                    continue
                h, sci_hdu, m2s = op
                try:
                    wcs = WCS(sci_hdu.header)
                    opened_list.append((h, sci_hdu, m2s, wcs))
                except Exception:
                    h.close()
            if opened_list:
                opened_per_band[band] = opened_list

        try:
            for _, row in sub.iterrows():
                scene_per_band = {}
                success = True
                for band in BANDS:
                    if band not in opened_per_band:
                        scene_per_band[band] = None
                        continue
                    bg_med = COSMOS_BAND_INFO[band]['bg_median'] if field == 'primer_cosmos' else 0.0
                    stamp_result = None
                    for (h, sci_hdu, m2s, wcs) in opened_per_band[band]:
                        res = cut_stamp(sci_hdu, wcs, row['RA'], row['DEC'],
                                        SCENE_SIZE, m2s, bg_med)
                        if res is not None:
                            stamp_result = res[0]
                            break
                    if stamp_result is None:
                        success = False
                        break
                    scene_per_band[band] = stamp_result

                if not success:
                    continue

                # Apply brightness + compactness QA on F444W if available, else F277W
                ref_band = 'F444W' if scene_per_band.get('F444W') is not None else 'F277W'
                ref = scene_per_band[ref_band]
                center_peak = ref[half-20:half+20, half-20:half+20].max()
                if center_peak < PEAK_MIN:
                    continue
                fm = ref[mask_medium].sum()
                if fm <= 0:
                    continue
                compactness = float(ref[mask_small].sum() / fm)
                if compactness > COMPACTNESS_MAX:
                    continue

                # Keep this scene
                for band in BANDS:
                    if scene_per_band.get(band) is None:
                        raw[band].append(np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.float32))
                    else:
                        raw[band].append(scene_per_band[band])
                manifest_rows.append({
                    'RA': float(row['RA']), 'DEC': float(row['DEC']),
                    'Z_PHOT_MEAN': float(row['Z_PHOT_MEAN']),
                    'Z_PHOT_STD':  float(row['Z_PHOT_STD']),
                    'TYPE':        str(row['TYPE']),
                    'field':       field,
                    'F444W_peak':  float(center_peak),
                    'compactness': compactness,
                    'has_F444W':   scene_per_band.get('F444W') is not None,
                })
        finally:
            # Always close every mosaic file we opened for this field
            for band_opened in opened_per_band.values():
                for h, _, _, _ in band_opened:
                    try: h.close()
                    except Exception: pass

        print(f'  passed: {sum(1 for r in manifest_rows if r["field"] == field)}')

    n_keep = len(manifest_rows)
    if n_keep == 0:
        print('No scenes passed QA.'); return

    for band in BANDS:
        arr = np.stack(raw[band], axis=0)
        path = OUT_DIR / f'scenes_{band}.npy'
        np.save(str(path), arr)
        print(f'  {band}: {arr.shape}  {arr.nbytes/1e6:.1f} MB -> {path.name}')

    manifest = pd.DataFrame(manifest_rows)
    manifest['scene_idx'] = np.arange(len(manifest))
    manifest.to_parquet(OUT_DIR / 'manifest.parquet', index=False)
    print(f'\nmanifest: {len(manifest)} scenes  ·  field counts: {manifest["field"].value_counts().to_dict()}')
    print(f'z_phot range: {manifest["Z_PHOT_MEAN"].min():.2f} – {manifest["Z_PHOT_MEAN"].max():.2f}')

    info = {
        'n_scenes': n_keep,
        'scene_size': SCENE_SIZE,
        'pixel_scale': 0.03,
        'source': 'Multi-field JWST mosaics (COSMOS-Web, PRIMER-UDS, CEERS, JADES GOODS-N)',
        'photo_z_filter': 'Z_PHOT_STD<0.15, MASKBITS=0, any LS galaxy TYPE (DEV/EXP/SER/REX)',
        'brightness_filter': f'central F444W (or F277W if no F444W) peak > {PEAK_MIN} sim units',
        'compactness_filter': f'sum(r<3)/sum(r<15) < {COMPACTNESS_MAX}',
        'nan_tolerance': NAN_TOLERANCE,
    }
    (OUT_DIR / 'scene_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
