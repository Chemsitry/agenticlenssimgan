"""
extend_fj_all_mosaics.py — measure JWST F115W/F150W/F277W magnitudes for every
FJ-calibration candidate that has a DESI sigma_v but is missing from sample_final,
using whichever JWST mosaic covers its sky position.

Supports four mosaic sources:
  - COSMOS-Web    (primer_cosmos and adjacent — data prep/raw_data/1727_mosaic/)
  - PRIMER-UDS    (primer_uds — cache_mosaics/primer-uds_<band>_drc_sci.fits.gz)
  - JADES GOODS-N (jades_gdn — cache_mosaics/jades-gdn_<band>_drz.fits)
  - CEERS EGS     (ceers_egs — cache_mosaics/fullceers_<band>.tar.gz, untarred to cache_mosaics/ceers_pointings/)

Each mosaic is read directly; photometry uses the FITS header's PIXAR_SR + BUNIT
to convert pixel sums to AB magnitudes (independent of any preprocessing).
"""
from __future__ import annotations
import json, sys, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

V9_ROOT       = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
DESI_CACHE    = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache')
MOSAIC_CACHE  = V9_ROOT / 'cache_mosaics'
COSMOS_MOSAIC_DIR = Path('/Users/nathankvinnesland/Desktop/data prep/raw_data/1727_mosaic')

BANDS = ['F115W', 'F150W', 'F277W']
APERTURE_RADIUS_ARCSEC = 1.0     # 1″ circular aperture (matches typical Kron radius for low-z ellipticals)
SKY_INNER_ARCSEC = 2.3
SKY_OUTER_ARCSEC = 3.2
STAMP_HALF_ARCSEC = 4.0          # cutout half-width

# Per-field → list of (band, path) pairs
def _mosaic_paths():
    out = {
        'primer_uds': {b: MOSAIC_CACHE / f'primer-uds_{b.lower()}_drc_sci.fits.gz' for b in BANDS},
        'jades_gdn':  {b: MOSAIC_CACHE / f'jades-gdn_{b.lower()}_drz.fits' for b in BANDS},
        'primer_cosmos': {b: list((COSMOS_MOSAIC_DIR / b).glob('mosaic*.fits'))[0] for b in BANDS},
    }
    # CEERS: tarballs extracted to ceers_pointings/fullceers_<band>/hlsp_..._sci.fits.gz
    ceers_dir = MOSAIC_CACHE / 'ceers_pointings'
    if ceers_dir.exists():
        per_band = {}
        for b in BANDS:
            files = sorted(ceers_dir.rglob(f'hlsp_ceers*{b.lower()}*_sci.fits*'))
            files = [f for f in files if 'bkgsub' not in f.name] or files
            if files:
                per_band[b] = files
        if per_band:
            out['ceers_egs'] = per_band
    return out


def maybe_extract_ceers():
    """Untar CEERS tarballs into cache_mosaics/ceers_pointings/ once.
    Skip silently if a tarball is incomplete (still downloading)."""
    out_dir = MOSAIC_CACHE / 'ceers_pointings'
    out_dir.mkdir(exist_ok=True)
    for band in BANDS:
        tarball = MOSAIC_CACHE / f'fullceers_{band.lower()}.tar.gz'
        if not tarball.exists():
            continue
        existing = list(out_dir.glob(f'*{band.lower()}*_i2d.fits*'))
        if existing:
            continue
        try:
            print(f'  extracting {tarball.name}...')
            with tarfile.open(tarball, 'r:gz') as tf:
                tf.extractall(out_dir, filter='data')
        except (tarfile.ReadError, EOFError, OSError) as e:
            print(f'    skipping (incomplete tarball: {e})')
    return out_dir


def mosaic_aperture_ab_mag(mosaic_path, ra, dec):
    """Open a mosaic FITS, take a small cutout at (ra, dec), and compute AB mag
    inside a 1″ aperture with local-sky subtraction. Returns AB mag or NaN.
    Returns NaN gracefully on any I/O error (incomplete downloads, corrupt files)."""
    try:
        return _mosaic_aperture_ab_mag_inner(mosaic_path, ra, dec)
    except (OSError, ValueError, TypeError, IndexError) as e:
        return float('nan')


def _mosaic_aperture_ab_mag_inner(mosaic_path, ra, dec):
    with fits.open(str(mosaic_path), memmap=True) as h:
        # Find the SCI HDU — usually 'SCI' or HDU 1
        sci_hdu = None
        for hdu in h:
            if hdu.name == 'SCI' or (sci_hdu is None and hdu.data is not None and hdu.data.ndim == 2):
                sci_hdu = hdu
                if hdu.name == 'SCI':
                    break
        if sci_hdu is None:
            return float('nan')
        sci = sci_hdu.data
        header = sci_hdu.header
        wcs = WCS(header)
        pixar_sr = float(header.get('PIXAR_SR', 0.0))
        bunit = str(header.get('BUNIT', '')).strip()
        # Pixel scale (arcsec/pix)
        if pixar_sr > 0:
            pix_arcsec = np.degrees(np.sqrt(pixar_sr)) * 3600.0
        else:
            try:
                pix_arcsec = abs(header.get('CDELT1', 0)) * 3600.0 or \
                             abs(header.get('CD1_1', 0)) * 3600.0
            except Exception:
                pix_arcsec = 0.03  # JWST NIRCam default
        # Convert RA/Dec → pixel
        try:
            x, y = wcs.all_world2pix(ra, dec, 0)
            x = float(x); y = float(y)
        except Exception:
            return float('nan')
        ny, nx = sci.shape
        half_pix = int(STAMP_HALF_ARCSEC / pix_arcsec)
        x0, y0 = int(x) - half_pix, int(y) - half_pix
        x1, y1 = x0 + 2*half_pix + 1, y0 + 2*half_pix + 1
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            return float('nan')
        stamp = sci[y0:y1, x0:x1].astype(np.float64)
        if not np.isfinite(stamp).any():
            return float('nan')
        stamp = np.nan_to_num(stamp, nan=0.0, posinf=0.0, neginf=0.0)
        # Build aperture + sky masks
        h_st, w_st = stamp.shape
        cy = (y - y0); cx = (x - x0)
        yy, xx = np.mgrid[:h_st, :w_st]
        rr = np.sqrt((yy - cy)**2 + (xx - cx)**2) * pix_arcsec
        src_mask = rr < APERTURE_RADIUS_ARCSEC
        sky_mask = (rr > SKY_INNER_ARCSEC) & (rr < SKY_OUTER_ARCSEC)
        if src_mask.sum() == 0 or sky_mask.sum() == 0:
            return float('nan')
        sky_per_pix = float(np.median(stamp[sky_mask]))
        src_sum = float((stamp[src_mask] - sky_per_pix).sum())
        # Convert to MJy depending on BUNIT
        if 'mjy/sr' in bunit.lower() or 'MJy/sr' in bunit:
            flux_mjy = src_sum * pixar_sr
        elif '10**(-9)' in bunit.lower() or 'njy' in bunit.lower():
            flux_mjy = src_sum * 1e-9 * 1e-6  # nJy → MJy
        elif 'mjy' in bunit.lower():
            flux_mjy = src_sum * 1e-6
        else:
            # Best guess: assume MJy/sr (most common for JWST i2d/drc)
            flux_mjy = src_sum * pixar_sr if pixar_sr > 0 else float('nan')
        if not np.isfinite(flux_mjy) or flux_mjy <= 0:
            return float('nan')
        flux_ujy = flux_mjy * 1e12
        return 23.9 - 2.5 * np.log10(flux_ujy)


def main():
    dv = pd.read_parquet(DESI_CACHE / 'dev_vdisp.parquet')
    sf = pd.read_parquet(DESI_CACHE / 'sample_final.parquet')
    have_targets = set(sf['TARGETID'].tolist())
    cand = dv[~dv['TARGETID'].isin(have_targets)].copy()
    print(f'dev_vdisp: {len(dv)}  ·  sample_final: {len(sf)}  ·  candidates: {len(cand)}')

    # Make sure CEERS is extracted (if tarballs present)
    maybe_extract_ceers()

    mosaic_paths = _mosaic_paths()
    print(f'mosaic paths available: {list(mosaic_paths.keys())}')

    new_rows = []
    for _, r in cand.iterrows():
        field = r['field_initial']
        if field not in mosaic_paths:
            continue
        per_band_paths = mosaic_paths[field]
        mags = {}
        for band in BANDS:
            paths = per_band_paths[band]
            if isinstance(paths, list):
                # multiple pointings (CEERS) — try each until one yields finite mag
                mag = float('nan')
                for p in paths:
                    m = mosaic_aperture_ab_mag(p, r['RA'], r['DEC'])
                    if np.isfinite(m):
                        mag = m; break
            else:
                mag = mosaic_aperture_ab_mag(paths, r['RA'], r['DEC'])
            mags[band] = mag
        if all(np.isfinite(mags[b]) for b in BANDS):
            new_rows.append({
                'RA': r['RA'], 'DEC': r['DEC'],
                'Z':  r['Z'],  'Z_PHOT_MEAN': r.get('Z_PHOT_MEAN', r['Z']),
                'VDISP': r['VDISP'], 'VDISP_IVAR': r['VDISP_IVAR'],
                'm115': mags['F115W'], 'm150': mags['F150W'], 'm277': mags['F277W'],
                'm115_err': np.nan, 'm150_err': np.nan, 'm277_err': np.nan,
                'TARGETID': r['TARGETID'],
                'field': f'{field}_mosaic',
                'jwst_match_arcsec': 0.0,
                'match_arcsec': float(r.get('match_arcsec', 0.0)),
            })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([sf, new_df], ignore_index=True)
    out_path = V9_ROOT / 'sample_final_extended.parquet'
    combined.to_parquet(out_path, index=False)
    print(f'\nwrote {out_path}: {len(combined)} galaxies '
          f'({len(sf)} catalog + {len(new_df)} mosaic-measured)')
    if len(new_df):
        print('\nNew mosaic-measured rows:')
        print(new_df[['RA','DEC','Z','VDISP','m115','m150','m277','field']].to_string(index=False))


if __name__ == '__main__':
    main()
