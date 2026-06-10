"""
prep_vela_v7.py — Extract VELA cosmological-sim source galaxies as v6-format stamps.

Reads downloaded VELA NIRCam tar.gz archives, walks the per-snapshot per-viewing-angle
FITS files, resamples to 0.03"/pix, crops/pads to 81x81, applies cosine apodization,
and saves stamps in the same .npy format prep_stamps_v6.py uses for COSMOS-Web sources.

The output (`prepped_mosaic_630/vela_sources/stamps_{band}.npy` + `vela_source_info.json`)
is a drop-in replacement for v6's `sources/`, so simulate_v7.py just changes the load path.

Pipeline:
  1. Read VELA catalog, filter to JWST NIRCam in our 4 bands at z=1-4 with mstar > 1e9
  2. Walk raw_data/vela/ to find extracted FITS files
  3. Group by (sim, cam, scale_factor) — keep only snapshots with all 4 bands present
  4. For each snapshot:
       - load image from each band's FITS (autodetect HDU)
       - resample from VELA pixel scale to 0.03"/pix (autodetected from header)
       - crop or pad to 81x81 centered on brightest pixel
       - cosine apodize (taper edge 15px, identical to prep_stamps_v6.py)
       - per-band normalize to sum=1
  5. Save stamps_{band}.npy + vela_source_info.json with snapshot catalog

Note on SED handling: stamps are per-band normalized (matches v6 behavior), so the
per-band relative fluxes from VELA are NOT preserved here. simulate_v7.py applies
its own SED via lens_colors / starforming_color_ratios. A future v8 could joint-
normalize to preserve VELA's intrinsic SED.

Usage:
    .venv/bin/python3 prep_vela_v7.py                       # extract from raw_data/vela/
    .venv/bin/python3 prep_vela_v7.py --max_stamps 500
    .venv/bin/python3 prep_vela_v7.py --vela_dir raw_data/vela --extract  # also untar archives first
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom

# v7 keeps VELA's pristine native resolution (0.015"/pix) so lenstronomy can
# ray-trace at high source-plane detail. Stamp is 162x162 to maintain the same
# physical extent as v6's 81x81 @ 0.03"/pix (both = 2.43 arcsec).
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
BANDS_LOWER = [b.lower() for b in BANDS]
TARGET_PIXEL_SCALE = 0.015  # arcsec/pix (VELA pristine native, ~half of v6)
STAMP_SIZE = 162            # px (162 * 0.015 = 2.43 arcsec, same as v6's 81 * 0.03)
TAPER_EDGE = 20             # px (smaller relative footprint than v6 to preserve more galaxy)

# Source-selection cuts (catalog-based)
Z_MIN, Z_MAX = 1.0, 4.0
MSTAR_MIN = 1e9
SFR_MIN = 1.0


# ── Reused from prep_stamps_v6.py ────────────────────────────────────────

def make_taper(size, edge):
    """Cosine taper: 1 in center, smooth to 0 over `edge` pixels at borders.

    Identical to prep_stamps_v6.py:make_taper.
    """
    taper = np.ones((size, size))
    for d in range(edge):
        w = 0.5 * (1 - np.cos(np.pi * d / edge))
        taper[d, :] *= w
        taper[-(d + 1), :] *= w
        taper[:, d] *= w
        taper[:, -(d + 1)] *= w
    return taper


# ── VELA-specific I/O ────────────────────────────────────────────────────

def extract_archives(vela_dir):
    """Untar any *.tar.gz archives that haven't been extracted yet."""
    print(f'Extracting tar.gz archives in {vela_dir}/...')
    n_extracted = 0
    for tarball in sorted(vela_dir.rglob('*.tar.gz')):
        # Check if already extracted by looking for any FITS sibling
        marker = tarball.with_suffix('.extracted')
        if marker.exists():
            continue
        print(f'  {tarball.name}')
        with tarfile.open(tarball, 'r:gz') as tf:
            tf.extractall(path=tarball.parent)
        marker.touch()
        n_extracted += 1
    print(f'  Extracted {n_extracted} new archives')


def find_fits_files(vela_dir):
    """Walk vela_dir for all NIRCam FITS files and return a dict keyed by
    (sim, cam, scale, band) -> Path.
    """
    files = {}
    for fits_path in vela_dir.rglob('hlsp_vela_jwst_nircam_*.fits'):
        # Filename: hlsp_vela_jwst_nircam_vela07-cam00-a0.250_f115w_v3_sim.fits
        stem = fits_path.stem
        parts = stem.split('_')
        # Expected: hlsp / vela / jwst / nircam / vela07-cam00-a0.250 / f115w / v3 / sim
        if len(parts) < 7:
            continue
        ident = parts[4]  # e.g. "vela07-cam00-a0.250"
        band = parts[5].upper()
        if band not in BANDS:
            continue
        try:
            sim, cam, scale_raw = ident.split('-')
        except ValueError:
            continue
        # Filename has scale like "a0.300"; catalog has "0.300"
        scale = scale_raw[1:] if scale_raw.startswith('a') else scale_raw
        files[(sim, cam, scale, band)] = fits_path
    return files


def autodetect_image_hdu(hdul):
    """Find the HDU with 2D image data.

    PREFER IMAGE_PRISTINE — VELA's intrinsic galaxy emission at ~0.015"/pix,
    NOT yet PSF-convolved. simulate_v7.py applies the real JWST PSF AFTER the
    lensing operation, which is the physically correct order (PSF acts at the
    observer side, not before light is bent by gravity).

    Fallback to IMAGE_PSF only if pristine isn't present (older VELA versions).
    """
    candidates = []
    for i, hdu in enumerate(hdul):
        if hdu.data is None or hdu.data.ndim != 2:
            continue
        name = (hdu.name or '').upper()
        candidates.append((i, name, hdu.data.shape))
    if not candidates:
        return None
    # PREFER pristine (intrinsic emission, no PSF) — physically correct for v7
    for i, name, shape in candidates:
        if any(k in name for k in ('PRISTINE', 'INTRINS', 'NOPSF')):
            return i
    # Fallback: PSF-convolved (less correct but always present)
    for i, name, shape in candidates:
        if 'IMAGE_PSF' in name or name == 'PSF':
            return i
    return candidates[0][0]


def get_pixel_scale(header):
    """Extract pixel scale (arcsec/pix) from FITS header. Tries CDELT, CD matrix,
    PIXSCALE, BAPSCALE — falls back to 0.06 (typical sunrise default for VELA gen3)."""
    for key in ('PIXSCALE', 'BAPSCALE', 'PIXELSCL'):
        if key in header:
            return float(header[key])
    if 'CDELT1' in header:
        return abs(float(header['CDELT1'])) * 3600  # deg -> arcsec
    if 'CD1_1' in header:
        cd11 = float(header['CD1_1'])
        cd12 = float(header.get('CD1_2', 0))
        return np.sqrt(cd11**2 + cd12**2) * 3600
    print(f'    WARNING: no pixel scale in header — defaulting to 0.06"/pix')
    return 0.06


def load_and_process(fits_path, target_scale=TARGET_PIXEL_SCALE,
                     stamp_size=STAMP_SIZE, taper=None):
    """Open one VELA FITS file, return a (stamp_size, stamp_size) processed stamp."""
    with fits.open(fits_path, memmap=False) as hdul:
        hdu_idx = autodetect_image_hdu(hdul)
        if hdu_idx is None:
            return None, None
        img = np.asarray(hdul[hdu_idx].data, dtype=np.float64)
        # IMAGE_PRISTINE has no pixel scale in its own header — must derive from
        # HDU 0 (IMAGE_PSF) which has PIXSCALE, NPIX, NPIXORIG.
        if hdu_idx == 0:
            scale = get_pixel_scale(hdul[0].header)
        else:
            psf_hdr = hdul[0].header
            psf_scale = get_pixel_scale(psf_hdr)
            npix = float(psf_hdr.get('NPIX', 374))
            npixorig = float(psf_hdr.get('NPIXORIG', 800))
            scale = psf_scale * (npix / npixorig)  # pristine is ~2x finer

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)

    # Resample from VELA scale to v6 target scale
    if abs(scale - target_scale) > 1e-4:
        zoom_factor = scale / target_scale
        img = zoom(img, zoom_factor, order=1, prefilter=False)
        img = np.clip(img, 0, None)

    # Center on brightest pixel
    if img.sum() <= 0:
        return None, scale
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    half = stamp_size // 2
    # Pad if necessary so the crop window fits
    pad_y = max(0, half - cy, half - (img.shape[0] - 1 - cy))
    pad_x = max(0, half - cx, half - (img.shape[1] - 1 - cx))
    if pad_y > 0 or pad_x > 0:
        img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)),
                     mode='constant', constant_values=0.0)
        cy += pad_y
        cx += pad_x

    y0, x0 = cy - half, cx - half
    stamp = img[y0:y0 + stamp_size, x0:x0 + stamp_size].copy()
    if stamp.shape != (stamp_size, stamp_size):
        return None, scale

    # Cosine taper (matches v6)
    if taper is not None:
        stamp *= taper

    return stamp.astype(np.float64), scale


def load_catalog_filter(catalog_path):
    """Read VELA catalog and return set of (sim, cam, scale_str, band) keys
    that pass the science cuts.
    """
    if not catalog_path.exists():
        print(f'  WARNING: catalog not found at {catalog_path} — keeping all snapshots')
        return None

    keep = set()
    n_total = 0
    with open(catalog_path) as f:
        header = f.readline().split()
        for line in f:
            n_total += 1
            parts = line.split()
            if len(parts) < 14:
                continue
            sim = parts[0]
            try:
                z = float(parts[1])
                scale = parts[2]
                cam = parts[3]
                instrument = parts[5]
                band_lower = parts[6]
                mstar = float(parts[9])
                sfr = float(parts[12])
            except (ValueError, IndexError):
                continue
            if instrument != 'nircam':
                continue
            if band_lower.upper() not in BANDS:
                continue
            if not (Z_MIN <= z <= Z_MAX):
                continue
            if mstar < MSTAR_MIN or sfr < SFR_MIN:
                continue
            keep.add((sim, cam, scale, band_lower.upper()))
    print(f'  Catalog: {len(keep)}/{n_total} entries pass cuts (z={Z_MIN}-{Z_MAX}, '
          f'M*>{MSTAR_MIN:.0e}, SFR>{SFR_MIN})')
    return keep


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract VELA source stamps for v7')
    parser.add_argument('--vela_dir', default='raw_data/vela', help='VELA download directory')
    parser.add_argument('--catalog', default='/tmp/vela_cat.txt',
                        help='VELA catalog (downloaded from MAST)')
    parser.add_argument('--out_dir', default='prepped_mosaic_630/vela_sources',
                        help='Output directory')
    parser.add_argument('--max_stamps', type=int, default=2000,
                        help='Max stamps to extract')
    parser.add_argument('--extract', action='store_true',
                        help='Untar archives before processing')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing stamps instead of overwriting')
    args = parser.parse_args()

    vela_dir = Path(args.vela_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vela_dir.exists():
        print(f'ERROR: VELA dir {vela_dir} does not exist. Run download.sh first.')
        sys.exit(1)

    if args.extract:
        extract_archives(vela_dir)

    print(f'\nLoading catalog from {args.catalog}')
    keep_set = load_catalog_filter(Path(args.catalog))

    print(f'\nWalking {vela_dir} for VELA NIRCam FITS files...')
    files = find_fits_files(vela_dir)
    print(f'  Found {len(files)} FITS files')

    # Apply catalog filter
    if keep_set is not None:
        files = {k: v for k, v in files.items() if k in keep_set}
        print(f'  After catalog cuts: {len(files)}')

    # Group by (sim, cam, scale) — must have all 4 bands
    snapshots = {}
    for (sim, cam, scale, band), path in files.items():
        snapshots.setdefault((sim, cam, scale), {})[band] = path
    snapshots = {k: v for k, v in snapshots.items() if len(v) == 4}
    print(f'  {len(snapshots)} snapshots have all 4 bands')

    if not snapshots:
        print('ERROR: no complete snapshots found.')
        sys.exit(1)

    snapshot_keys = sorted(snapshots.keys())
    if len(snapshot_keys) > args.max_stamps:
        # Subsample uniformly
        idx = np.linspace(0, len(snapshot_keys) - 1, args.max_stamps).astype(int)
        snapshot_keys = [snapshot_keys[i] for i in idx]
        print(f'  Subsampled to {len(snapshot_keys)} (max_stamps={args.max_stamps})')

    # Pre-compute taper
    taper = make_taper(STAMP_SIZE, TAPER_EDGE)

    # Process all snapshots
    stamps_per_band = {band: [] for band in BANDS}
    catalog_out = []
    detected_scales = []
    n_kept = 0
    n_skipped = 0

    for i, key in enumerate(snapshot_keys):
        sim, cam, scale = key
        bands_paths = snapshots[key]

        processed = {}
        ok = True
        for band in BANDS:
            stamp, vela_scale = load_and_process(bands_paths[band], taper=taper)
            if stamp is None:
                ok = False
                break
            if vela_scale is not None:
                detected_scales.append(vela_scale)

            total = stamp.sum()
            if total <= 0:
                ok = False
                break
            processed[band] = (stamp / total).astype(np.float32)

        if not ok:
            n_skipped += 1
            continue

        for band in BANDS:
            stamps_per_band[band].append(processed[band])
        catalog_out.append({'sim': sim, 'cam': cam, 'scale': scale})
        n_kept += 1

        if n_kept % 50 == 0:
            print(f'    [{n_kept}/{len(snapshot_keys)}] processed')

    print(f'\nKept {n_kept} stamps ({n_skipped} skipped)')

    if detected_scales:
        med_scale = float(np.median(detected_scales))
        print(f'  VELA pixel scale: median={med_scale:.4f}"/pix '
              f'(min={min(detected_scales):.4f}, max={max(detected_scales):.4f})')
    else:
        med_scale = 0.06

    # Save (or append to existing)
    print(f'\nSaving to {out_dir}/')
    for band in BANDS:
        new_arr = np.array(stamps_per_band[band])
        existing_path = out_dir / f'stamps_{band}.npy'
        if args.append and existing_path.exists():
            existing = np.load(str(existing_path))
            arr = np.concatenate([existing, new_arr], axis=0)
            print(f'  stamps_{band}.npy  {existing.shape[0]} existing + {new_arr.shape[0]} new = {arr.shape}  {arr.nbytes/1e6:.1f} MB')
        else:
            arr = new_arr
            print(f'  stamps_{band}.npy  shape={arr.shape}  {arr.nbytes/1e6:.1f} MB')
        np.save(str(existing_path), arr)

    # Update catalog
    if args.append and (out_dir / 'vela_source_info.json').exists():
        with open(out_dir / 'vela_source_info.json') as f:
            old_info = json.load(f)
        catalog_out = old_info.get('sources', []) + catalog_out
        n_kept = len(catalog_out)

    info = {
        'n_sources': n_kept,
        'stamp_size': STAMP_SIZE,
        'pixel_scale': TARGET_PIXEL_SCALE,
        'vela_native_scale_median': med_scale,
        'taper_edge': TAPER_EDGE,
        'detect_band': 'N/A (catalog-driven)',
        'concentration_max': None,
        'denoise_sigma': 0.0,
        'z_range': [Z_MIN, Z_MAX],
        'mstar_min': MSTAR_MIN,
        'sfr_min': SFR_MIN,
        'sources': catalog_out,
    }
    with open(out_dir / 'vela_source_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f'\nDone. {n_kept} VELA source stamps in {out_dir}/')


if __name__ == '__main__':
    main()
