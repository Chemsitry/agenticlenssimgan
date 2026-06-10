"""
prep_stamps_v6.py — Extract real galaxy stamps for INTERPOL light profiles.

Extracts two types of stamps from COSMOS-Web DR0.5 mosaics:
  1. Source galaxies: extended/star-forming (for lensed arcs)
  2. Lens galaxies: compact/elliptical (for foreground lens light)

v6 fixes over v3 INTERPOL attempt:
  - Cosine apodization on BOTH source and lens stamps (v3 only had it on lenses)
    This eliminates hard rectangular edges that get warped by lensing ray-trace
  - Gaussian denoising (sigma=1.0) on source stamps before normalization
    Prevents noise amplification in highly magnified arc regions
  - Larger source stamps (81x81 vs 65x65) for more apodization room

Output: prepped_mosaic_630/sources/ and prepped_mosaic_630/lenses/

Usage:
    .venv/bin/python3 prep_stamps_v6.py
    .venv/bin/python3 prep_stamps_v6.py --n_sources 500 --n_lenses 300
"""

import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']


def make_taper(size, edge):
    """Cosine taper: 1 in center, smooth to 0 over `edge` pixels at borders."""
    taper = np.ones((size, size))
    for d in range(edge):
        w = 0.5 * (1 - np.cos(np.pi * d / edge))
        taper[d, :] *= w
        taper[-(d+1), :] *= w
        taper[:, d] *= w
        taper[:, -(d+1)] *= w
    return taper


def detect_sources(sci, bg_median, bg_std, stamp_size, sigma_thresh, margin):
    """Find local maxima above threshold, return (y, x, flux, concentration) list."""
    ny, nx = sci.shape
    half = stamp_size // 2
    detect_thresh = bg_median + sigma_thresh * bg_std
    candidates = []
    tile_size = 4000

    for ty in range(margin, ny - margin, tile_size):
        for tx in range(margin, nx - margin, tile_size):
            ty_end = min(ty + tile_size, ny - margin)
            tx_end = min(tx + tile_size, nx - margin)
            tile = sci[ty:ty_end, tx:tx_end]

            bright = np.where(tile > detect_thresh)
            if len(bright[0]) == 0:
                continue

            for i in range(len(bright[0])):
                py = bright[0][i] + ty
                px = bright[1][i] + tx

                # Local maximum check (5x5)
                region = sci[py-2:py+3, px-2:px+3]
                if region.shape != (5, 5) or not np.isfinite(region).all():
                    continue
                if sci[py, px] != np.max(region):
                    continue

                # Measure in stamp aperture
                y0, x0 = py - half, px - half
                stamp = sci[y0:y0+stamp_size, x0:x0+stamp_size]
                if stamp.shape != (stamp_size, stamp_size):
                    continue
                if not np.all(np.isfinite(stamp)):
                    continue

                stamp_sub = stamp - bg_median
                yy, xx = np.mgrid[:stamp_size, :stamp_size]
                r = np.sqrt((yy - half)**2 + (xx - half)**2)

                core_flux = np.sum(stamp_sub[r < 8])
                total_flux = np.sum(stamp_sub[r < half])
                if total_flux <= 0:
                    continue

                concentration = core_flux / total_flux
                candidates.append((py, px, float(total_flux), float(concentration)))

    return candidates


def extract_stamps(band_data, candidates, n_want, stamp_size, bands,
                   taper_edge, denoise_sigma, min_isolation, valid_frac=0.95):
    """Extract multi-band stamps with background subtraction, denoising, apodization."""
    half = stamp_size // 2
    taper = make_taper(stamp_size, taper_edge)
    yy, xx = np.mgrid[:stamp_size, :stamp_size]
    r = np.sqrt((yy - half)**2 + (xx - half)**2)

    # Sort by flux (brightest first)
    candidates.sort(key=lambda x: -x[2])

    selected = []
    stamps_per_band = {band: [] for band in bands}
    catalog = []

    for cy, cx, flux_det, conc in candidates:
        if len(selected) >= n_want:
            break

        # Isolation check
        too_close = False
        for sy, sx in selected:
            if abs(cy - sy) < min_isolation and abs(cx - sx) < min_isolation:
                too_close = True
                break
        if too_close:
            continue

        # Extract all bands at same position
        y0, x0 = cy - half, cx - half
        raw_stamps = {}
        all_valid = True

        for band in bands:
            sci = band_data[band]['sci']
            stamp = sci[y0:y0+stamp_size, x0:x0+stamp_size]
            if stamp.shape != (stamp_size, stamp_size):
                all_valid = False
                break
            valid = np.isfinite(stamp) & (stamp != 0)
            if valid.mean() < valid_frac:
                all_valid = False
                break
            raw_stamps[band] = stamp.copy()

        if not all_valid:
            continue

        # Process: bg-sub, clip, denoise, apodize, normalize
        processed = {}
        ok = True

        for band in bands:
            stamp = raw_stamps[band].astype(np.float64)
            stamp -= band_data[band]['bg_median']
            stamp = np.nan_to_num(stamp, nan=0.0)

            # Sigma threshold: zero out noise pixels
            noise_thresh = 2.5 * band_data[band]['bg_std']
            stamp[stamp < noise_thresh] = 0.0

            # Gaussian denoise (critical for source stamps that get lensed)
            if denoise_sigma > 0:
                stamp = gaussian_filter(stamp, sigma=denoise_sigma)
                stamp[stamp < 0] = 0.0

            # Cosine apodization (eliminates hard edges)
            stamp *= taper

            total = stamp.sum()
            if total <= 0:
                ok = False
                break

            # Check centering
            frac_core = stamp[r < 12].sum() / total
            if frac_core < 0.25:
                ok = False
                break

            processed[band] = (stamp / total).astype(np.float32)

        if not ok:
            continue

        for band in bands:
            stamps_per_band[band].append(processed[band])

        selected.append((cy, cx))
        catalog.append({
            'y': int(cy), 'x': int(cx),
            'flux_detect': float(flux_det),
            'concentration': float(conc),
        })

        if len(selected) % 100 == 0:
            print(f'    {len(selected)}/{n_want}')

    return stamps_per_band, catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sources', type=int, default=500)
    parser.add_argument('--n_lenses', type=int, default=300)
    parser.add_argument('--source_stamp', type=int, default=81,
                        help='Source stamp size (larger than v3 65 for apodization room)')
    parser.add_argument('--lens_stamp', type=int, default=101)
    parser.add_argument('--prepped_dir', default='prepped_mosaic_630')
    args = parser.parse_args()

    mosaic_dir = Path('raw_data/1727_mosaic')
    prepped_dir = Path(args.prepped_dir)

    with open(prepped_dir / 'band_info.json') as f:
        band_info = json.load(f)

    # ── Open all mosaics ────────────────────────────────────────────────
    print('Opening mosaics...')
    band_data = {}
    for band in BANDS:
        fits_files = list((mosaic_dir / band).glob('mosaic*.fits'))
        if not fits_files:
            raise FileNotFoundError(f'No FITS file in {mosaic_dir / band}')
        hdul = fits.open(str(fits_files[0]), memmap=True)
        sci = hdul[1].data
        band_data[band] = {
            'sci': sci, 'hdul': hdul,
            'ny': sci.shape[0], 'nx': sci.shape[1],
            'bg_median': band_info[band]['bg_median'],
            'bg_std': band_info[band]['bg_std'],
        }
        print(f'  {band}: {sci.shape[0]}x{sci.shape[1]}  bg_std={band_info[band]["bg_std"]:.6f}')

    margin = max(args.source_stamp, args.lens_stamp) // 2 + 100

    # ── Source galaxies (extended, star-forming) ────────────────────────
    print(f'\nDetecting extended sources in F277W (15-sigma)...')
    src_det = band_data['F277W']
    src_candidates = detect_sources(
        src_det['sci'], src_det['bg_median'], src_det['bg_std'],
        args.source_stamp, sigma_thresh=15, margin=margin)

    # Filter: extended only (concentration < 0.45)
    src_candidates = [(y, x, f, c) for y, x, f, c in src_candidates if c < 0.45]
    # Require minimum S/N
    min_sn = 50 * src_det['bg_std'] * args.source_stamp
    src_candidates = [(y, x, f, c) for y, x, f, c in src_candidates if f > min_sn]
    print(f'  {len(src_candidates)} extended source candidates')

    print(f'\nExtracting {args.n_sources} source stamps ({args.source_stamp}x{args.source_stamp})...')
    print(f'  Denoise sigma=1.0, taper edge=15px')
    src_stamps, src_catalog = extract_stamps(
        band_data, src_candidates, args.n_sources, args.source_stamp, BANDS,
        taper_edge=15, denoise_sigma=1.0, min_isolation=args.source_stamp)

    src_dir = prepped_dir / 'sources'
    src_dir.mkdir(parents=True, exist_ok=True)
    n_src = len(src_catalog)
    print(f'  Extracted {n_src} source galaxies')

    for band in BANDS:
        arr = np.array(src_stamps[band])
        np.save(str(src_dir / f'stamps_{band}.npy'), arr)
        print(f'  {band}: {arr.shape}  {arr.nbytes/1e6:.1f} MB')

    src_info = {
        'n_sources': n_src, 'stamp_size': args.source_stamp,
        'pixel_scale': 0.03, 'detect_band': 'F277W', 'detect_sigma': 15,
        'concentration_max': 0.45, 'denoise_sigma': 1.0, 'taper_edge': 15,
        'sources': src_catalog,
    }
    with open(src_dir / 'source_info.json', 'w') as f:
        json.dump(src_info, f, indent=2)

    # ── Lens galaxies (compact, elliptical) ─────────────────────────────
    print(f'\nDetecting compact galaxies in F444W (25-sigma)...')
    lens_det = band_data['F444W']
    lens_candidates = detect_sources(
        lens_det['sci'], lens_det['bg_median'], lens_det['bg_std'],
        args.lens_stamp, sigma_thresh=25, margin=margin)

    # Filter: compact/elliptical (0.45 < concentration < 0.75)
    lens_candidates = [(y, x, f, c) for y, x, f, c in lens_candidates
                       if 0.45 < c < 0.75]
    # Require resolved extent: reject stars
    # (will be checked via mid-ring flux during extraction)
    print(f'  {len(lens_candidates)} compact galaxy candidates')

    print(f'\nExtracting {args.n_lenses} lens stamps ({args.lens_stamp}x{args.lens_stamp})...')
    print(f'  Denoise sigma=0.5, taper edge=20px')
    lens_stamps, lens_catalog = extract_stamps(
        band_data, lens_candidates, args.n_lenses, args.lens_stamp, BANDS,
        taper_edge=20, denoise_sigma=0.5, min_isolation=int(1.5 * args.lens_stamp))

    lens_dir = prepped_dir / 'lenses'
    lens_dir.mkdir(parents=True, exist_ok=True)
    n_lens = len(lens_catalog)
    print(f'  Extracted {n_lens} lens galaxies')

    for band in BANDS:
        arr = np.array(lens_stamps[band])
        np.save(str(lens_dir / f'stamps_{band}.npy'), arr)
        print(f'  {band}: {arr.shape}  {arr.nbytes/1e6:.1f} MB')

    lens_info = {
        'n_lenses': n_lens, 'stamp_size': args.lens_stamp,
        'pixel_scale': 0.03, 'detect_band': 'F444W', 'detect_sigma': 25,
        'concentration_range': [0.45, 0.75], 'denoise_sigma': 0.5, 'taper_edge': 20,
        'lenses': lens_catalog,
    }
    with open(lens_dir / 'lens_info.json', 'w') as f:
        json.dump(lens_info, f, indent=2)

    # Cleanup
    for band in BANDS:
        band_data[band]['hdul'].close()

    print(f'\nDone! {n_src} source + {n_lens} lens stamps in {prepped_dir}/')


if __name__ == '__main__':
    main()
