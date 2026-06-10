"""
prep_lenses_v8.py — Raw mosaic cutouts with absolute calibration.

Key insight (why all prior v8 attempts failed):

  Every past version either (a) segmented out the non-galaxy pixels — which
  killed the natural sky-noise texture around the galaxy and made the lens
  look "too clean" vs. real neighbors, or (b) preserved raw sky but then got
  amp-scaled by 1000× inside the simulator because the stamp was normalized
  to sum=1 — producing a bright square box.

  The fix is to put the stamp in the SAME absolute flux units as the
  background cutouts (mjysr_to_sim), so:
    - sky pixels inside the stamp match sky pixels in the bg cutout
    - peak-matching amp is a modest number (order 1) that doesn't blow up sky
    - the galaxy's natural PSF diffraction spikes and halo → sky transition
      are preserved exactly as they appear in real JWST data.

Operation (per band, per galaxy):
  1. Open the DR0.5 mosaic for this band.
  2. Crop a 201x201 cutout at the position recorded in lenses_v7/lens_info.json.
  3. Subtract global bg_median (so sky averages to ~0).
  4. Multiply by mjysr_to_sim — puts stamp in the same units as backgrounds.
  5. Apply a gentle cosine edge taper (15 px at the stamp border) so the
     rectangular stamp doesn't show a box edge during INTERPOL rendering.
  6. NO normalize-by-sum. NO segmentation. NO denoise.

Output: prepped_mosaic_630/lenses_v8/
"""

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
MOSAIC_DIR = Path('raw_data/1727_mosaic')
PREPPED_DIR = Path('prepped_mosaic_630')
SRC_INFO = PREPPED_DIR / 'lenses_v7' / 'lens_info.json'
OUT_DIR = PREPPED_DIR / 'lenses_v8'

STAMP_SIZE = 201
EDGE_TAPER_PX = 15

sum_to_flux = 6.501853565914121  # same constant simulate_v8 uses


def make_edge_taper(size, edge):
    t = np.ones((size, size), dtype=np.float32)
    for d in range(edge):
        w = 0.5 * (1 - np.cos(np.pi * d / edge))
        t[d, :] *= w
        t[-(d + 1), :] *= w
        t[:, d] *= w
        t[:, -(d + 1)] *= w
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SRC_INFO) as f:
        src_info = json.load(f)
    with open(PREPPED_DIR / 'band_info.json') as f:
        band_info = json.load(f)

    positions = [(int(l['y']), int(l['x'])) for l in src_info['lenses']]
    n = len(positions)
    print(f'Extracting {n} raw cutouts with absolute calibration')
    print(f'  stamp_size={STAMP_SIZE}  edge_taper={EDGE_TAPER_PX}px')

    taper = make_edge_taper(STAMP_SIZE, EDGE_TAPER_PX)
    half = STAMP_SIZE // 2

    out = {b: np.zeros((n, STAMP_SIZE, STAMP_SIZE), dtype=np.float32)
           for b in BANDS}
    dropped = set()

    for band in BANDS:
        info = band_info[band]
        mjysr_to_sim = info['pixar_sr'] * 1e15 * sum_to_flux
        bg_median = info['bg_median']
        fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
        if not fits_files:
            raise FileNotFoundError(f'no mosaic for {band}')
        print(f'  {band}: bg_median={bg_median:.4e}  '
              f'mjysr_to_sim={mjysr_to_sim:.4f}')
        with fits.open(str(fits_files[0]), memmap=True) as hdul:
            sci = hdul[1].data
            for i, (y, x) in enumerate(positions):
                y0, x0 = y - half, x - half
                stamp = sci[y0:y0 + STAMP_SIZE, x0:x0 + STAMP_SIZE]
                if stamp.shape != (STAMP_SIZE, STAMP_SIZE):
                    dropped.add(i)
                    continue
                s = stamp.astype(np.float32)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                s = s - bg_median                 # sky → ~0 mean
                s = s * mjysr_to_sim               # to sim units (same as bg)
                s = s * taper                      # gentle box-edge fade
                out[band][i] = s

                if (i + 1) % 50 == 0 and band == BANDS[0]:
                    print(f'    {i + 1}/{n}')

    if dropped:
        keep = np.array([i for i in range(n) if i not in dropped])
        for b in BANDS:
            out[b] = out[b][keep]
        n = len(keep)
        print(f'  dropped {len(dropped)} edge stamps; keeping {n}')

    for b in BANDS:
        path = OUT_DIR / f'stamps_{b}.npy'
        np.save(str(path), out[b])
        peaks = out[b].max(axis=(1, 2))
        print(f'  {b}: {out[b].shape}  peak median={np.median(peaks):.2f} '
              f'(range {peaks.min():.2f}-{peaks.max():.2f})')

    info_out = {
        'n_lenses': n,
        'stamp_size': STAMP_SIZE,
        'pixel_scale': 0.03,
        'edge_taper_px': EDGE_TAPER_PX,
        'absolute_calibration': True,
        'derived_from': 'raw DR0.5 mosaic cutouts at lenses_v7 positions',
        'notes': 'Raw cutouts in sim units (mjysr_to_sim). No normalize-by-sum, '
                 'no segmentation, no denoise. Only bg_median subtraction + '
                 'gentle edge taper. Keeps natural PSF spikes, halo-to-sky '
                 'transition, and sky noise at same level as backgrounds.',
        'source_positions': src_info['lenses'],
    }
    with open(OUT_DIR / 'lens_info.json', 'w') as f:
        json.dump(info_out, f, indent=2)
    print(f'\nDone — {n} absolute-calibrated lens stamps in {OUT_DIR}/')


if __name__ == '__main__':
    main()
