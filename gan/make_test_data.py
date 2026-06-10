"""
make_test_data.py — Create synthetic data for pipeline testing.

This generates small but realistic-shaped fake data so you can run the
full Stage 0 pipeline (prep_real_targets -> audit -> pca) without needing
the 160 GB COSMOS-Web mosaics.

It creates:
  - raw_data/1727_mosaic/{band}/mosaic_fake.fits  (tiny synthetic FITS mosaics)
  - prepped_mosaic/band_info.json                 (realistic sky stats)
  - output/v3/images_{band}.npy                   (fake sim images)
  - output/v3/sources_{band}.npy                  (fake arc masks)
  - output/v3/lensed.npy, theta_Es.npy, etc.

The fake mosaics are 2000x2000 pixels (vs the real ~36000x30000), so
prep_real_targets will extract fewer cutouts but the code path is identical.

Usage:
    python -m gan.make_test_data
    python -m gan.make_test_data --n-sim 500 --n-real 200 --mosaic-size 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']

# Realistic sky levels from a typical COSMOS-Web pointing (MJy/sr)
BAND_SKY = {'F115W': 0.00312, 'F150W': 0.00428, 'F277W': 0.00615, 'F444W': 0.00891}
BAND_NOISE = {'F115W': 0.00028, 'F150W': 0.00031, 'F277W': 0.00045, 'F444W': 0.00062}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-sim',       type=int, default=500,
                   help='Number of fake sim images (default: 500)')
    p.add_argument('--n-real',      type=int, default=300,
                   help='Number of real cutouts to aim for (default: 300)')
    p.add_argument('--size',        type=int, default=125)
    p.add_argument('--mosaic-size', type=int, default=3000,
                   help='Fake mosaic side length in pixels (default: 3000). '
                        'Must be large enough to fit --n-real cutouts.')
    p.add_argument('--seed',        type=int, default=7)
    return p.parse_args()


def make_fake_mosaic(path: Path, ny: int, nx: int,
                     bg_median: float, bg_noise: float,
                     rng: np.random.Generator) -> None:
    """Create a minimal 2-extension FITS file (Primary + SCI) with Gaussian noise."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Gaussian sky + a handful of fake "galaxies" (2D Gaussians) so radial
    # profiles and power spectra look somewhat realistic
    sci = rng.normal(bg_median, bg_noise, (ny, nx)).astype(np.float32)

    # Add ~50 fake blobs of varying brightness
    n_blobs = 50
    ys = rng.integers(20, ny - 20, n_blobs)
    xs = rng.integers(20, nx - 20, n_blobs)
    fluxes = rng.exponential(bg_noise * 30, n_blobs)
    sizes  = rng.uniform(2, 8, n_blobs)
    yy, xx = np.mgrid[:ny, :nx]
    for y0, x0, flux, sigma in zip(ys, xs, fluxes, sizes):
        sci += flux * np.exp(-((yy - y0)**2 + (xx - x0)**2) / (2 * sigma**2))

    primary = fits.PrimaryHDU()
    primary.header['SIMPLE'] = True

    image_hdu = fits.ImageHDU(data=sci)
    image_hdu.header['EXTNAME'] = 'SCI'
    image_hdu.header['PHOTMJSR'] = 0.0  # not used by prep_real_targets
    image_hdu.header['PIXAR_SR'] = 2.1154e-14

    hdul = fits.HDUList([primary, image_hdu])
    hdul.writeto(str(path), overwrite=True)
    print(f'  Created {path}  ({ny}x{nx}, {path.stat().st_size/1e6:.1f} MB)')


def make_band_info(prepped_dir: Path) -> None:
    band_info = {}
    for band in BANDS:
        band_info[band] = {
            'pixar_sr':    2.1154e-14,
            'photmjsr':    None,
            'xposure':     6184.0,
            'pixel_scale': 0.03,
            'bg_median':   BAND_SKY[band],
            'bg_std':      BAND_NOISE[band],
            'n_backgrounds': 0,
            'n_psf_stars':   0,
            'mosaic_shape':  [3000, 3000],
        }
    prepped_dir.mkdir(parents=True, exist_ok=True)
    with open(prepped_dir / 'band_info.json', 'w') as f:
        json.dump(band_info, f, indent=2)
    print(f'  Created prepped_mosaic/band_info.json')


def make_fake_sim(out_dir: Path, n: int, size: int,
                  rng: np.random.Generator) -> None:
    """
    Create fake sim images.  Sim images differ from real in a detectable way:
    they have a bright central blob (the "lens") which the real cutouts don't.
    This ensures the PCA baseline has something to find.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size // 2, size // 2

    for band in BANDS:
        noise = BAND_NOISE[band]
        images = rng.normal(0, noise, (n, size, size)).astype(np.float32)
        # Add a centered Gaussian lens galaxy to every sim image
        for i in range(n):
            amp   = rng.exponential(noise * 50)
            sigma = rng.uniform(3, 8)
            images[i] += amp * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
            # Add a faint arc at a random offset from center
            arc_r = rng.uniform(5, 20)
            arc_theta = rng.uniform(0, 2 * np.pi)
            ay = cy + arc_r * np.sin(arc_theta)
            ax = cx + arc_r * np.cos(arc_theta)
            arc_amp = rng.exponential(noise * 10)
            images[i] += arc_amp * np.exp(-((yy - ay)**2 + (xx - ax)**2) / (2 * 3**2))

        np.save(str(out_dir / f'images_{band}.npy'), images)
        # sources: arc-only (just the faint arc without lens light)
        sources = np.zeros((n, size, size), dtype=np.float32)
        np.save(str(out_dir / f'sources_{band}.npy'), sources)
        print(f'  {band}: {images.shape} -> output/v3/images_{band}.npy')

    np.save(str(out_dir / 'lensed.npy'),   np.ones(n, dtype=np.int32))
    np.save(str(out_dir / 'theta_Es.npy'), rng.uniform(0.2, 1.5, n).astype(np.float32))
    np.save(str(out_dir / 'z_lens.npy'),   rng.uniform(0.1, 2.0, n).astype(np.float32))
    np.save(str(out_dir / 'z_source.npy'), rng.uniform(0.5, 7.0, n).astype(np.float32))
    np.save(str(out_dir / 'masses.npy'),   rng.uniform(11, 13, n).astype(np.float32))
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({'note': 'SYNTHETIC TEST DATA — not real simulations'}, f)
    print(f'  Created {n} synthetic sim images in output/v3/')


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    print(f'Creating synthetic test data (size={args.size}, '
          f'n_sim={args.n_sim}, mosaic={args.mosaic_size}x{args.mosaic_size})')
    print('NOTE: This is FAKE data for pipeline testing only.\n')

    # 1. Fake FITS mosaics
    print('1/3  Fake COSMOS-Web mosaics:')
    mosaic_dir = Path('raw_data/1727_mosaic')
    for band in BANDS:
        path = mosaic_dir / band / 'mosaic_fake.fits'
        make_fake_mosaic(path, args.mosaic_size, args.mosaic_size,
                         BAND_SKY[band], BAND_NOISE[band], rng)

    # 2. band_info.json (replaces running prep_mosaic.py)
    print('\n2/3  prepped_mosaic/band_info.json:')
    make_band_info(Path('prepped_mosaic'))

    # 3. Fake sim output (replaces running simulate_v3.py)
    print(f'\n3/3  Fake sim images (output/v3/):')
    make_fake_sim(Path('output/v3'), args.n_sim, args.size, rng)

    print('\nDone.  You can now run the Stage 0 pipeline:')
    print('  python -m gan.data.prep_real_targets --smoke-test')
    print('  python -m gan.data.audit_distributions --n-sim 200 --n-real 200')
    print('  python -m gan.baselines.pca --n-sim 300 --n-real 200')
    print()
    print('When you have the real COSMOS-Web mosaics:')
    print('  rm -rf raw_data/ prepped_mosaic/ output/v3/')
    print('  ln -s /path/to/real/mosaics raw_data')
    print('  python prep_mosaic.py')
    print('  python simulate_v3.py --n 30000')


if __name__ == '__main__':
    main()
