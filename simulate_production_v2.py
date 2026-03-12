"""
simulate_production_v2.py

NERSC production script: generates 50,000 gravitational lens simulation images
using improved physics (empirical PSF, SLACS-calibrated parameters, SIE+SHEAR).

Designed for NERSC Perlmutter / Cori with multiprocessing.Pool across CPU cores.
A 32-core node at ~9 ms/image = ~7.5 min for 50k images.

Usage:
    python simulate_production_v2.py --n_images 50000 --n_workers 32 --out /pscratch/sd/n/natekv/v2

    # Test run (100 images, 4 workers):
    python simulate_production_v2.py --n_images 100 --n_workers 4 --out output/v2_prod_test

Dependencies:
    lenstronomy, scipy, numpy, astropy
    prepped/psf_median.npy   (run python build_psf.py first)
    prepped/real_backgrounds.npy  (run python prep_jwst.py first)
"""

import argparse
import os
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

# ── Configuration ─────────────────────────────────────────────────────────────

PIXELS     = 125
PIXEL_SIZE = 0.031        # arcsec/pix — NIRCam SW channel
EXP_TIME   = 1380

sum_to_flux  = 6.501853565914121
PIXAR_SR     = 2.29232933396454e-14
mjysr_to_sim = PIXAR_SR * 1e15 * sum_to_flux   # ≈ 149.0

# These will be set from the loaded PSF kernel at process init
_PSF_KERNEL    = None
_REAL_BGS_SIM  = None
_BG_RNG        = None

# ── Worker initializer (runs once per process in the pool) ────────────────────

def _init_worker(psf_kernel, real_bgs_sim):
    """Load shared arrays into worker-local globals to avoid repeated IPC."""
    global _PSF_KERNEL, _REAL_BGS_SIM, _BG_RNG
    _PSF_KERNEL   = psf_kernel
    _REAL_BGS_SIM = real_bgs_sim
    _BG_RNG       = np.random.default_rng()
    # Disable numba on macOS / some NERSC environments
    os.environ.setdefault('NUMBA_DISABLE_JIT', '1')


def get_real_background():
    return _REAL_BGS_SIM[_BG_RNG.integers(len(_REAL_BGS_SIM))]


# ── Helper: stellar mass ──────────────────────────────────────────────────────

def stellar_mass(M, z):
    mM10=11.88; mu=0.019; mM00=0.0282; nu=-0.72
    gamma0=0.556; gamma1=-0.26; beta0=1.06; beta1=0.17
    M1   = 10**(11.88*(z+1)**mu)
    mM0  = mM00*(z+1)**nu
    gamma = gamma0*(z+1)**gamma1
    beta  = beta1*z + beta0
    shmr  = 2*mM0/((M/M1)**(-beta)+(M/M1)**gamma)
    return shmr * M


def ML_ratio(z):
    return 10**(2.15259223299506*np.log10(z)+6.61731435158865)


# ── Core simulation function ──────────────────────────────────────────────────

def simulate_one_v2(lensed: bool, seed: int):
    """
    Generate one simulated lensed or non-lensed image.

    Returns
    -------
    dict with keys: image, image_source, theta_E, z_lens, z_source, mass, lensed
    """
    rng = np.random.default_rng(seed)

    # ── SLACS-calibrated redshifts ─────────────────────────────────────────
    z_lens = float(truncnorm.rvs(
        a=(0.05 - 0.3) / 0.15,
        b=(0.90 - 0.3) / 0.15,
        loc=0.3, scale=0.15,
        random_state=int(rng.integers(int(1e9)))
    ))

    for _ in range(100):
        z_source = float(truncnorm.rvs(
            a=(0.6 - 1.5) / 0.8,
            b=(3.0 - 1.5) / 0.8,
            loc=1.5, scale=0.8,
            random_state=int(rng.integers(int(1e9)))
        ))
        if z_source > z_lens + 0.05:
            break
    z_source = max(z_source, z_lens + 0.05)

    # ── SLACS-calibrated velocity dispersion -> Einstein radius ────────────
    sigma_v = float(truncnorm.rvs(
        a=(100 - 215) / 50,
        b=(400 - 215) / 50,
        loc=215, scale=50,
        random_state=int(rng.integers(int(1e9)))
    ))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E    = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    if lensed and not (0.5 <= theta_E <= 1.5):
        return simulate_one_v2(lensed=lensed, seed=int(rng.integers(int(1e9))))

    log_mass = float(rng.uniform(13.0, 14.5))
    mass     = 10**log_mass
    mStar    = stellar_mass(mass, z_lens)

    # ── Shape parameters ───────────────────────────────────────────────────
    e1, e2        = rng.normal(0, 0.15, size=2).clip(-0.5, 0.5)
    R_sersic_lens = float(truncnorm.rvs(0, 3, loc=0.3, scale=0.3,
                                        random_state=int(rng.integers(int(1e9)))))
    n_sersic_lens = float(rng.uniform(2, 6))
    R_sersic_src  = float(truncnorm.rvs(0, 3, loc=0.15, scale=0.15,
                                        random_state=int(rng.integers(int(1e9)))))
    n_sersic_src  = float(rng.uniform(1, 4))
    e1s, e2s      = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    # ── Source position relative to Einstein radius ────────────────────────
    if lensed and theta_E > 0:
        src_offset = float(rng.uniform(0.0, 0.3 * theta_E))
        src_angle  = float(rng.uniform(0, 2 * np.pi))
        center_x   = src_offset * np.cos(src_angle)
        center_y   = src_offset * np.sin(src_angle)
    else:
        center_x, center_y = rng.normal(0, 0.25, size=2)

    # ── External shear: polar -> Cartesian ────────────────────────────────
    # lenstronomy SHEAR uses gamma1/gamma2 (not gamma_ext/psi_ext)
    gamma_ext = float(rng.uniform(0.0, 0.08))
    psi_ext   = float(rng.uniform(0, np.pi))
    gamma1    = gamma_ext * np.cos(2 * psi_ext)
    gamma2    = gamma_ext * np.sin(2 * psi_ext)

    # ── Lenstronomy setup ──────────────────────────────────────────────────
    kernel = _PSF_KERNEL.astype(np.float64)
    kernel = np.clip(kernel, 0, None)  # remove tiny negatives from median stack
    kernel /= kernel.sum()
    psf_class = PSF(psf_type='PIXEL',
                    kernel_point_source=kernel,
                    kernel_point_source_normalisation=True)

    kwargs_data = {
        'background_rms': 0,
        'exposure_time': EXP_TIME,
        'ra_at_xy_0':  -PIXELS/2 * PIXEL_SIZE,
        'dec_at_xy_0': -PIXELS/2 * PIXEL_SIZE,
        'transform_pix2angle': np.array([[PIXEL_SIZE, 0.], [0., PIXEL_SIZE]]),
        'image_data': np.zeros((PIXELS, PIXELS))
    }
    kwargs_numerics = {'supersampling_factor': 3, 'supersampling_convolution': True}

    data_class             = ImageData(**kwargs_data)
    source_model_class     = LightModel(['SERSIC_ELLIPSE'])
    lens_light_model_class = LightModel(['SERSIC_ELLIPSE'])
    lens_model_class       = LensModel(['SIE', 'SHEAR'],
                                        z_lens=z_lens, z_source=z_source)

    image_model = ImageModel(
        data_class=data_class, psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )

    kwargs_lens = [
        {'theta_E': theta_E, 'e1': float(e1), 'e2': float(e2),
         'center_x': 0., 'center_y': 0.},
        {'gamma1': gamma1, 'gamma2': gamma2}   # Cartesian shear components
    ]
    kwargs_lens_light = [{'amp': 1, 'R_sersic': R_sersic_lens,
                          'n_sersic': n_sersic_lens, 'e1': float(e1), 'e2': float(e2),
                          'center_x': 0., 'center_y': 0.}]
    kwargs_source = [{'amp': 1, 'R_sersic': R_sersic_src, 'n_sersic': n_sersic_src,
                      'e1': float(e1s), 'e2': float(e2s),
                      'center_x': float(center_x), 'center_y': float(center_y)}]

    # ── Calibrate source amplitude ─────────────────────────────────────────
    scale_up     = 10**float(rng.uniform(0, 2))
    src_flux_njy = float(truncnorm.rvs(0, 3, loc=50, scale=80,
                                       random_state=int(rng.integers(int(1e9)))))
    calc_sum_src = sum_to_flux * src_flux_njy

    img_src = image_model.image(kwargs_lens, kwargs_source,
                                kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                                source_add=True, lens_light_add=False)
    s_src = np.sum(img_src)
    if s_src <= 0:
        return simulate_one_v2(lensed=lensed, seed=int(rng.integers(int(1e9))))
    kwargs_source[0]['amp'] = (calc_sum_src / s_src) * scale_up

    if not lensed:
        kwargs_source[0]['amp'] = 0.0

    # ── Calibrate lens light amplitude ─────────────────────────────────────
    lStar         = mStar / ML_ratio(max(z_lens, 0.01))
    calc_sum_lens = sum_to_flux * lStar
    img_lens = image_model.image(kwargs_lens, kwargs_source,
                                 kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                                 source_add=False, lens_light_add=True)
    s_lens = np.sum(img_lens)
    if s_lens <= 0:
        return simulate_one_v2(lensed=lensed, seed=int(rng.integers(int(1e9))))
    kwargs_lens_light[0]['amp'] = calc_sum_lens / s_lens

    # ── Arc/lens brightness floor ──────────────────────────────────────────
    # arc_flux = calc_sum_src * scale_up;  lens_flux = calc_sum_lens
    # Both scale linearly with their amps, so the ratio needs no extra render.
    ARC_LENS_MIN_RATIO = 1e-3
    if lensed and calc_sum_lens > 0:
        arc_flux = calc_sum_src * scale_up
        if arc_flux < ARC_LENS_MIN_RATIO * calc_sum_lens:
            boost = (ARC_LENS_MIN_RATIO * calc_sum_lens) / arc_flux
            kwargs_source[0]['amp'] *= boost

    # ── Final images ───────────────────────────────────────────────────────
    image = image_model.image(kwargs_lens, kwargs_source,
                              kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                              source_add=True, lens_light_add=True)
    image_source = image_model.image(kwargs_lens, kwargs_source,
                                     kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                                     source_add=True, lens_light_add=False)

    bg = get_real_background()
    return {
        'image':        (image + bg).astype(np.float32),
        'image_source': (image_source + bg).astype(np.float32),
        'theta_E':      theta_E,
        'z_lens':       z_lens,
        'z_source':     z_source,
        'mass':         mass,
        'lensed':       float(lensed),
    }


# ── Job wrapper for multiprocessing ──────────────────────────────────────────

def _worker(args):
    lensed, seed = args
    try:
        return simulate_one_v2(lensed=lensed, seed=seed)
    except Exception as exc:
        return {'error': str(exc), 'seed': seed, 'lensed': float(lensed)}


# ── Append-and-save helper (NERSC-style incremental save) ────────────────────

def append_and_save(path: str, key: str, new_data: np.ndarray) -> None:
    file_path = os.path.join(path, f'{key}.npy')
    if os.path.exists(file_path):
        old = np.load(file_path)
        combined = np.concatenate([old, new_data], axis=0)
    else:
        combined = new_data
    np.save(file_path, combined)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NERSC production lens simulation (v2 physics)')
    parser.add_argument('--n_images',  type=int, default=50000,
                        help='Total images (half lensed, half non-lensed)')
    parser.add_argument('--n_workers', type=int,
                        default=min(32, mp.cpu_count()),
                        help='Number of parallel workers (default: cpu_count up to 32)')
    parser.add_argument('--out',       default='/pscratch/sd/n/natekv/v2',
                        help='Output directory')
    parser.add_argument('--psf',       default='prepped/psf_median.npy',
                        help='Path to empirical PSF kernel')
    parser.add_argument('--backgrounds', default='prepped/real_backgrounds.npy',
                        help='Path to real_backgrounds.npy')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Save to disk every N images (default: 500)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Master RNG seed')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load shared data ───────────────────────────────────────────────────
    print(f'Loading PSF kernel from {args.psf}')
    psf_kernel = np.load(args.psf).astype(np.float32)
    psf_kernel = np.clip(psf_kernel, 0, None)  # remove tiny negatives
    psf_kernel /= psf_kernel.sum()
    print(f'  shape={psf_kernel.shape}  sum={psf_kernel.sum():.6f}')

    print(f'Loading backgrounds from {args.backgrounds}')
    real_bgs    = np.load(args.backgrounds)
    real_bgs_sim = (real_bgs * mjysr_to_sim).astype(np.float32)
    print(f'  {len(real_bgs_sim)} patches  dtype={real_bgs_sim.dtype}')

    # ── Build job list ─────────────────────────────────────────────────────
    n_lensed  = args.n_images // 2
    n_nonlens = args.n_images - n_lensed
    lensed_flags = [False] * n_nonlens + [True] * n_lensed
    rng_main  = np.random.default_rng(args.seed)
    rng_main.shuffle(lensed_flags)
    seeds = [int(rng_main.integers(int(1e9))) for _ in range(args.n_images)]
    jobs  = list(zip(lensed_flags, seeds))

    print(f'\nJob: {args.n_images} images '
          f'({n_lensed} lensed, {n_nonlens} non-lensed) '
          f'| {args.n_workers} workers')
    print(f'Output: {args.out}')

    # ── Run ────────────────────────────────────────────────────────────────
    t0 = time.time()
    n_errors = 0

    # Buffers
    buf_images  = []
    buf_srcs    = []
    buf_labels  = []
    buf_tEs     = []
    buf_zl      = []
    buf_zs      = []
    buf_masses  = []

    ctx = mp.get_context('spawn')   # spawn is safer for lenstronomy on macOS/NERSC
    with ctx.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(psf_kernel, real_bgs_sim)
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, jobs,
                                                        chunksize=4)):
            if 'error' in result:
                n_errors += 1
                print(f'  [WARN] Error at job {i}: {result["error"]}')
                continue

            buf_images.append(result['image'])
            buf_srcs.append(result['image_source'])
            buf_labels.append(result['lensed'])
            buf_tEs.append(result['theta_E'])
            buf_zl.append(result['z_lens'])
            buf_zs.append(result['z_source'])
            buf_masses.append(result['mass'])

            # Periodic save
            if len(buf_images) >= args.batch_size:
                append_and_save(args.out, 'images',       np.array(buf_images))
                append_and_save(args.out, 'image_source', np.array(buf_srcs))
                append_and_save(args.out, 'lensed',       np.array(buf_labels))
                append_and_save(args.out, 'theta_Es',     np.array(buf_tEs))
                append_and_save(args.out, 'z_lens',       np.array(buf_zl))
                append_and_save(args.out, 'z_source',     np.array(buf_zs))
                append_and_save(args.out, 'masses',       np.array(buf_masses))
                n_saved = np.load(os.path.join(args.out, 'images.npy')).shape[0]
                elapsed = time.time() - t0
                rate    = n_saved / elapsed
                eta     = (args.n_images - n_saved) / max(rate, 1e-6)
                print(f'  [{n_saved}/{args.n_images}]  '
                      f'{elapsed:.0f}s elapsed  {rate:.1f} img/s  '
                      f'ETA {eta:.0f}s  errors={n_errors}',
                      flush=True)
                buf_images = []; buf_srcs  = []; buf_labels = []
                buf_tEs    = []; buf_zl    = []; buf_zs     = []
                buf_masses = []

    # Flush remaining
    if buf_images:
        append_and_save(args.out, 'images',       np.array(buf_images))
        append_and_save(args.out, 'image_source', np.array(buf_srcs))
        append_and_save(args.out, 'lensed',       np.array(buf_labels))
        append_and_save(args.out, 'theta_Es',     np.array(buf_tEs))
        append_and_save(args.out, 'z_lens',       np.array(buf_zl))
        append_and_save(args.out, 'z_source',     np.array(buf_zs))
        append_and_save(args.out, 'masses',       np.array(buf_masses))

    total = time.time() - t0
    n_final = np.load(os.path.join(args.out, 'images.npy')).shape[0]
    print(f'\n=== Done: {n_final} images in {total:.0f}s '
          f'({total/max(n_final,1):.2f}s/img)  errors={n_errors} ===')
    for f in sorted(os.listdir(args.out)):
        fp = os.path.join(args.out, f)
        if f.endswith('.npy'):
            print(f'  {f:<30} {os.path.getsize(fp)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
