"""
simulate_multiband.py

Multi-band JWST gravitational lens simulation using COSMOS-Web backgrounds.
Generates images in F115W, F150W, F277W, F444W simultaneously.

Each lens system is rendered at all 4 bands with:
  - Per-band empirical PSF from COSMOS-Web stars
  - Per-band real background patches from COSMOS-Web
  - SED-dependent flux scaling (elliptical template for lens, star-forming for source)
  - Poisson shot noise calibrated per band
  - Native pixel scales: SW=0.031"/pix (125x125), LW=0.063"/pix (63x63)

Usage:
    .venv/bin/python3 simulate_multiband.py          # 10 test images
    .venv/bin/python3 simulate_multiband.py --n 1000  # full dataset
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.stats import truncnorm

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

# ── Paths ──────────────────────────────────────────────────────────────────

PREPPED_DIR = Path('prepped_cosmicwebb')
OUT_DIR     = Path('output/multiband')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
SW_BANDS = {'F115W', 'F150W'}
LW_BANDS = {'F277W', 'F444W'}

# ── Load per-band calibration info ─────────────────────────────────────────

with open(PREPPED_DIR / 'band_info.json') as f:
    BAND_INFO = json.load(f)

# ── Per-band constants ─────────────────────────────────────────────────────

sum_to_flux = 6.501853565914121   # nJy -> sim unit conversion

BAND_CONFIG = {}
for band in BANDS:
    info = BAND_INFO[band]
    is_sw = band in SW_BANDS
    pixels = 125 if is_sw else 63
    pixel_scale = 0.031 if is_sw else 0.063  # use nominal scales

    pixar_sr = info['pixar_sr']
    photmjsr = info['photmjsr']
    xposure  = info['xposure']
    gain     = 2.05  # NIRCam typical

    mjysr_to_sim = pixar_sr * 1e15 * sum_to_flux
    sim_to_elec  = (1.0 / mjysr_to_sim) / photmjsr * xposure * gain

    BAND_CONFIG[band] = {
        'pixels': pixels,
        'pixel_scale': pixel_scale,
        'pixar_sr': pixar_sr,
        'photmjsr': photmjsr,
        'xposure': xposure,
        'mjysr_to_sim': mjysr_to_sim,
        'sim_to_elec': sim_to_elec,
    }

# Print summary
print("Band configurations:")
for band in BANDS:
    cfg = BAND_CONFIG[band]
    print(f"  {band}: {cfg['pixels']}x{cfg['pixels']} @ {cfg['pixel_scale']}\"/pix  "
          f"mjysr_to_sim={cfg['mjysr_to_sim']:.2f}  sim_to_elec={cfg['sim_to_elec']:.2f}")

# ── Load backgrounds and PSFs ──────────────────────────────────────────────

backgrounds = {}
psf_kernels = {}

for band in BANDS:
    cfg = BAND_CONFIG[band]
    bg_raw = np.load(str(PREPPED_DIR / band / 'backgrounds.npy'))
    bg_raw = np.nan_to_num(bg_raw, nan=0.0)  # replace NaN with 0
    backgrounds[band] = (bg_raw * cfg['mjysr_to_sim']).astype(np.float32)
    print(f"  {band} backgrounds: {backgrounds[band].shape}, "
          f"median={np.median(backgrounds[band]):.2f} sim units")

    kernel = np.load(str(PREPPED_DIR / band / 'psf_median.npy')).astype(np.float64)
    kernel = np.clip(kernel, 0, None)
    kernel /= kernel.sum()
    psf_kernels[band] = kernel
    print(f"  {band} PSF: {kernel.shape}, peak at {np.unravel_index(np.argmax(kernel), kernel.shape)}")

rng_bg = np.random.default_rng()

def get_background(band):
    bgs = backgrounds[band]
    return bgs[rng_bg.integers(len(bgs))]

# ── SED color ratios ───────────────────────────────────────────────────────
# Approximate rest-frame SED colors relative to F115W for:
#   - Elliptical galaxy (lens): old stellar population, red SED
#   - Star-forming galaxy (source): blue SED with emission lines
#
# These are approximate flux ratios at z~0.3 (lens) and z~1.5 (source).
# Normalized so F115W = 1.0.
#
# For a more sophisticated approach, one would use stellar population
# synthesis (e.g., FSPS/python-fsps) to compute k-corrections per redshift.
# These fixed ratios are a reasonable first approximation.

def elliptical_color_ratios(z_lens):
    """Approximate NIRCam flux ratios for an elliptical at z_lens, relative to F115W.

    Ellipticals are red: brighter at longer wavelengths.
    At z~0.3, the 4000A break falls between F115W and F150W.
    """
    # Simple interpolation based on typical elliptical SED shape
    # Higher z -> 4000A break shifts redder -> SW bands get fainter
    f150w = 1.3 + 0.3 * z_lens    # slightly brighter than F115W
    f277w = 1.8 + 0.8 * z_lens    # significantly brighter (rest-frame NIR)
    f444w = 2.0 + 1.2 * z_lens    # brightest (rest-frame NIR)
    return {'F115W': 1.0, 'F150W': f150w, 'F277W': f277w, 'F444W': f444w}


def starforming_color_ratios(z_source):
    """Approximate NIRCam flux ratios for a star-forming galaxy at z_source.

    Star-forming galaxies are blue: relatively brighter at shorter wavelengths.
    At z~1.5, Lyman break is well below F115W so all bands detect the source.
    """
    # Blue SED: drops toward longer wavelengths
    f150w = 0.9 - 0.05 * z_source
    f277w = 0.5 - 0.05 * z_source
    f444w = 0.3 - 0.03 * z_source
    # Floor at 0.05 (always some flux)
    return {
        'F115W': 1.0,
        'F150W': max(f150w, 0.05),
        'F277W': max(f277w, 0.05),
        'F444W': max(f444w, 0.05),
    }


# ── Poisson noise ─────────────────────────────────────────────────────────

def add_poisson_noise(image_sim, band, rng=None):
    """Add Poisson shot noise to noiseless sim image."""
    if rng is None:
        rng = np.random.default_rng()
    s2e = BAND_CONFIG[band]['sim_to_elec']
    electrons = np.clip(image_sim * s2e, 0, None)
    noisy = rng.poisson(electrons).astype(np.float64)
    return (noisy / s2e).astype(np.float32)


# ── Mass / stellar-mass helpers ────────────────────────────────────────────

def stellar_mass(M, z):
    mM10=11.88; mu=0.019; mM00=0.0282; nu=-0.72
    gamma0=0.556; gamma1=-0.26; beta0=1.06; beta1=0.17
    M1 = 10**(11.88*(z+1)**mu)
    mM0 = mM00*(z+1)**nu
    gamma = gamma0*(z+1)**gamma1
    beta  = beta1*z + beta0
    shmr  = 2*mM0/((M/M1)**(-beta)+(M/M1)**gamma)
    return shmr * M

def ML_ratio(z):
    return 10**(2.15259223299506*np.log10(z)+6.61731435158865)


# ── Make lenstronomy objects per band ──────────────────────────────────────

def make_psf_obj(band):
    kernel = psf_kernels[band]
    return PSF(psf_type='PIXEL',
               kernel_point_source=kernel,
               kernel_point_source_normalisation=True)

def make_kwargs_data(band):
    cfg = BAND_CONFIG[band]
    pixels = cfg['pixels']
    pixel_scale = cfg['pixel_scale']
    return {
        'background_rms': 0,
        'exposure_time': cfg['xposure'],
        'ra_at_xy_0': -pixels/2 * pixel_scale,
        'dec_at_xy_0': -pixels/2 * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0.], [0., pixel_scale]]),
        'image_data': np.zeros((pixels, pixels))
    }

KWARGS_NUMERICS = {
    'supersampling_factor': 3,
    'supersampling_convolution': True,
}

ARC_LENS_MIN_RATIO = 1e-2

# ── Multi-band simulation ─────────────────────────────────────────────────

def simulate_one_multiband(lensed=True, seed=None):
    """Simulate one lens system in all 4 bands.

    Returns dict with keys per band, each containing (image, image_source).
    Also returns shared params: theta_E, z_lens, z_source, mass, mStar.
    """
    rng = np.random.default_rng(seed)

    # ── SLACS-calibrated redshifts ─────────────────────────────────────
    z_lens = float(truncnorm.rvs(
        a=(0.05 - 0.3) / 0.15, b=(0.90 - 0.3) / 0.15,
        loc=0.3, scale=0.15,
        random_state=int(rng.integers(int(1e9)))))

    z_src_min = max(0.6, z_lens + 0.05)
    for _ in range(100):
        z_source = float(truncnorm.rvs(
            a=(0.6 - 1.5) / 0.8, b=(3.0 - 1.5) / 0.8,
            loc=1.5, scale=0.8,
            random_state=int(rng.integers(int(1e9)))))
        if z_source > z_lens + 0.05:
            break
    z_source = max(z_source, z_src_min)

    # ── Velocity dispersion -> Einstein radius ─────────────────────────
    sigma_v = float(truncnorm.rvs(
        a=(100 - 215) / 50, b=(400 - 215) / 50,
        loc=215, scale=50,
        random_state=int(rng.integers(int(1e9)))))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    if lensed and not (0.5 <= theta_E <= 1.5):
        return simulate_one_multiband(lensed=lensed, seed=int(rng.integers(int(1e9))))

    # ── Halo mass (galaxy-scale) ───────────────────────────────────────
    log_mass = float(rng.uniform(11.5, 13.0))
    mass = 10**log_mass
    mStar = stellar_mass(mass, z_lens)

    # ── Galaxy shape parameters (shared across bands) ──────────────────
    e1, e2 = rng.normal(0, 0.15, size=2).clip(-0.5, 0.5)
    R_sersic_lens = float(truncnorm.rvs(0, 3, loc=0.3, scale=0.3,
                          random_state=int(rng.integers(int(1e9)))))
    n_sersic_lens = float(rng.uniform(2, 6))
    R_sersic_src = float(truncnorm.rvs(0, 3, loc=0.15, scale=0.15,
                         random_state=int(rng.integers(int(1e9)))))
    n_sersic_src = float(rng.uniform(1, 4))
    e1s, e2s = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    if lensed and theta_E > 0:
        src_offset = float(rng.uniform(0.0, 0.3 * theta_E))
        src_angle = float(rng.uniform(0, 2 * np.pi))
        center_x = src_offset * np.cos(src_angle)
        center_y = src_offset * np.sin(src_angle)
    else:
        center_x, center_y = rng.normal(0, 0.25, size=2)

    gamma_ext = float(rng.uniform(0.0, 0.08))
    psi_ext = float(rng.uniform(0, np.pi))
    gamma1 = gamma_ext * np.cos(2 * psi_ext)
    gamma2 = gamma_ext * np.sin(2 * psi_ext)

    # ── Source calibration (done once in F115W, then scaled) ───────────
    scale_up = 10**float(rng.uniform(0, 2))
    src_flux_njy = float(truncnorm.rvs(0, 3, loc=50, scale=80,
                         random_state=int(rng.integers(int(1e9)))))
    calc_sum_src_f115w = sum_to_flux * src_flux_njy
    lStar = mStar / ML_ratio(max(z_lens, 0.01))
    calc_sum_lens_f115w = sum_to_flux * lStar

    # SED color ratios
    lens_colors = elliptical_color_ratios(z_lens)
    src_colors = starforming_color_ratios(z_source)

    # ── Render in each band ────────────────────────────────────────────
    band_results = {}

    for band in BANDS:
        cfg = BAND_CONFIG[band]
        pixels = cfg['pixels']
        pixel_scale = cfg['pixel_scale']

        kwargs_data = make_kwargs_data(band)
        data_class = ImageData(**kwargs_data)
        psf_class = make_psf_obj(band)

        source_model = LightModel(['SERSIC_ELLIPSE'])
        lens_light_model = LightModel(['SERSIC_ELLIPSE'])
        lens_model = LensModel(['SIE', 'SHEAR'], z_lens=z_lens, z_source=z_source)

        image_model = ImageModel(
            data_class=data_class, psf_class=psf_class,
            lens_model_class=lens_model,
            source_model_class=source_model,
            lens_light_model_class=lens_light_model,
            kwargs_numerics=KWARGS_NUMERICS)

        kwargs_lens = [
            {'theta_E': theta_E, 'e1': float(e1), 'e2': float(e2),
             'center_x': 0., 'center_y': 0.},
            {'gamma1': gamma1, 'gamma2': gamma2}
        ]
        kwargs_lens_light = [{
            'amp': 1, 'R_sersic': R_sersic_lens, 'n_sersic': n_sersic_lens,
            'e1': float(e1), 'e2': float(e2), 'center_x': 0., 'center_y': 0.}]
        kwargs_source = [{
            'amp': 1, 'R_sersic': R_sersic_src, 'n_sersic': n_sersic_src,
            'e1': float(e1s), 'e2': float(e2s),
            'center_x': float(center_x), 'center_y': float(center_y)}]

        # Scale fluxes by SED color ratio relative to F115W
        calc_sum_src = calc_sum_src_f115w * src_colors[band]
        calc_sum_lens = calc_sum_lens_f115w * lens_colors[band]

        # Calibrate source amplitude
        img_src = image_model.image(kwargs_lens, kwargs_source,
                                    kwargs_lens_light=kwargs_lens_light,
                                    source_add=True, lens_light_add=False)
        s_src = np.sum(img_src)
        if s_src <= 0:
            return simulate_one_multiband(lensed=lensed, seed=int(rng.integers(int(1e9))))
        kwargs_source[0]['amp'] = (calc_sum_src / s_src) * scale_up

        if not lensed:
            kwargs_source[0]['amp'] = 0.0

        # Calibrate lens light amplitude
        img_lens = image_model.image(kwargs_lens, kwargs_source,
                                     kwargs_lens_light=kwargs_lens_light,
                                     source_add=False, lens_light_add=True)
        s_lens = np.sum(img_lens)
        if s_lens <= 0:
            return simulate_one_multiband(lensed=lensed, seed=int(rng.integers(int(1e9))))
        kwargs_lens_light[0]['amp'] = calc_sum_lens / s_lens

        # Arc/lens floor
        if lensed and calc_sum_lens > 0:
            arc_flux = calc_sum_src * scale_up
            lens_flux = calc_sum_lens
            if arc_flux < ARC_LENS_MIN_RATIO * lens_flux:
                boost = (ARC_LENS_MIN_RATIO * lens_flux) / arc_flux
                kwargs_source[0]['amp'] *= boost

        # Final renders
        image = image_model.image(kwargs_lens, kwargs_source,
                                  kwargs_lens_light=kwargs_lens_light,
                                  source_add=True, lens_light_add=True)
        image_source = image_model.image(kwargs_lens, kwargs_source,
                                         kwargs_lens_light=kwargs_lens_light,
                                         source_add=True, lens_light_add=False)

        band_results[band] = {
            'image': image.astype(np.float32),
            'image_source': image_source.astype(np.float32),
        }

    return band_results, theta_E, z_lens, z_source, mass, mStar


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='Number of images (half lensed, half not)')
    args = parser.parse_args()

    N = args.n
    N_EACH = N // 2

    print(f"\nGenerating {N} images ({N_EACH} lensed + {N_EACH} non-lensed) in {len(BANDS)} bands...")

    # Storage
    all_images = {band: np.zeros((N, BAND_CONFIG[band]['pixels'], BAND_CONFIG[band]['pixels']),
                                 dtype=np.float32) for band in BANDS}
    all_sources = {band: np.zeros_like(all_images[band]) for band in BANDS}
    labels = np.zeros(N)
    theta_Es = np.zeros(N)
    z_lenses = np.zeros(N)
    z_sources = np.zeros(N)
    masses = np.zeros(N)

    rng_main = np.random.default_rng(99)
    jobs = ([(i, False) for i in range(N_EACH)] +
            [(i + N_EACH, True) for i in range(N_EACH)])

    t0 = time.time()
    for idx, lensed in jobs:
        label = 'lensed' if lensed else 'non-lensed'
        print(f'  [{idx+1}/{N}] {label}...', end=' ', flush=True)
        t1 = time.time()

        result = simulate_one_multiband(lensed=lensed, seed=int(rng_main.integers(int(1e9))))
        band_results, theta_E, z_lens, z_source, mass, mStar = result

        rng_noise = np.random.default_rng(int(rng_main.integers(int(1e9))))

        for band in BANDS:
            img = band_results[band]['image']
            img_noisy = add_poisson_noise(img, band, rng=rng_noise)
            bg = get_background(band)
            all_images[band][idx] = img_noisy + bg
            all_sources[band][idx] = band_results[band]['image_source']

        labels[idx] = 1.0 if lensed else 0.0
        theta_Es[idx] = theta_E
        z_lenses[idx] = z_lens
        z_sources[idx] = z_source
        masses[idx] = mass

        arc_ratio = ''
        if lensed:
            src_sum = np.sum(np.clip(band_results['F115W']['image_source'], 0, None))
            lens_sum = np.sum(np.clip(band_results['F115W']['image'] - band_results['F115W']['image_source'], 0, None))
            if lens_sum > 0:
                arc_ratio = f'  arc/lens={src_sum/lens_sum:.3f}'
        print(f'{time.time()-t1:.1f}s  θE={theta_E:.2f}" zl={z_lens:.2f} zs={z_source:.2f}{arc_ratio}')

    total = time.time() - t0
    print(f'\nDone — {N} images in {total:.1f}s ({total/N:.2f}s/image)')

    # ── Save ───────────────────────────────────────────────────────────
    for band in BANDS:
        np.save(str(OUT_DIR / f'images_{band}.npy'), all_images[band])
        np.save(str(OUT_DIR / f'sources_{band}.npy'), all_sources[band])
    np.save(str(OUT_DIR / 'lensed.npy'), labels)
    np.save(str(OUT_DIR / 'theta_Es.npy'), theta_Es)
    np.save(str(OUT_DIR / 'z_lens.npy'), z_lenses)
    np.save(str(OUT_DIR / 'z_source.npy'), z_sources)
    np.save(str(OUT_DIR / 'masses.npy'), masses)

    print(f'\nSaved to {OUT_DIR}/')
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == '.npy':
            print(f'  {f.name:<30} {f.stat().st_size/1e6:.1f} MB')

    # ── Preview ────────────────────────────────────────────────────────
    order_nl = np.where(labels == 0)[0]
    order_l = np.where(labels == 1)[0]
    n_show = min(5, len(order_nl), len(order_l))

    # 8 rows: 4 bands + RGB lensed + RGB arc-only + lens-subtracted + arc-only
    fig, axs = plt.subplots(8, n_show, figsize=(4*n_show, 33), dpi=100)

    def show(ax, img, title, cmap='gray', stretch='asinh', lw_frac=0.001):
        d = np.nan_to_num(np.clip(img, 0, None), nan=0.0)
        vmin = float(np.percentile(d, 0.5))
        vmax = float(np.percentile(d, 99.9))
        if vmax <= vmin:
            vmax = vmin + 1.0
        if stretch == 'asinh':
            lw = max((vmax - vmin) * lw_frac, 1e-6)
            norm = AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)
            im = ax.imshow(d, norm=norm, origin='lower', cmap=cmap)
        else:
            im = ax.imshow(d, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=7)
        ax.axis('off')

    def make_rgb(r, g, b, stretch_q=10, stretch_min=0):
        """Make color-preserving asinh-stretched RGB from 3 arrays.

        Applies asinh stretch to luminance, then redistributes to R/G/B
        so relative color ratios are preserved. Ellipticals appear red,
        star-forming arcs appear blue.
        """
        from scipy.ndimage import zoom
        target_shape = r.shape
        # Upsample LW -> SW resolution if needed
        if g.shape != target_shape:
            g = zoom(g, np.array(target_shape) / np.array(g.shape), order=1)
        if b.shape != target_shape:
            b = zoom(b, np.array(target_shape) / np.array(b.shape), order=1)
        r = np.nan_to_num(np.clip(r, 0, None), nan=0.0)
        g = np.nan_to_num(np.clip(g, 0, None), nan=0.0)
        b = np.nan_to_num(np.clip(b, 0, None), nan=0.0)

        # Luminance = sum of channels
        lum = r + g + b
        lum_max = np.max(lum)
        if lum_max <= 0:
            return np.zeros((*target_shape, 3))

        # Asinh stretch the luminance
        lum_stretched = np.arcsinh(lum * stretch_q)
        lum_stretched /= np.max(lum_stretched)

        # Redistribute stretched luminance back to channels
        # preserving the original R:G:B ratio at each pixel
        rgb = np.stack([r, g, b], axis=-1)
        lum_safe = np.where(lum > 0, lum, 1.0)  # avoid division by zero
        for ch in range(3):
            rgb[:, :, ch] = rgb[:, :, ch] / lum_safe * lum_stretched

        return np.clip(rgb, 0, 1)

    def show_rgb(ax, rgb, title):
        ax.imshow(rgb, origin='lower')
        ax.set_title(title, fontsize=7)
        ax.axis('off')

    # Rows 0-3: each band, lensed images only (asinh)
    for row, band in enumerate(BANDS):
        for col in range(n_show):
            idx = order_l[col]
            tE = theta_Es[idx]
            show(axs[row, col], all_images[band][idx],
                 f'{band} lensed θE={tE:.2f}" zl={z_lenses[idx]:.2f}')
        axs[row, 0].set_ylabel(f'{band}\n(asinh)', fontsize=10)

    # Row 4: RGB composite (R=F277W, G=F150W, B=F115W) — lensed full images
    # Normalize by mjysr_to_sim to convert to common surface brightness units (MJy/sr)
    # This removes the ~4x pixel solid angle bias of LW vs SW channels
    for col in range(n_show):
        idx = order_l[col]
        rgb = make_rgb(
            all_images['F277W'][idx] / BAND_CONFIG['F277W']['mjysr_to_sim'],
            all_images['F150W'][idx] / BAND_CONFIG['F150W']['mjysr_to_sim'],
            all_images['F115W'][idx] / BAND_CONFIG['F115W']['mjysr_to_sim'],
        )
        show_rgb(axs[4, col], rgb,
                 f'RGB lensed θE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f}')
    axs[4, 0].set_ylabel('RGB\n(F277W/F150W/F115W)', fontsize=10)

    # Row 5: RGB composite — arc-only (no lens light, no background)
    for col in range(n_show):
        idx = order_l[col]
        rgb = make_rgb(
            all_sources['F277W'][idx] / BAND_CONFIG['F277W']['mjysr_to_sim'],
            all_sources['F150W'][idx] / BAND_CONFIG['F150W']['mjysr_to_sim'],
            all_sources['F115W'][idx] / BAND_CONFIG['F115W']['mjysr_to_sim'],
            stretch_q=50,
        )
        show_rgb(axs[5, col], rgb,
                 f'RGB arcs θE={theta_Es[idx]:.2f}"')
    axs[5, 0].set_ylabel('RGB arcs\n(F277W/F150W/F115W)', fontsize=10)

    # Row 6: lens-subtracted F115W (asinh)
    for col in range(n_show):
        idx = order_l[col]
        show(axs[6, col],
             all_sources['F115W'][idx] + np.median(all_images['F115W'][idx]),
             f'F115W lens-sub θE={theta_Es[idx]:.2f}"',
             stretch='asinh', lw_frac=0.01)
    axs[6, 0].set_ylabel('F115W\nlens-sub', fontsize=10)

    # Row 7: arc-only F115W (inferno)
    for col in range(n_show):
        idx = order_l[col]
        show(axs[7, col], all_sources['F115W'][idx],
             f'arcs θE={theta_Es[idx]:.2f}"', cmap='inferno', stretch='linear')
    axs[7, 0].set_ylabel('F115W\narc-only', fontsize=10)

    plt.suptitle(f'COSMOS-Web multi-band simulation ({len(BANDS)} bands, {N} images)', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'preview.png'), dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {OUT_DIR}/preview.png')


if __name__ == '__main__':
    main()
