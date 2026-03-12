"""
simulate_v3.py — COWLS-calibrated multi-band JWST gravitational lens simulation

v3 changes from v2 (simulate_multiband.py):
  Parameter adjustments based on COWLS (Nightingale et al. 2025, MNRAS 543, 203-222),
  "The COSMOS-Web Lens Survey", which found >100 strong lenses in COSMOS-Web data
  using the same 4 NIRCam bands (F115W, F150W, F277W, F444W).

  Key COWLS findings that drove parameter changes:
    - ~50% of lens galaxies are above z=1, some beyond z=2
    - Source galaxies span z~0.1 to z~9 (into epoch of reionization)
    - Einstein radii mostly below 1.0" (smaller than SLACS)
    - Lower stellar masses than SLACS (most log M* < 11)
    - Lens magnitudes (F444W): 16-22 AB (fainter than SLACS 15-17)
    - SIE + external shear mass model (same as our pipeline)

  Specific parameter changes (v2 -> v3):
    z_lens:    TruncNorm(0.3, 0.15, [0.05, 0.90]) -> TruncNorm(0.7, 0.4, [0.05, 2.5])
    z_source:  TruncNorm(1.5, 0.8, [0.6, 3.0])    -> TruncNorm(2.5, 1.5, [0.5, 7.0])
    sigma_v:   TruncNorm(215, 50, [100, 400])       -> TruncNorm(180, 50, [80, 350])
    theta_E:   filter [0.5, 1.5]"                    -> filter [0.2, 1.5]"
    log_mass:  U(11.5, 13.0)                         -> U(11.0, 13.0)
    SED:       Linear z-scaling                      -> Lyman-break aware for high-z sources

  Unchanged from v2:
    - Per-band empirical PSF from COSMOS-Web stars
    - Per-band real background patches from COSMOS-Web
    - Poisson shot noise calibrated per band
    - SIE + external shear lens model
    - Color-preserving RGB rendering (R=F444W, G=F277W, B=avg(F115W,F150W))
      Following COSMOS-Web convention (Casey, Franco et al.)
    - UV slope diversity: uv_slope ~ N(-0.1, 0.7) for source color variety
    - Common 30mas pixel grid (all bands 125x125) from DR0.5 mosaics

Usage:
    .venv/bin/python3 simulate_v3.py          # 10 test images
    .venv/bin/python3 simulate_v3.py --n 1000 # full dataset
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

# ── Version ───────────────────────────────────────────────────────────────

VERSION = 'v3'
REFERENCE = 'Nightingale+2025 (COWLS, MNRAS 543, 203)'

# ── Paths ─────────────────────────────────────────────────────────────────

PREPPED_DIR = Path('prepped_mosaic')
OUT_DIR     = Path('output/v3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
# DR0.5 mosaics: all bands resampled to common 30mas grid (no SW/LW distinction)

# ── Load per-band calibration info ───────────────────────────────────────

with open(PREPPED_DIR / 'band_info.json') as f:
    BAND_INFO = json.load(f)

# ── Per-band constants ───────────────────────────────────────────────────

sum_to_flux = 6.501853565914121   # nJy -> sim unit conversion

BAND_CONFIG = {}
for band in BANDS:
    info = BAND_INFO[band]
    pixels = 125       # all bands at 30mas in DR0.5 mosaics
    pixel_scale = 0.03 # common pixel scale

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

print(f"simulate_{VERSION} — params calibrated to {REFERENCE}")
print("Band configurations:")
for band in BANDS:
    cfg = BAND_CONFIG[band]
    print(f"  {band}: {cfg['pixels']}x{cfg['pixels']} @ {cfg['pixel_scale']}\"/pix  "
          f"mjysr_to_sim={cfg['mjysr_to_sim']:.2f}  sim_to_elec={cfg['sim_to_elec']:.2f}")

# ── Load backgrounds and PSFs ────────────────────────────────────────────

backgrounds = {}
psf_kernels = {}

for band in BANDS:
    cfg = BAND_CONFIG[band]
    bg_raw = np.load(str(PREPPED_DIR / band / 'backgrounds.npy'))
    bg_raw = np.nan_to_num(bg_raw, nan=0.0)
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


# ── SED color ratios (v3: Lyman-break aware for high-z sources) ──────────
#
# v3 changes:
#   - Elliptical SED: extended to z~2.5 lens redshifts. At higher z_lens the
#     4000A break moves through the NIRCam bands, making F115W/F150W
#     progressively fainter relative to LW bands.
#   - Star-forming SED: added Lyman-break dropout handling for z > 4.
#     UV slope diversity for realistic arc color range.

def elliptical_color_ratios(z_lens):
    """NIRCam flux ratios for an elliptical at z_lens, relative to F115W.

    Extended from v2 to handle z_lens up to ~2.5.
    At z > 1, the 4000A break moves through F150W and into F277W,
    making the SED even redder in the NIRCam bands.
    """
    f150w = 1.3 + 0.4 * z_lens
    f277w = 1.8 + 1.2 * z_lens
    f444w = 2.0 + 1.8 * z_lens
    return {'F115W': 1.0, 'F150W': f150w, 'F277W': f277w, 'F444W': f444w}


def starforming_color_ratios(z_source, uv_slope=0.5):
    """NIRCam flux ratios for a star-forming galaxy at z_source.

    v3: Lyman-break aware + UV slope diversity.

    uv_slope controls the f_nu spectral index: ratio = (1.15/lambda)^uv_slope
      uv_slope > 0: blue (dust-free young starburst)
      uv_slope ~ 0: flat in f_nu
      uv_slope < 0: red (dusty star-forming, or older stellar population)
    """
    # Lyman break wavelength at this redshift (in microns)
    ly_break_um = 0.1216 * (1 + z_source)

    # Band effective wavelengths (microns)
    band_waves = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}

    ratios = {}
    for band, lam_eff in band_waves.items():
        if lam_eff < ly_break_um:
            # Below Lyman break — strong IGM absorption
            suppression = max(0.0, (lam_eff / ly_break_um) ** 3)
            ratio = suppression * 0.05
        else:
            # Above Lyman break — UV continuum with slope diversity
            ratio = (1.15 / lam_eff) ** uv_slope
        ratios[band] = max(ratio, 0.01)

    # Normalize to F115W if it's above the break
    f115w_val = ratios['F115W']
    if f115w_val > 0.01:
        for band in ratios:
            ratios[band] /= f115w_val
    else:
        max_val = max(ratios.values())
        if max_val > 0.01:
            for band in ratios:
                ratios[band] /= max_val

    return ratios


# ── Poisson noise ────────────────────────────────────────────────────────

def add_poisson_noise(image_sim, band, rng=None):
    """Add Poisson shot noise to noiseless sim image."""
    if rng is None:
        rng = np.random.default_rng()
    s2e = BAND_CONFIG[band]['sim_to_elec']
    electrons = np.clip(image_sim * s2e, 0, None)
    noisy = rng.poisson(electrons).astype(np.float64)
    return (noisy / s2e).astype(np.float32)


# ── Mass / stellar-mass helpers ──────────────────────────────────────────

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


# ── Make lenstronomy objects per band ────────────────────────────────────

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

# ── v3 parameter distributions (COWLS-calibrated) ────────────────────────
#
# All changes annotated with v2 -> v3 values and justification.
#
# COWLS key findings (Nightingale+2025):
#   - Lens redshifts: ~50% above z=1, photometric z up to ~4
#   - Source redshifts: z~0.1 to z~9
#   - Einstein radii: mostly below 1.0" (cf. SLACS > 0.8")
#   - Stellar masses: mostly log M* < 11 (cf. SLACS > 11)
#   - Lens magnitudes: F444W 16-22 AB (cf. SLACS 15-17)

# v2: TruncNorm(loc=0.3, scale=0.15, [0.05, 0.90])  — SLACS
# v3: TruncNorm(loc=0.7, scale=0.4,  [0.05, 2.5])   — COWLS
ZLENS_LOC, ZLENS_SCALE = 0.7, 0.4
ZLENS_LO, ZLENS_HI = 0.05, 2.5

# v2: TruncNorm(loc=1.5, scale=0.8, [0.6, 3.0])     — SLACS
# v3: TruncNorm(loc=2.5, scale=1.5, [0.5, 7.0])     — COWLS (sources to z~9)
ZSRC_LOC, ZSRC_SCALE = 2.5, 1.5
ZSRC_LO, ZSRC_HI = 0.5, 7.0

# v2: TruncNorm(loc=215, scale=50, [100, 400]) km/s  — SLACS
# v3: TruncNorm(loc=180, scale=50, [80, 350]) km/s   — COWLS (lower-mass, smaller theta_E)
SIGMA_LOC, SIGMA_SCALE = 180, 50
SIGMA_LO, SIGMA_HI = 80, 350

# v2: theta_E must be in [0.5, 1.5]"                  — SLACS
# v3: theta_E must be in [0.2, 1.5]"                  — COWLS (most below 1.0")
THETA_E_LO, THETA_E_HI = 0.2, 1.5

# v2: log_mass ~ U(11.5, 13.0)                        — SLACS
# v3: log_mass ~ U(11.0, 13.0)                        — COWLS (lower-mass lenses)
LOGMASS_LO, LOGMASS_HI = 11.0, 13.0


# ── Multi-band simulation ───────────────────────────────────────────────

def simulate_one_multiband(lensed=True, seed=None):
    """Simulate one lens system in all 4 bands.

    Returns dict with keys per band, each containing (image, image_source).
    Also returns shared params: theta_E, z_lens, z_source, mass, mStar.
    """
    rng = np.random.default_rng(seed)

    # ── COWLS-calibrated redshifts ────────────────────────────────────
    z_lens = float(truncnorm.rvs(
        a=(ZLENS_LO - ZLENS_LOC) / ZLENS_SCALE,
        b=(ZLENS_HI - ZLENS_LOC) / ZLENS_SCALE,
        loc=ZLENS_LOC, scale=ZLENS_SCALE,
        random_state=int(rng.integers(int(1e9)))))

    z_src_min = max(ZSRC_LO, z_lens + 0.1)
    for _ in range(100):
        z_source = float(truncnorm.rvs(
            a=(ZSRC_LO - ZSRC_LOC) / ZSRC_SCALE,
            b=(ZSRC_HI - ZSRC_LOC) / ZSRC_SCALE,
            loc=ZSRC_LOC, scale=ZSRC_SCALE,
            random_state=int(rng.integers(int(1e9)))))
        if z_source > z_lens + 0.1:
            break
    z_source = max(z_source, z_src_min)

    # ── Velocity dispersion -> Einstein radius ───────────────────────
    sigma_v = float(truncnorm.rvs(
        a=(SIGMA_LO - SIGMA_LOC) / SIGMA_SCALE,
        b=(SIGMA_HI - SIGMA_LOC) / SIGMA_SCALE,
        loc=SIGMA_LOC, scale=SIGMA_SCALE,
        random_state=int(rng.integers(int(1e9)))))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    if lensed and not (THETA_E_LO <= theta_E <= THETA_E_HI):
        return simulate_one_multiband(lensed=lensed, seed=int(rng.integers(int(1e9))))

    # ── Halo mass (galaxy-scale) ─────────────────────────────────────
    log_mass = float(rng.uniform(LOGMASS_LO, LOGMASS_HI))
    mass = 10**log_mass
    mStar = stellar_mass(mass, z_lens)

    # ── Galaxy shape parameters (shared across bands) ────────────────
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

    # ── Source calibration (done once in F115W, then scaled) ─────────
    scale_up = 10**float(rng.uniform(0, 2))
    src_flux_njy = float(truncnorm.rvs(0, 3, loc=50, scale=80,
                         random_state=int(rng.integers(int(1e9)))))
    calc_sum_src_f115w = sum_to_flux * src_flux_njy
    lStar = mStar / ML_ratio(max(z_lens, 0.01))
    calc_sum_lens_f115w = sum_to_flux * lStar

    # SED color ratios — randomize UV slope for source diversity
    uv_slope = float(np.clip(rng.normal(-0.1, 0.7), -1.5, 1.5))
    lens_colors = elliptical_color_ratios(z_lens)
    src_colors = starforming_color_ratios(z_source, uv_slope=uv_slope)

    # ── Render in each band ──────────────────────────────────────────
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

    return band_results, theta_E, z_lens, z_source, mass, mStar, uv_slope


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='Number of images (half lensed, half not)')
    args = parser.parse_args()

    N = args.n
    N_EACH = N // 2

    print(f"\nGenerating {N} images ({N_EACH} lensed + {N_EACH} non-lensed) in {len(BANDS)} bands...")
    print(f"  z_lens ~ TruncNorm({ZLENS_LOC}, {ZLENS_SCALE}, [{ZLENS_LO}, {ZLENS_HI}])")
    print(f"  z_src  ~ TruncNorm({ZSRC_LOC}, {ZSRC_SCALE}, [{ZSRC_LO}, {ZSRC_HI}])")
    print(f"  sigma  ~ TruncNorm({SIGMA_LOC}, {SIGMA_SCALE}, [{SIGMA_LO}, {SIGMA_HI}]) km/s")
    print(f"  theta_E filter: [{THETA_E_LO}, {THETA_E_HI}] arcsec")
    print(f"  log_mass ~ U({LOGMASS_LO}, {LOGMASS_HI})")

    # Storage
    all_images = {band: np.zeros((N, BAND_CONFIG[band]['pixels'], BAND_CONFIG[band]['pixels']),
                                 dtype=np.float32) for band in BANDS}
    all_sources = {band: np.zeros_like(all_images[band]) for band in BANDS}
    all_clean = {band: np.zeros_like(all_images[band]) for band in BANDS}  # sim only, no background
    labels = np.zeros(N)
    theta_Es = np.zeros(N)
    z_lenses = np.zeros(N)
    z_sources = np.zeros(N)
    masses = np.zeros(N)
    sigma_vs = np.zeros(N)  # v3: also save velocity dispersions

    rng_main = np.random.default_rng(99)
    jobs = ([(i, False) for i in range(N_EACH)] +
            [(i + N_EACH, True) for i in range(N_EACH)])

    t0 = time.time()
    for idx, lensed in jobs:
        label = 'lensed' if lensed else 'non-lensed'
        print(f'  [{idx+1}/{N}] {label}...', end=' ', flush=True)
        t1 = time.time()

        result = simulate_one_multiband(lensed=lensed, seed=int(rng_main.integers(int(1e9))))
        band_results, theta_E, z_lens, z_source, mass, mStar, uv_slope = result

        rng_noise = np.random.default_rng(int(rng_main.integers(int(1e9))))

        for band in BANDS:
            img = band_results[band]['image']
            img_noisy = add_poisson_noise(img, band, rng=rng_noise)
            bg = get_background(band)
            all_images[band][idx] = img_noisy + bg
            all_sources[band][idx] = band_results[band]['image_source']
            all_clean[band][idx] = img_noisy  # simulation + noise, no background

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

        # Print SED color ratios for first lensed image
        sed_info = ''
        if lensed and idx == N_EACH:
            src_colors = starforming_color_ratios(z_source, uv_slope=uv_slope)
            sed_info = f'\n    src SED (β={uv_slope:.2f}): ' + ' '.join(f'{b}={src_colors[b]:.2f}' for b in BANDS)

        print(f'{time.time()-t1:.1f}s  θE={theta_E:.2f}" zl={z_lens:.2f} zs={z_source:.2f}{arc_ratio}{sed_info}')

    total = time.time() - t0
    print(f'\nDone — {N} images in {total:.1f}s ({total/N:.2f}s/image)')

    # Print parameter summary
    lensed_mask = labels == 1
    print(f'\nParameter summary (lensed only):')
    print(f'  z_lens:   {z_lenses[lensed_mask].min():.2f} - {z_lenses[lensed_mask].max():.2f}  (mean {z_lenses[lensed_mask].mean():.2f})')
    print(f'  z_source: {z_sources[lensed_mask].min():.2f} - {z_sources[lensed_mask].max():.2f}  (mean {z_sources[lensed_mask].mean():.2f})')
    print(f'  theta_E:  {theta_Es[lensed_mask].min():.2f} - {theta_Es[lensed_mask].max():.2f}  (mean {theta_Es[lensed_mask].mean():.2f})')

    # ── Save ─────────────────────────────────────────────────────────
    for band in BANDS:
        np.save(str(OUT_DIR / f'images_{band}.npy'), all_images[band])
        np.save(str(OUT_DIR / f'sources_{band}.npy'), all_sources[band])
    np.save(str(OUT_DIR / 'lensed.npy'), labels)
    np.save(str(OUT_DIR / 'theta_Es.npy'), theta_Es)
    np.save(str(OUT_DIR / 'z_lens.npy'), z_lenses)
    np.save(str(OUT_DIR / 'z_source.npy'), z_sources)
    np.save(str(OUT_DIR / 'masses.npy'), masses)

    # Save version metadata
    meta = {
        'version': VERSION,
        'reference': REFERENCE,
        'n_images': N,
        'params': {
            'z_lens': f'TruncNorm({ZLENS_LOC}, {ZLENS_SCALE}, [{ZLENS_LO}, {ZLENS_HI}])',
            'z_source': f'TruncNorm({ZSRC_LOC}, {ZSRC_SCALE}, [{ZSRC_LO}, {ZSRC_HI}])',
            'sigma_v': f'TruncNorm({SIGMA_LOC}, {SIGMA_SCALE}, [{SIGMA_LO}, {SIGMA_HI}]) km/s',
            'theta_E_filter': f'[{THETA_E_LO}, {THETA_E_HI}] arcsec',
            'log_mass': f'U({LOGMASS_LO}, {LOGMASS_HI})',
            'arc_lens_min_ratio': ARC_LENS_MIN_RATIO,
            'sed_source': 'Lyman-break aware star-forming',
            'sed_lens': 'Elliptical (extended z range)',
        },
        'bands': BANDS,
    }
    with open(OUT_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to {OUT_DIR}/')
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in ('.npy', '.json'):
            print(f'  {f.name:<30} {f.stat().st_size/1e6:.1f} MB')

    # ── Preview ──────────────────────────────────────────────────────
    order_nl = np.where(labels == 0)[0]
    order_l = np.where(labels == 1)[0]
    n_show = min(5, len(order_nl), len(order_l))

    # 11 rows: 4 bands + RGB clean sim + RGB lensed + RGB arcs + RGB lens-only + RGB arcs boosted + lens-sub + arc-only
    fig, axs = plt.subplots(11, n_show, figsize=(4*n_show, 46), dpi=100)

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

    def make_rgb(r, g, b, stretch=0.2, Q=10, sat_boost=1.5, percentile_bg=1.0):
        """Lupton et al. (2004) RGB composite — standard astronomical method.

        stretch: linear region size (lower = more contrast in faint features)
        Q: asinh softening (higher = more dynamic range compression)
        sat_boost: color saturation multiplier (>1 = more vivid)
        percentile_bg: percentile to subtract as background (black level)
        """
        from astropy.visualization import make_lupton_rgb
        r = np.nan_to_num(r, nan=0.0).astype(np.float64)
        g = np.nan_to_num(g, nan=0.0).astype(np.float64)
        b = np.nan_to_num(b, nan=0.0).astype(np.float64)
        # Subtract background so sky is black
        for ch in [r, g, b]:
            ch -= np.percentile(ch, percentile_bg)
            ch[ch < 0] = 0
        rgb = make_lupton_rgb(r, g, b, stretch=stretch, Q=Q, minimum=0)
        # Boost color saturation
        if sat_boost != 1.0:
            rgb_f = rgb.astype(np.float32) / 255.0
            lum = 0.2989 * rgb_f[:,:,0] + 0.5870 * rgb_f[:,:,1] + 0.1140 * rgb_f[:,:,2]
            for ch in range(3):
                rgb_f[:,:,ch] = lum + sat_boost * (rgb_f[:,:,ch] - lum)
            rgb = (np.clip(rgb_f, 0, 1) * 255).astype(np.uint8)
        return rgb

    def show_rgb(ax, rgb_img, title):
        ax.imshow(rgb_img, origin='lower')
        ax.set_title(title, fontsize=7)
        ax.axis('off')

    # Pixel solid angle normalization factors
    _norm = {band: BAND_CONFIG[band]['mjysr_to_sim'] for band in BANDS}

    # Rows 0-3: each band (asinh)
    for row, band in enumerate(BANDS):
        for col in range(n_show):
            idx = order_l[col]
            tE = theta_Es[idx]
            show(axs[row, col], all_images[band][idx],
                 f'{band} θE={tE:.2f}" zl={z_lenses[idx]:.2f} zs={z_sources[idx]:.2f}')
        axs[row, 0].set_ylabel(f'{band}\n(asinh)', fontsize=10)

    # RGB channel mapping: R=F444W, G=F277W, B=(F115W+F150W)/2
    # Standard COSMOS-Web convention (Casey, Franco et al.)
    def _rgb_channels(img_dict, norm_dict):
        """Return (r, g, b) from 4-band image dict using COSMOS-Web convention."""
        r = img_dict['F444W'] / norm_dict['F444W']
        g = img_dict['F277W'] / norm_dict['F277W']
        b = 0.5 * (img_dict['F115W'] / norm_dict['F115W'] +
                    img_dict['F150W'] / norm_dict['F150W'])
        return r, g, b

    # Row 4: RGB clean simulation (lens + arcs + noise, NO background)
    for col in range(n_show):
        idx = order_l[col]
        clean_imgs = {band: all_clean[band][idx] for band in BANDS}
        r, g, b = _rgb_channels(clean_imgs, _norm)
        rgb = make_rgb(r, g, b, stretch=0.1, Q=10, sat_boost=1.5, percentile_bg=0)
        show_rgb(axs[4, col], rgb,
                 f'Clean sim θE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f}')
    axs[4, 0].set_ylabel('RGB clean\n(no background)', fontsize=10)

    # Row 5: RGB lensed (full image with background)
    for col in range(n_show):
        idx = order_l[col]
        imgs = {band: all_images[band][idx] for band in BANDS}
        r, g, b = _rgb_channels(imgs, _norm)
        rgb = make_rgb(r, g, b, stretch=0.15, Q=12, sat_boost=1.5)
        show_rgb(axs[5, col], rgb,
                 f'RGB lensed θE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f}')
    axs[5, 0].set_ylabel('RGB lensed\n(+ background)', fontsize=10)

    # Row 6: RGB arcs only (no lens, no background)
    for col in range(n_show):
        idx = order_l[col]
        srcs = {band: all_sources[band][idx] for band in BANDS}
        r, g, b = _rgb_channels(srcs, _norm)
        rgb = make_rgb(r, g, b, stretch=0.05, Q=8, sat_boost=2.0, percentile_bg=0)
        show_rgb(axs[6, col], rgb,
                 f'RGB arcs θE={theta_Es[idx]:.2f}" zs={z_sources[idx]:.2f}')
    axs[6, 0].set_ylabel('RGB arcs\n(source only)', fontsize=10)

    # Row 7: RGB lens-only (full image minus arcs, includes background)
    for col in range(n_show):
        idx = order_l[col]
        lens_imgs = {band: np.clip(all_images[band][idx] - all_sources[band][idx], 0, None)
                     for band in BANDS}
        r, g, b = _rgb_channels(lens_imgs, _norm)
        rgb = make_rgb(r, g, b, stretch=0.15, Q=12, sat_boost=1.5)
        show_rgb(axs[7, col], rgb,
                 f'RGB lens-only zl={z_lenses[idx]:.2f}')
    axs[7, 0].set_ylabel('RGB lens\n(no arcs)', fontsize=10)

    # Row 8: RGB arcs boosted (arc peak matched to lens peak)
    for col in range(n_show):
        idx = order_l[col]
        srcs = {band: all_sources[band][idx] for band in BANDS}
        arc_r, arc_g, arc_b = _rgb_channels(srcs, _norm)
        arc_peak = max(np.max(np.abs(arc_r)), np.max(np.abs(arc_g)), np.max(np.abs(arc_b)), 1e-10)
        lens_imgs = {band: np.clip(all_images[band][idx] - all_sources[band][idx], 0, None)
                     for band in BANDS}
        lens_r, lens_g, lens_b = _rgb_channels(lens_imgs, _norm)
        lens_peak = max(np.max(np.abs(lens_r)), np.max(np.abs(lens_g)), np.max(np.abs(lens_b)), 1e-10)
        boost = lens_peak / arc_peak
        rgb = make_rgb(arc_r * boost, arc_g * boost, arc_b * boost, stretch=0.05, Q=8, sat_boost=2.0, percentile_bg=0)
        show_rgb(axs[8, col], rgb,
                 f'RGB arcs boosted {boost:.0f}x')
    axs[8, 0].set_ylabel('RGB arcs\n(boosted)', fontsize=10)

    # Row 9: lens-subtracted F115W (asinh)
    for col in range(n_show):
        idx = order_l[col]
        show(axs[9, col],
             all_sources['F115W'][idx] + np.median(all_images['F115W'][idx]),
             f'F115W lens-sub θE={theta_Es[idx]:.2f}"',
             stretch='asinh', lw_frac=0.01)
    axs[9, 0].set_ylabel('F115W\nlens-sub', fontsize=10)

    # Row 10: arc-only F115W (inferno)
    for col in range(n_show):
        idx = order_l[col]
        show(axs[10, col], all_sources['F115W'][idx],
             f'arcs θE={theta_Es[idx]:.2f}"', cmap='inferno', stretch='linear')
    axs[10, 0].set_ylabel('F115W\narc-only', fontsize=10)

    plt.suptitle(f'COSMOS-Web {VERSION} simulation — COWLS-calibrated ({len(BANDS)} bands, {N} images)', fontsize=12)
    plt.tight_layout()
    # Auto-increment preview filename: preview_1.png, preview_2.png, ...
    existing = sorted(OUT_DIR.glob('preview_*.png'))
    next_n = 1
    if existing:
        nums = [int(p.stem.split('_')[1]) for p in existing if p.stem.split('_')[1].isdigit()]
        if nums:
            next_n = max(nums) + 1
    preview_name = f'preview_{next_n}.png'
    plt.savefig(str(OUT_DIR / preview_name), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {OUT_DIR}/{preview_name}')


if __name__ == '__main__':
    main()
