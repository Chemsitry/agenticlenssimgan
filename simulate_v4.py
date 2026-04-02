"""
simulate_v4.py — Validated multi-band JWST gravitational lens simulation

Based on validate_f115w.py (validated against real COWLS II data):
  - 630x630 px at 0.03"/pix (18.9" FoV, matches COWLS II Figure 1)
  - 4 NIRCam bands: F115W, F150W, F277W, F444W
  - Sersic profiles for lens + source (clean, no artifacts)
  - Per-band empirical PSF from DR0.5 mosaics
  - Per-band real backgrounds (spatially matched across bands)
  - Poisson noise calibrated per band
  - COWLS-calibrated parameter distributions (Nightingale+2025)
  - Percentile+gamma RGB rendering tuned to COWLS II style

Usage:
    .venv/bin/python3 simulate_v4.py              # 10 test images
    .venv/bin/python3 simulate_v4.py --n 2000     # full dataset
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import multiprocessing as mp

# ── Configuration ────────────────────────────────────────────────────────

VERSION = 'v4'
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
IMAGE_SIZE = 630
PIXEL_SCALE = 0.03  # arcsec/pix (630 * 0.03 = 18.9" FoV)
PREPPED_DIR = Path('prepped_mosaic_630')
OUT_DIR = Path('output/v4')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sum_to_flux = 6.501853565914121
gain = 2.05

# ── COWLS-calibrated parameter distributions ─────────────────────────────

ZLENS_LOC, ZLENS_SCALE = 0.7, 0.4
ZLENS_LO, ZLENS_HI = 0.05, 2.5

ZSRC_LOC, ZSRC_SCALE = 2.5, 1.5
ZSRC_LO, ZSRC_HI = 0.5, 7.0

SIGMA_LOC, SIGMA_SCALE = 180, 50
SIGMA_LO, SIGMA_HI = 80, 350

THETA_E_LO, THETA_E_HI = 0.5, 1.5
LOGMASS_LO, LOGMASS_HI = 11.0, 13.0
ARC_LENS_MIN_RATIO = 0.10

# ── Load per-band calibration ────────────────────────────────────────────

with open(PREPPED_DIR / 'band_info.json') as f:
    all_band_info = json.load(f)

BAND_CONFIG = {}
psf_kernels = {}
backgrounds = {}

print("Loading per-band data:")
for band in BANDS:
    info = all_band_info[band]
    pixar_sr = info['pixar_sr']
    photmjsr = info['photmjsr']
    xposure = info['xposure']

    mjysr_to_sim = pixar_sr * 1e15 * sum_to_flux
    sim_to_elec = (1.0 / mjysr_to_sim) / photmjsr * xposure * gain

    BAND_CONFIG[band] = {
        'mjysr_to_sim': mjysr_to_sim,
        'sim_to_elec': sim_to_elec,
    }

    psf = np.load(str(PREPPED_DIR / band / 'psf_median.npy')).astype(np.float64)
    psf = np.clip(psf, 0, None)
    psf /= psf.sum()
    psf_kernels[band] = psf

    bg_raw = np.load(str(PREPPED_DIR / band / 'backgrounds.npy'))
    bg_raw = np.nan_to_num(bg_raw, nan=0.0)
    backgrounds[band] = (bg_raw * mjysr_to_sim).astype(np.float32)

    print(f"  {band}: sim_to_elec={sim_to_elec:.2f}  backgrounds={backgrounds[band].shape}")


# ── Helper functions ─────────────────────────────────────────────────────

def elliptical_color_ratios(z_lens):
    # Recalibrated against COWLS II PyAutoLens photometry (15 lenses)
    # Linear fits to real F_band/F_F115W vs z_lens
    f150w = 1.68 + 0.09 * z_lens
    f277w = 8.79 + 8.40 * z_lens
    f444w = 0.42 + 18.11 * z_lens
    return {'F115W': 1.0, 'F150W': f150w, 'F277W': f277w, 'F444W': f444w}


def starforming_color_ratios(z_source, uv_slope=0.5):
    # Recalibrated against COWLS II PyAutoLens source photometry (15 sources)
    # Real sources are much redder than a simple UV power law — rest-frame
    # optical/Balmer break lands in F277W/F444W at z~1-3, plus dust reddening.
    ly_break_um = 0.1216 * (1 + z_source)
    band_waves = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}
    ratios = {}
    for band, lam_eff in band_waves.items():
        if lam_eff < ly_break_um:
            suppression = max(0.0, (lam_eff / ly_break_um) ** 3)
            ratio = suppression * 0.05
        else:
            # Rest-frame wavelength
            lam_rest = lam_eff / (1 + z_source)
            # Balmer/4000A break boost: rest-frame optical is brighter
            if lam_rest > 0.35:
                break_boost = 1.0 + 3.0 * min((lam_rest - 0.35) / 0.3, 1.0)
            else:
                break_boost = 1.0
            # UV slope (blueward of break)
            uv_part = (1.15 / lam_eff) ** uv_slope
            ratio = uv_part * break_boost
        ratios[band] = max(ratio, 0.01)
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


def add_poisson_noise(image_sim, band, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    s2e = BAND_CONFIG[band]['sim_to_elec']
    electrons = np.nan_to_num(image_sim * s2e, nan=0.0, posinf=0.0, neginf=0.0)
    electrons = np.clip(electrons, 0, None)
    large = electrons > 1e8
    if large.any():
        noisy = np.empty_like(electrons)
        noisy[~large] = rng.poisson(electrons[~large]).astype(np.float64)
        noisy[large] = rng.normal(electrons[large], np.sqrt(electrons[large]))
        noisy = np.clip(noisy, 0, None)
    else:
        noisy = rng.poisson(electrons).astype(np.float64)
    return (noisy / s2e).astype(np.float32)


def stellar_mass(M, z):
    mM00 = 0.0282; nu = -0.72
    gamma0 = 0.556; gamma1 = -0.26; beta0 = 1.06; beta1 = 0.17
    M1 = 10**(11.88 * (z + 1)**0.019)
    mM0 = mM00 * (z + 1)**nu
    gamma = gamma0 * (z + 1)**gamma1
    beta = beta1 * z + beta0
    shmr = 2 * mM0 / ((M / M1)**(-beta) + (M / M1)**gamma)
    return shmr * M


def ML_ratio(z):
    return 10**(2.15259223299506 * np.log10(z) + 6.61731435158865)


def make_psf_obj(band):
    return PSF(psf_type='PIXEL',
               kernel_point_source=psf_kernels[band],
               kernel_point_source_normalisation=True)


def make_kwargs_data():
    return {
        'background_rms': 0,
        'exposure_time': 6184.0,
        'ra_at_xy_0': -IMAGE_SIZE / 2 * PIXEL_SCALE,
        'dec_at_xy_0': -IMAGE_SIZE / 2 * PIXEL_SCALE,
        'transform_pix2angle': np.array([[PIXEL_SCALE, 0.], [0., PIXEL_SCALE]]),
        'image_data': np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    }


KWARGS_NUMERICS = {
    'supersampling_factor': 3,
    'supersampling_convolution': True,
}


# ── Simulate one system ──────────────────────────────────────────────────

def simulate_one(lensed=True, seed=None):
    """Simulate one lens system in all 4 bands. Returns dict."""
    rng = np.random.default_rng(seed)

    # Sample redshifts
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

    # Velocity dispersion -> Einstein radius
    sigma_v = float(truncnorm.rvs(
        a=(SIGMA_LO - SIGMA_LOC) / SIGMA_SCALE,
        b=(SIGMA_HI - SIGMA_LOC) / SIGMA_SCALE,
        loc=SIGMA_LOC, scale=SIGMA_SCALE,
        random_state=int(rng.integers(int(1e9)))))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    if lensed and not (THETA_E_LO <= theta_E <= THETA_E_HI):
        return simulate_one(lensed=lensed, seed=int(rng.integers(int(1e9))))

    # Halo mass
    log_mass = float(rng.uniform(LOGMASS_LO, LOGMASS_HI))
    mass = 10**log_mass
    mStar = stellar_mass(mass, z_lens)

    # Lens shape — fixed Sersic params (validated)
    e1, e2 = 0.05, 0.02
    R_sersic_lens = 0.4
    n_sersic_lens = 4

    # Source shape — fixed Sersic params (validated)
    R_sersic_src = 0.15
    n_sersic_src = 1.5
    e1s, e2s = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    # Source position — always at 0.3 * theta_E (validated, produces visible arcs)
    if lensed and theta_E > 0:
        src_offset = 0.3 * theta_E
        src_angle = float(rng.uniform(0, 2 * np.pi))
        center_x = src_offset * np.cos(src_angle)
        center_y = src_offset * np.sin(src_angle)
    else:
        center_x, center_y = rng.normal(0, 0.25, size=2)

    # External shear — modest fixed values (validated)
    gamma1, gamma2 = 0.02, 0.01

    # Lens mass model
    kwargs_lens = [
        {'theta_E': theta_E, 'e1': float(e1), 'e2': float(e2),
         'center_x': 0., 'center_y': 0.},
        {'gamma1': gamma1, 'gamma2': gamma2}
    ]

    # Light model templates
    kwargs_lens_light = [{
        'amp': 1, 'R_sersic': R_sersic_lens, 'n_sersic': n_sersic_lens,
        'e1': float(e1), 'e2': float(e2), 'center_x': 0., 'center_y': 0.
    }]
    kwargs_source = [{
        'amp': 1, 'R_sersic': R_sersic_src, 'n_sersic': n_sersic_src,
        'e1': float(e1s), 'e2': float(e2s),
        'center_x': float(center_x), 'center_y': float(center_y)
    }]

    # SED color ratios
    uv_slope = float(np.clip(rng.normal(-0.5, 1.0), -2.5, 1.5))
    lens_colors = elliptical_color_ratios(z_lens)
    src_colors = starforming_color_ratios(z_source, uv_slope=uv_slope)

    # Render all bands — calibration matches validate_f115w.py approach
    band_results = {}
    target_ratio = 0.25  # fixed arc/lens ratio (validated)

    for band in BANDS:
        data_class = ImageData(**make_kwargs_data())
        psf_class = make_psf_obj(band)
        lens_model = LensModel(['SIE', 'SHEAR'], z_lens=z_lens, z_source=z_source)
        source_model = LightModel(['SERSIC_ELLIPSE'])
        lens_light_model = LightModel(['SERSIC_ELLIPSE'])

        im = ImageModel(
            data_class=data_class, psf_class=psf_class,
            lens_model_class=lens_model,
            source_model_class=source_model,
            lens_light_model_class=lens_light_model,
            kwargs_numerics=KWARGS_NUMERICS)

        # Calibrate lens amp: render at amp=1, measure peak, scale to match
        # real COWLS data range (~1000-4000 sim units peak, validated)
        kw_ll_cal = [{**kwargs_lens_light[0], 'amp': 1.0}]
        kw_src_cal = [{**kwargs_source[0], 'amp': 1.0}]

        img_lens_unit = im.image(kwargs_lens, kw_src_cal,
                                  kwargs_lens_light=kw_ll_cal,
                                  source_add=False, lens_light_add=True)
        peak_unit = float(np.max(img_lens_unit))
        sum_lens_unit = float(np.sum(img_lens_unit))

        if peak_unit <= 0 or sum_lens_unit <= 0:
            return simulate_one(lensed=lensed, seed=int(rng.integers(int(1e9))))

        # Target peak brightness scaled by SED color ratio
        # Base peak ~2000 sim units in F115W (validated against Lens E)
        target_peak = 2000.0 * lens_colors[band]
        amp_lens = target_peak / peak_unit

        # Calibrate source amp via fixed arc/lens ratio
        img_src_unit = im.image(kwargs_lens, kw_src_cal,
                                 kwargs_lens_light=kw_ll_cal,
                                 source_add=True, lens_light_add=False)
        sum_src_unit = float(np.sum(img_src_unit))

        if sum_src_unit <= 0:
            return simulate_one(lensed=lensed, seed=int(rng.integers(int(1e9))))

        sum_lens = sum_lens_unit * amp_lens
        amp_src = (sum_lens * target_ratio / sum_src_unit)

        if not lensed:
            amp_src = 0.0

        # Final render with calibrated amps
        kw_ll = [{**kwargs_lens_light[0], 'amp': amp_lens}]
        kw_src = [{**kwargs_source[0], 'amp': amp_src}]

        image = im.image(kwargs_lens, kw_src,
                          kwargs_lens_light=kw_ll,
                          source_add=True, lens_light_add=True)
        image_source = im.image(kwargs_lens, kw_src,
                                 kwargs_lens_light=kw_ll,
                                 source_add=True, lens_light_add=False)

        band_results[band] = {
            'image': image.astype(np.float32),
            'image_source': image_source.astype(np.float32),
        }

    return band_results, theta_E, z_lens, z_source, mass


# ── Parallel worker ──────────────────────────────────────────────────────

def _simulate_worker(args):
    idx, lensed, seed, noise_seed = args
    result = simulate_one(lensed=lensed, seed=seed)
    band_results, theta_E, z_lens, z_source, mass = result

    rng_noise = np.random.default_rng(noise_seed)
    bg_idx = int(rng_noise.integers(len(backgrounds[BANDS[0]])))

    out = {'idx': idx, 'label': 1.0 if lensed else 0.0,
           'theta_E': theta_E, 'z_lens': z_lens, 'z_source': z_source, 'mass': mass,
           'images': {}, 'sources': {}}

    for band in BANDS:
        img = band_results[band]['image']
        img_noisy = add_poisson_noise(img, band, rng=rng_noise)
        bg = backgrounds[band][bg_idx]
        out['images'][band] = (img_noisy + bg).astype(np.float32)
        out['sources'][band] = band_results[band]['image_source'].astype(np.float32)

    return out


# ── RGB rendering ────────────────────────────────────────────────────────

def make_rgb(r, g, b, smooth=3.5, gamma=0.30, lum_gate=0.08):
    r = gaussian_filter(np.nan_to_num(r, nan=0.0).astype(np.float64), smooth)
    g = gaussian_filter(np.nan_to_num(g, nan=0.0).astype(np.float64), smooth)
    b = gaussian_filter(np.nan_to_num(b, nan=0.0).astype(np.float64), smooth)
    out = np.zeros((*r.shape, 3))
    for i, ch in enumerate([r, g, b]):
        bg = np.percentile(ch, 30)
        ch = ch - bg
        ch[ch < 0] = 0
        vlo = np.percentile(ch, 1)
        vhi = np.percentile(ch, 99.8)
        if vhi <= vlo:
            vhi = vlo + 1
        out[:, :, i] = np.clip((ch - vlo) / (vhi - vlo), 0, 1)
    out = np.power(out, gamma)
    if lum_gate > 0:
        lum = np.max(out, axis=2)
        gate = np.clip((lum - lum_gate) / lum_gate, 0, 1)
        for i in range(3):
            out[:, :, i] *= gate
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def build_rgb(img_dict):
    norm = {band: BAND_CONFIG[band]['mjysr_to_sim'] for band in BANDS}
    r = img_dict['F444W'] / norm['F444W'] * 1.5
    g = 0.5 * (img_dict['F277W'] / norm['F277W'] +
                img_dict['F150W'] / norm['F150W'])
    b = img_dict['F115W'] / norm['F115W'] * 0.5
    return make_rgb(r, g, b)


# ── Preview ──────────────────────────────────────────────────────────────

def make_preview(all_images, all_sources, labels, theta_Es, z_lenses, z_sources, N):
    lensed_idx = np.where(labels == 1)[0]
    n_show = min(5, len(lensed_idx))
    if n_show == 0:
        return

    order = lensed_idx[:n_show]
    fig, axs = plt.subplots(5, n_show, figsize=(4 * n_show, 20), dpi=100)
    if n_show == 1:
        axs = axs[:, np.newaxis]

    for col, idx in enumerate(order):
        for row, band in enumerate(BANDS):
            img = all_images[band][idx]
            d = np.nan_to_num(np.clip(img, 0, None), nan=0.0)
            vmin = float(np.percentile(d, 0.5))
            vmax = float(np.percentile(d, 99.9))
            if vmax <= vmin:
                vmax = vmin + 1.0
            lw = max((vmax - vmin) * 0.001, 1e-6)
            norm = AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)
            axs[row, col].imshow(d, norm=norm, origin='lower', cmap='gray')
            axs[row, col].set_title(
                f'{band} tE={theta_Es[idx]:.2f}" zl={z_lenses[idx]:.2f}', fontsize=8)
            axs[row, col].axis('off')

        # RGB row
        imgs = {band: all_images[band][idx] for band in BANDS}
        rgb = build_rgb(imgs)
        axs[4, col].imshow(rgb, origin='lower')
        axs[4, col].set_title(f'RGB zs={z_sources[idx]:.2f}', fontsize=8)
        axs[4, col].axis('off')

    for row, band in enumerate(BANDS):
        axs[row, 0].set_ylabel(band, fontsize=10)
    axs[4, 0].set_ylabel('RGB', fontsize=10)

    plt.suptitle(f'simulate_v4 — {N} images (630px, 18.9" FoV)', fontsize=12)
    plt.tight_layout()

    # Auto-increment preview filename
    existing = list(OUT_DIR.glob('preview_*.png'))
    num = len(existing) + 1
    path = OUT_DIR / f'preview_{num}.png'
    plt.savefig(str(path), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='v4 multi-band lens simulation (630px, validated)')
    parser.add_argument('--n', type=int, default=10, help='Number of images (half lensed, half not)')
    args = parser.parse_args()

    N = args.n
    N_EACH = N // 2

    print(f"\nsimulate_v4 — validated multi-band pipeline")
    print(f"  {IMAGE_SIZE}x{IMAGE_SIZE} px @ {PIXEL_SCALE}\"/pix = {IMAGE_SIZE*PIXEL_SCALE:.1f}\" FoV")
    print(f"\nGenerating {N} images ({N_EACH} lensed + {N_EACH} non-lensed)...")

    # Storage
    all_images = {}
    all_sources = {}
    for band in BANDS:
        all_images[band] = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        all_sources[band] = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    labels = np.zeros(N)
    theta_Es = np.zeros(N)
    z_lenses = np.zeros(N)
    z_sources = np.zeros(N)
    masses = np.zeros(N)

    # Pre-generate seeds
    rng_main = np.random.default_rng(99)
    jobs = []
    for i in range(N_EACH):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i, False, seed, noise_seed))
    for i in range(N_EACH):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i + N_EACH, True, seed, noise_seed))

    # Parallel generation
    n_workers = mp.cpu_count()
    ctx = mp.get_context('fork')
    chunksize = max(1, min(50, N // (n_workers * 4)))
    print(f"  Using {n_workers} workers (fork), chunksize={chunksize}")

    done = 0
    t0 = time.time()

    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_simulate_worker, jobs, chunksize=chunksize):
            idx = result['idx']
            for band in BANDS:
                all_images[band][idx] = result['images'][band]
                all_sources[band][idx] = result['sources'][band]
            labels[idx] = result['label']
            theta_Es[idx] = result['theta_E']
            z_lenses[idx] = result['z_lens']
            z_sources[idx] = result['z_source']
            masses[idx] = result['mass']

            done += 1
            if done % 10 == 0 or done == N:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (N - done) / rate if rate > 0 else 0
                print(f'  [{done}/{N}] {elapsed:.0f}s elapsed, {rate:.1f} img/s, ETA {eta/60:.0f}m')

    total = time.time() - t0
    print(f'\nDone — {N} images in {total:.1f}s ({total/N:.2f}s/image)')

    # Print parameter summary
    lensed_mask = labels == 1
    if lensed_mask.any():
        print(f'\nParameter summary (lensed only):')
        print(f'  z_lens:   {z_lenses[lensed_mask].min():.2f} - {z_lenses[lensed_mask].max():.2f}  (mean {z_lenses[lensed_mask].mean():.2f})')
        print(f'  z_source: {z_sources[lensed_mask].min():.2f} - {z_sources[lensed_mask].max():.2f}  (mean {z_sources[lensed_mask].mean():.2f})')
        print(f'  theta_E:  {theta_Es[lensed_mask].min():.2f} - {theta_Es[lensed_mask].max():.2f}  (mean {theta_Es[lensed_mask].mean():.2f})')

    # Save outputs
    print(f'\nSaving to {OUT_DIR}/')
    for band in BANDS:
        np.save(str(OUT_DIR / f'images_{band}.npy'), all_images[band])
        np.save(str(OUT_DIR / f'sources_{band}.npy'), all_sources[band])
        sz = all_images[band].nbytes / 1e6
        print(f'  images_{band}.npy  {sz:.1f} MB')

    np.save(str(OUT_DIR / 'lensed.npy'), labels)
    np.save(str(OUT_DIR / 'theta_Es.npy'), theta_Es)
    np.save(str(OUT_DIR / 'z_lens.npy'), z_lenses)
    np.save(str(OUT_DIR / 'z_source.npy'), z_sources)
    np.save(str(OUT_DIR / 'masses.npy'), masses)

    meta = {
        'version': VERSION,
        'n_images': N,
        'image_size': IMAGE_SIZE,
        'pixel_scale': PIXEL_SCALE,
        'fov_arcsec': IMAGE_SIZE * PIXEL_SCALE,
        'bands': BANDS,
        'params': {
            'z_lens': f'TruncNorm({ZLENS_LOC}, {ZLENS_SCALE}, [{ZLENS_LO}, {ZLENS_HI}])',
            'z_source': f'TruncNorm({ZSRC_LOC}, {ZSRC_SCALE}, [{ZSRC_LO}, {ZSRC_HI}])',
            'sigma_v': f'TruncNorm({SIGMA_LOC}, {SIGMA_SCALE}, [{SIGMA_LO}, {SIGMA_HI}]) km/s',
            'theta_E_filter': f'[{THETA_E_LO}, {THETA_E_HI}] arcsec',
            'log_mass': f'U({LOGMASS_LO}, {LOGMASS_HI})',
        },
    }
    with open(OUT_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Preview
    make_preview(all_images, all_sources, labels, theta_Es, z_lenses, z_sources, N)


if __name__ == '__main__':
    main()
