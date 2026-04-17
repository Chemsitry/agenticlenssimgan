"""
simulate_v8.py — v7 pipeline + cleaned lens stamps (lenses_v8).

v8 change (only one):
  - LENS light: switch from lenses_v7 (hard-edged connected-component
    segmentation) to lenses_v8 (smooth distance-transform segmentation
    taper + mild gaussian denoise). Fixes the chunky low-surface-brightness
    boundary that showed up as speckle around the lens galaxy under arcsinh
    stretch.
  - Sources and background pipeline unchanged from v7 (VELA INTERPOL).

Requires:
    prep_vela_v7.py    populates prepped_mosaic_630/vela_sources/
    prep_lenses_v8.py  populates prepped_mosaic_630/lenses_v8/ (from v7 stamps)

Usage:
    .venv/bin/python3 prep_lenses_v8.py           # post-process v7 lens stamps
    .venv/bin/python3 simulate_v8.py              # 10 test images
    .venv/bin/python3 simulate_v8.py --n 2000     # full dataset
    .venv/bin/python3 simulate_v8.py --sersic     # fallback to v4-style Sersic
"""

import os
import sys
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm
from scipy.ndimage import gaussian_filter, rotate
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

# Parse --sersic early (before module-level stamp loading)
USE_SERSIC = '--sersic' in sys.argv

# ── Configuration ────────────────────────────────────────────────────────

VERSION = 'v8'
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
IMAGE_SIZE = 630
PIXEL_SCALE = 0.03  # arcsec/pix (630 * 0.03 = 18.9" FoV)
PREPPED_DIR = Path('prepped_mosaic_630')
OUT_DIR = Path('output/v8')
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

    print(f"  {band}: sim_to_elec={sim_to_elec:.2f}")

# Real scene cutouts (one-image-per-sample: scene IS the lens's real environment)
scenes = {b: np.load(str(PREPPED_DIR / 'scenes_v8' / f'scenes_{b}.npy'))
          for b in BANDS}
with open(PREPPED_DIR / 'scenes_v8' / 'scene_info.json') as f:
    scene_info = json.load(f)
n_scenes = len(scenes[BANDS[0]])
print(f"  Loaded {n_scenes} real scenes ({scenes[BANDS[0]].shape[1]}x{scenes[BANDS[0]].shape[2]}) from scenes_v8/")


# ── Load galaxy stamps (for INTERPOL light models) ───────────────────────

source_stamps = None
lens_stamps = None
SOURCE_STAMP_SCALE = 0.03
LENS_STAMP_SCALE = 0.03

if USE_SERSIC:
    print("\n  Using SERSIC_ELLIPSE for all light (--sersic flag)")
else:
    # Source stamps — VELA cosmological-sim galaxies (v7 change)
    src_dir = PREPPED_DIR / 'vela_sources'
    if src_dir.exists():
        try:
            source_stamps = {}
            for band in BANDS:
                arr = np.load(str(src_dir / f'stamps_{band}.npy'))
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                source_stamps[band] = arr
            with open(src_dir / 'vela_source_info.json') as f:
                src_info = json.load(f)
            SOURCE_STAMP_SCALE = src_info.get('pixel_scale', 0.03)
            # Remove stamps that are all-zero in any band
            valid = np.ones(len(source_stamps[BANDS[0]]), dtype=bool)
            for band in BANDS:
                valid &= source_stamps[band].sum(axis=(1, 2)) > 0
            for band in BANDS:
                source_stamps[band] = source_stamps[band][valid]
            n_src = len(source_stamps[BANDS[0]])
            print(f"\n  Loaded {n_src} VELA source stamps ({src_info['stamp_size']}x{src_info['stamp_size']})")
        except Exception as e:
            print(f"\n  WARNING: Could not load VELA source stamps: {e}")
            source_stamps = None
    else:
        print(f"\n  No VELA source stamps — using SERSIC_ELLIPSE sources")
        print(f"  (Run prep_vela_v7.py first)")

    # v8 architectural change: no lens stamps. The lens galaxy is already
    # present in the real scene cutout (scenes_v8/) — we don't model it.
    mode_src = "INTERPOL" if source_stamps is not None else "SERSIC_ELLIPSE"
    print(f"  Source model: {mode_src}  |  Lens light: REAL (scene cutout)")


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


def make_delta_psf():
    """Delta-function PSF for INTERPOL stamps that are already PSF-convolved."""
    delta = np.zeros((3, 3))
    delta[1, 1] = 1.0
    return PSF(psf_type='PIXEL',
               kernel_point_source=delta,
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


KWARGS_NUMERICS_SERSIC = {
    'supersampling_factor': 3,
    'supersampling_convolution': True,
}

KWARGS_NUMERICS_INTERPOL = {
    'supersampling_factor': 5,
    'supersampling_convolution': True,
}


def augment_stamp(stamp, angle, do_flip):
    """Apply rotation + optional flip with explicit, shared parameters.

    Angle and flip are decided once per system (outside the per-band loop)
    so all 4 bands of the same galaxy get the SAME transform — preserving
    physical cross-band morphology consistency.
    """
    out = rotate(stamp, angle, reshape=False, order=1, mode='constant', cval=0.0)
    if do_flip:
        out = np.ascontiguousarray(np.fliplr(out))
    out = np.clip(out, 0, None)
    total = out.sum()
    if total > 0:
        out /= total
    return out


def augment_stamp_absolute(stamp, angle, do_flip):
    """For absolute-calibrated stamps (v8 lens): rotate/flip WITHOUT
    renormalizing or clipping negatives. Preserves natural flux levels and
    sky-noise fluctuations (which are small negative values around zero)."""
    out = rotate(stamp, angle, reshape=False, order=1, mode='constant', cval=0.0)
    if do_flip:
        out = np.ascontiguousarray(np.fliplr(out))
    return out


# ── Simulate one system ──────────────────────────────────────────────────

def simulate_one(lensed=True, seed=None, scene_idx=None):
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

    # Lens mass ellipticity
    e1, e2 = 0.05, 0.02

    # Decide light model modes
    use_interpol_src = source_stamps is not None

    # Source stamp selection + augmentation params (SHARED across bands)
    if use_interpol_src:
        n_stamps = len(source_stamps[BANDS[0]])
        src_stamp_idx = int(rng.integers(n_stamps))
        src_angle = float(rng.uniform(0, 360))
        src_flip = bool(rng.random() > 0.5)
    else:
        R_sersic_src = 0.15
        n_sersic_src = 1.5
        e1s, e2s = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    # Scene (real JWST cutout containing a real elliptical near center)
    if scene_idx is None:
        scene_idx = int(rng.integers(n_scenes))

    # Source position — same as v4 (validated)
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

    # SED color ratios (source only — lens light is real data in the scene,
    # already has its own SED baked in)
    uv_slope = float(np.clip(rng.normal(-0.5, 1.0), -2.5, 1.5))
    src_colors = starforming_color_ratios(z_source, uv_slope=uv_slope)
    # Normalize so F444W = 1.0 (reference band for total-flux anchoring below).
    _ref = src_colors['F444W'] if src_colors['F444W'] > 0 else 1.0
    src_colors = {b: src_colors[b] / _ref for b in BANDS}

    # Anchor the lensed arc flux to the scene's real F444W central brightness,
    # THEN apply per-band color ratios so the arc has the right SED
    # (star-forming galaxy at z_source, with Lyman break + Balmer break).
    scene_F444W = scenes['F444W'][scene_idx]
    sH_ref = scene_F444W.shape[0] // 2
    ref_lens_peak = float(
        scene_F444W[sH_ref - 20:sH_ref + 20, sH_ref - 20:sH_ref + 20].max())
    ref_flux_proxy = ref_lens_peak * 50.0  # ~ effective-area × peak

    # Render all bands
    band_results = {}
    target_ratio = 0.25  # arc total flux ÷ lens F444W total flux (at F444W)

    for band in BANDS:
        data_class = ImageData(**make_kwargs_data())
        psf_for_src = make_psf_obj(band)
        kwargs_num = (KWARGS_NUMERICS_INTERPOL if use_interpol_src
                      else KWARGS_NUMERICS_SERSIC)

        # Source model
        if use_interpol_src:
            stamp = augment_stamp(source_stamps[band][src_stamp_idx].copy(),
                                  src_angle, src_flip)
            source_model = LightModel(['INTERPOL'])
            kwargs_source = [{
                'image': stamp.astype(np.float64),
                'amp': 1,
                'center_x': float(center_x),
                'center_y': float(center_y),
                'phi_G': 0.0,
                'scale': SOURCE_STAMP_SCALE,
            }]
        else:
            source_model = LightModel(['SERSIC_ELLIPSE'])
            kwargs_source = [{
                'amp': 1, 'R_sersic': R_sersic_src, 'n_sersic': n_sersic_src,
                'e1': float(e1s), 'e2': float(e2s),
                'center_x': float(center_x), 'center_y': float(center_y)
            }]

        # No lens-light model — the lens is real data in the scene cutout.
        # Use a dummy lens-light that renders to zero so lenstronomy is happy.
        lens_light_model = LightModel(['SERSIC_ELLIPSE'])
        kwargs_lens_light = [{'amp': 0.0, 'R_sersic': 0.1, 'n_sersic': 1.0,
                              'e1': 0.0, 'e2': 0.0,
                              'center_x': 0.0, 'center_y': 0.0}]

        lens_model = LensModel(['SIE', 'SHEAR'],
                                z_lens=z_lens, z_source=z_source)

        im_src = ImageModel(
            data_class=data_class, psf_class=psf_for_src,
            lens_model_class=lens_model,
            source_model_class=source_model,
            lens_light_model_class=lens_light_model,
            kwargs_numerics=kwargs_num)

        # Calibrate source amp to target brightness relative to the scene's
        # central galaxy. Peak of the scene's central 40x40 region approximates
        # the lens galaxy peak; we scale the source so its integrated flux is
        # a target_ratio fraction of the lens peak × an effective area.
        kw_src_cal = [{**kwargs_source[0], 'amp': 1.0}]
        img_src_unit = im_src.image(kwargs_lens, kw_src_cal,
                                     kwargs_lens_light=kwargs_lens_light,
                                     source_add=True, lens_light_add=False)
        sum_src_unit = float(np.sum(img_src_unit))
        if sum_src_unit <= 0:
            return simulate_one(lensed=lensed,
                                seed=int(rng.integers(int(1e9))))

        scene = scenes[band][scene_idx]
        # Source amp: anchored to F444W lens peak, scaled by per-band src color.
        # This imposes a physical SED on the arc regardless of source type
        # (VELA stamps lost intrinsic colors during per-band normalization;
        # Sersic never had colors).
        amp_src = (ref_flux_proxy * target_ratio * src_colors[band]
                    / sum_src_unit) if lensed else 0.0

        kw_src = [{**kwargs_source[0], 'amp': amp_src}]
        image_src_part = im_src.image(kwargs_lens, kw_src,
                                        kwargs_lens_light=kwargs_lens_light,
                                        source_add=True,
                                        lens_light_add=False)

        band_results[band] = {
            'image_src_only': image_src_part.astype(np.float32),
            'scene': scene.astype(np.float32),
            'scene_idx': scene_idx,
        }

    return band_results, theta_E, z_lens, z_source, mass


# ── Parallel worker ──────────────────────────────────────────────────────

def _simulate_worker(args):
    idx, lensed, seed, noise_seed, scene_idx = args
    result = simulate_one(lensed=lensed, seed=seed, scene_idx=scene_idx)
    band_results, theta_E, z_lens, z_source, mass = result

    rng_noise = np.random.default_rng(noise_seed)

    out = {'idx': idx, 'label': 1.0 if lensed else 0.0,
           'theta_E': theta_E, 'z_lens': z_lens, 'z_source': z_source, 'mass': mass,
           'images': {}, 'sources': {}, 'galaxies': {}}

    for band in BANDS:
        src_part = band_results[band]['image_src_only']
        scene = band_results[band]['scene']
        # Poisson noise only on the NEW source photons we're injecting — the
        # scene already contains its real JWST observation noise.
        src_noisy = add_poisson_noise(src_part, band, rng=rng_noise)
        out['images'][band] = (scene + src_noisy).astype(np.float32)
        out['sources'][band] = src_part
        out['galaxies'][band] = scene.astype(np.float32)

    return out


# ── RGB rendering ────────────────────────────────────────────────────────

def make_rgb(r, g, b, smooth=1.0, gamma=0.50, lum_gate=0.0):
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

    mode = 'VELA INTERPOL' if source_stamps is not None else 'Sersic'
    plt.suptitle(f'simulate_v8 [{mode}] — {N} images (630px, 18.9" FoV)', fontsize=12)
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
    global OUT_DIR
    parser = argparse.ArgumentParser(description='v8 multi-band lens simulation (VELA sources + cleaned lens stamps)')
    parser.add_argument('--n', type=int, default=10, help='Number of images (half lensed, half not)')
    parser.add_argument('--sersic', action='store_true', help='Force Sersic light profiles (v4 mode)')
    parser.add_argument('--out_dir', default=None, help='Output directory (default: output/v8)')
    args = parser.parse_args()

    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    N = args.n
    N_EACH = N // 2

    print(f"\nsimulate_v8 — VELA sources + cleaned lens stamps (lenses_v8)")
    print(f"  {IMAGE_SIZE}x{IMAGE_SIZE} px @ {PIXEL_SCALE}\"/pix = {IMAGE_SIZE*PIXEL_SCALE:.1f}\" FoV")
    print(f"\nGenerating {N} images ({N_EACH} lensed + {N_EACH} non-lensed)...")

    # Storage
    all_images = {}
    all_sources = {}
    all_galaxies = {}
    for band in BANDS:
        all_images[band] = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        all_sources[band] = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        all_galaxies[band] = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    labels = np.zeros(N)
    theta_Es = np.zeros(N)
    z_lenses = np.zeros(N)
    z_sources = np.zeros(N)
    masses = np.zeros(N)

    # Pre-generate seeds and scene assignments.
    # Non-lensed images must have unique scenes (otherwise they'd be
    # pixel-identical — same real JWST cutout with nothing added).
    # Lensed images may reuse scenes (lensed arcs differ per system).
    rng_main = np.random.default_rng(99)
    scene_indices_nonlensed = rng_main.choice(
        n_scenes, size=N_EACH, replace=False)
    scene_indices_lensed = rng_main.integers(
        n_scenes, size=N_EACH)
    jobs = []
    for i in range(N_EACH):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i, False, seed, noise_seed,
                     int(scene_indices_nonlensed[i])))
    for i in range(N_EACH):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i + N_EACH, True, seed, noise_seed,
                     int(scene_indices_lensed[i])))

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
                all_galaxies[band][idx] = result['galaxies'][band]
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
        np.save(str(OUT_DIR / f'galaxies_{band}.npy'), all_galaxies[band])
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
