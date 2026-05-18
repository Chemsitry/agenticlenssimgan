"""
simulate_v9_consistent.py — v8 pipeline with **physically consistent** lens
mass and lens light.

The only change vs simulate_v8.py:

  σ_v is no longer drawn from TruncNorm(180, 50). Instead, for each scene
  cutout it is **derived from the cutout's own apparent F115W/F150W/F277W
  photometry** via a Faber-Jackson calibration (slope fixed at -10, per-band
  intercept fit on 5 DESI×JWST DEV galaxies — see fit_calibration.py and
  fj_params.json in this directory). This ties the SIE Einstein radius to
  the brightness of the actual galaxy that's painted into the scene.

  An inter-band consistency check rejects cutouts whose per-band predictions
  span more than SIGMA_CONSISTENCY_MAX_DEX dex (default 0.40, ~factor 2.5);
  the simulation picks a different scene in that case.

Inputs are read from ~/Desktop/data prep/prepped_mosaic_630/ (same as v8).
Outputs are written to ~/Desktop/data prep v9_consistent/output/v9_consistent/
so the original v8 outputs are never touched.

Usage:
    python3 simulate_v9_consistent.py              # default test run
    python3 simulate_v9_consistent.py --n 2000     # full dataset
    python3 simulate_v9_consistent.py --sersic     # source = Sersic instead of VELA
"""

import os
import sys
sys.setrecursionlimit(50000)  # simulate_one retries via tail recursion when scenes fall outside the target θ_E bin
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

# Parse --sersic and --scenes early (before module-level stamp loading)
USE_SERSIC = '--sersic' in sys.argv

# --scenes v8 | v9 | v10 (default = auto: prefer v10, then v9, then v8)
def _parse_scenes_flag():
    if '--scenes' in sys.argv:
        i = sys.argv.index('--scenes')
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower()
    return 'auto'
SCENES_VERSION = _parse_scenes_flag()

# ── Configuration ────────────────────────────────────────────────────────

VERSION = 'v9_consistent'
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
IMAGE_SIZE = 630
PIXEL_SCALE = 0.03  # arcsec/pix (630 * 0.03 = 18.9" FoV)

# v9_consistent: read inputs from the v8 project; write to a new dir so we
# never overwrite v8 outputs.
DATA_PREP_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep')
PREPPED_DIR = DATA_PREP_ROOT / 'prepped_mosaic_630'
V9_ROOT     = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
OUT_DIR     = V9_ROOT / 'output' / 'v9_consistent'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# σ_v calibration + cutout photometry live in this directory
import sys as _sys
_sys.path.insert(0, str(V9_ROOT))
import cutout_photometry as cphot
import calibration as cal

# Photometry / consistency tuning
PHOTOM_APERTURE_PX = 60                # 1.8" radius circular aperture on scene center
PHOTOM_SKY_ANNULUS = (70, 95)          # residual-sky annulus
SIGMA_CONSISTENCY_MAX_DEX = 0.40       # max spread of per-band log10(σ_v) predictions; >0.4 dex (factor 2.5) rejects

sum_to_flux = 6.501853565914121
gain = 2.05

# ── COWLS-calibrated parameter distributions ─────────────────────────────

ZLENS_LOC, ZLENS_SCALE = 0.7, 0.4
ZLENS_LO, ZLENS_HI = 0.05, 2.5

ZSRC_LOC, ZSRC_SCALE = 2.5, 1.5
ZSRC_LO, ZSRC_HI = 0.5, 7.0

SIGMA_LOC, SIGMA_SCALE = 180, 50
SIGMA_LO, SIGMA_HI = 80, 350

THETA_E_LO, THETA_E_HI = 0.5, 2.5   # widened for stratified-bin generation
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
# Scene-catalog selection. Default: prefer v11 (JADES ellipticals) → v10 → v9 → v8.
# Override with --scenes vXX to force.
import pandas as _pd
_V12_SCENES_DIR = V9_ROOT / 'prepped_scenes_v12'
_V11_SCENES_DIR = V9_ROOT / 'prepped_scenes_v11'
_V10_SCENES_DIR = V9_ROOT / 'prepped_scenes_v10'
_V9_SCENES_DIR  = V9_ROOT / 'prepped_scenes_v9'
_FORCE_V8 = SCENES_VERSION == 'v8'
_FORCE_V9 = SCENES_VERSION == 'v9'
_FORCE_V10 = SCENES_VERSION == 'v10'
_FORCE_V11 = SCENES_VERSION == 'v11'
_FORCE_V12 = SCENES_VERSION == 'v12'
if (not _FORCE_V8 and not _FORCE_V9 and not _FORCE_V10 and not _FORCE_V11) and (_FORCE_V12 or (SCENES_VERSION == 'auto' and (_V12_SCENES_DIR / f'scenes_{BANDS[0]}.npy').exists())):
    scenes = {b: np.load(str(_V12_SCENES_DIR / f'scenes_{b}.npy')) for b in BANDS}
    with open(_V12_SCENES_DIR / 'scene_info.json') as f:
        scene_info = json.load(f)
    scenes_manifest = _pd.read_parquet(_V12_SCENES_DIR / 'manifest.parquet')
    n_scenes = len(scenes[BANDS[0]])
    SCENES_SOURCE = 'v12 unified (JADES DR5 + DESI calibration galaxies)'
    print(f"  Loaded {n_scenes} v12 scenes from prepped_scenes_v12/")
    print(f"  z_phot range: {scenes_manifest['Z_PHOT_MEAN'].min():.2f} – {scenes_manifest['Z_PHOT_MEAN'].max():.2f}")
    print(f"  by field: {scenes_manifest['field'].value_counts().to_dict()}")
    print(f"  with measured σ_v: {int(scenes_manifest['has_measured_sigma_v'].sum())}")
elif (not _FORCE_V8 and not _FORCE_V9 and not _FORCE_V10) and (_FORCE_V11 or (SCENES_VERSION == 'auto' and (_V11_SCENES_DIR / f'scenes_{BANDS[0]}.npy').exists())):
    scenes = {b: np.load(str(_V11_SCENES_DIR / f'scenes_{b}.npy')) for b in BANDS}
    with open(_V11_SCENES_DIR / 'scene_info.json') as f:
        scene_info = json.load(f)
    scenes_manifest = _pd.read_parquet(_V11_SCENES_DIR / 'manifest.parquet')
    n_scenes = len(scenes[BANDS[0]])
    SCENES_SOURCE = 'v11 JADES DR5 confirmed ellipticals (GOODS-N + GOODS-S)'
    print(f"  Loaded {n_scenes} v11 JADES-elliptical scenes from prepped_scenes_v11/")
    print(f"  z_phot range: {scenes_manifest['Z_PHOT_MEAN'].min():.2f} – {scenes_manifest['Z_PHOT_MEAN'].max():.2f}")
    print(f"  by field: {scenes_manifest['field'].value_counts().to_dict()}")
elif (not _FORCE_V8 and not _FORCE_V9) and (_FORCE_V10 or (SCENES_VERSION == 'auto' and (_V10_SCENES_DIR / f'scenes_{BANDS[0]}.npy').exists())):
    scenes = {b: np.load(str(_V10_SCENES_DIR / f'scenes_{b}.npy')) for b in BANDS}
    with open(_V10_SCENES_DIR / 'scene_info.json') as f:
        scene_info = json.load(f)
    scenes_manifest = _pd.read_parquet(_V10_SCENES_DIR / 'manifest.parquet')
    n_scenes = len(scenes[BANDS[0]])
    SCENES_SOURCE = 'v10 multi-field (COSMOS-Web + PRIMER-UDS + JADES GOODS-N + CEERS)'
    print(f"  Loaded {n_scenes} v10 scenes ({scenes[BANDS[0]].shape[1]}x{scenes[BANDS[0]].shape[2]}) from prepped_scenes_v10/")
    print(f"  z_phot range: {scenes_manifest['Z_PHOT_MEAN'].min():.2f} – {scenes_manifest['Z_PHOT_MEAN'].max():.2f}")
    print(f"  by field: {scenes_manifest['field'].value_counts().to_dict()}")
elif (not _FORCE_V8) and (_FORCE_V9 or (SCENES_VERSION == 'auto' and (_V9_SCENES_DIR / f'scenes_{BANDS[0]}.npy').exists())):
    scenes = {b: np.load(str(_V9_SCENES_DIR / f'scenes_{b}.npy')) for b in BANDS}
    with open(_V9_SCENES_DIR / 'scene_info.json') as f:
        scene_info = json.load(f)
    scenes_manifest = _pd.read_parquet(_V9_SCENES_DIR / 'manifest.parquet')
    n_scenes = len(scenes[BANDS[0]])
    SCENES_SOURCE = 'v9 confirmed-DEV with known photo-z'
    print(f"  Loaded {n_scenes} v9 scenes ({scenes[BANDS[0]].shape[1]}x{scenes[BANDS[0]].shape[2]}) from prepped_scenes_v9/")
    print(f"  z_phot range: {scenes_manifest['Z_PHOT_MEAN'].min():.2f} – {scenes_manifest['Z_PHOT_MEAN'].max():.2f}")
else:
    scenes = {b: np.load(str(PREPPED_DIR / 'scenes_v8' / f'scenes_{b}.npy'))
              for b in BANDS}
    with open(PREPPED_DIR / 'scenes_v8' / 'scene_info.json') as f:
        scene_info = json.load(f)
    scenes_manifest = None
    n_scenes = len(scenes[BANDS[0]])
    SCENES_SOURCE = 'v8 (legacy — no known photo-z per scene)'
    print(f"  Loaded {n_scenes} v8 scenes ({scenes[BANDS[0]].shape[1]}x{scenes[BANDS[0]].shape[2]}) from scenes_v8/")


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

class _RetryAttempt(Exception):
    """Internal — raised when the current attempt should be re-rolled."""
    pass


def simulate_one(lensed=True, seed=None, scene_idx=None, target_theta_range=None,
                 eligible_scenes=None):
    """Iteratively retry. If eligible_scenes is given, retries pick from that pool
    instead of all n_scenes — preserves per-bin scene selection across retries."""
    MAX_RETRIES = 50000
    base_rng = np.random.default_rng(seed)
    for attempt in range(MAX_RETRIES):
        if attempt == 0:
            cur_scene = scene_idx
        elif eligible_scenes is not None and len(eligible_scenes):
            cur_scene = int(eligible_scenes[base_rng.integers(len(eligible_scenes))])
        else:
            cur_scene = None  # let _simulate_one_attempt pick randomly
        try:
            return _simulate_one_attempt(
                lensed=lensed,
                seed=int(base_rng.integers(int(1e9))),
                scene_idx=cur_scene,
                target_theta_range=target_theta_range,
            )
        except _RetryAttempt:
            continue
    raise RuntimeError(f"simulate_one: gave up after {MAX_RETRIES} retries "
                       f"(target_theta_range={target_theta_range})")


def _simulate_one_attempt(lensed=True, seed=None, scene_idx=None, target_theta_range=None):
    """One attempt at simulating a lens system. Raises _RetryAttempt to retry."""
    rng = np.random.default_rng(seed)

    # ── v9 change ────────────────────────────────────────────────────────
    # Choose scene_idx FIRST so we can read the cutout galaxy's actual photo-z
    # from the v9 manifest (if available) and use that as z_lens. Falls back
    # to a TruncNorm draw if no manifest is loaded (i.e., legacy v8 scenes).
    if scene_idx is None:
        scene_idx = int(rng.integers(n_scenes))

    if scenes_manifest is not None:
        # Real photo-z of the galaxy in this cutout — no more random draw.
        z_lens = float(scenes_manifest['Z_PHOT_MEAN'].iloc[scene_idx])
        # Clip to simulation bounds so downstream code never sees a wild value.
        z_lens = float(np.clip(z_lens, ZLENS_LO, ZLENS_HI))
    else:
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

    # ── v9_consistent change ──────────────────────────────────────────────
    # Derive σ_v from the lens cutout's photometry instead of drawing from
    # TruncNorm. This ties the bending strength to the actual brightness
    # of the galaxy painted into the scene, fixing the v8 inconsistency
    # where lens light and lens mass were sampled independently.
    # scene_idx was chosen above (z_lens block); no need to re-sample here.

    # Measure the lens galaxy's apparent AB mag in each calibration band.
    photom_mags = {}
    for _band in ('F115W', 'F150W', 'F277W'):
        photom_mags[_band] = cphot.cutout_ab_mag(
            scenes[_band][scene_idx],
            aperture_radius_pix=PHOTOM_APERTURE_PX,
            sky_annulus=PHOTOM_SKY_ANNULUS,
        )

    # If the manifest has a measured DESI σ_v for this scene, use it directly
    # — strictly more accurate than FJ-predicted σ_v.
    use_measured = False
    sigma_v_per_band = {}
    sigma_v_consensus = float('nan')
    if (scenes_manifest is not None
            and 'has_measured_sigma_v' in scenes_manifest.columns
            and bool(scenes_manifest['has_measured_sigma_v'].iloc[scene_idx])):
        sigma_v_consensus = float(scenes_manifest['sigma_v_measured'].iloc[scene_idx])
        use_measured = True
    else:
        sigma_v_consensus, sigma_v_per_band = cal.predict_sigma_v_multiband(
            photom_mags, z_lens)
        band_log_sigs = np.array(
            [np.log10(v) for v in sigma_v_per_band.values()
             if np.isfinite(v) and v > 0])
        if (band_log_sigs.size < 2
                or (band_log_sigs.max() - band_log_sigs.min())
                    > SIGMA_CONSISTENCY_MAX_DEX
                or not np.isfinite(sigma_v_consensus)):
            raise _RetryAttempt()

    sigma_v = float(np.clip(sigma_v_consensus, SIGMA_LO, SIGMA_HI))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    # Per-bin θ_E target if specified (for stratified generation), else global window.
    te_lo, te_hi = target_theta_range if target_theta_range is not None else (THETA_E_LO, THETA_E_HI)
    if lensed and not (te_lo <= theta_E <= te_hi):
        raise _RetryAttempt()

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

    # v9 note: scene_idx was already chosen above so σ_v could be derived from
    # its photometry; no need to re-sample here.

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
    # v9 multi-field: F444W mosaic may not be available for non-COSMOS-Web scenes.
    # If F444W peak is essentially zero, fall back to F277W as the brightness anchor.
    scene_F444W = scenes['F444W'][scene_idx]
    sH_ref = scene_F444W.shape[0] // 2
    ref_lens_peak = float(
        scene_F444W[sH_ref - 20:sH_ref + 20, sH_ref - 20:sH_ref + 20].max())
    if ref_lens_peak <= 0.1:
        scene_ref = scenes['F277W'][scene_idx]
        ref_lens_peak = float(
            scene_ref[sH_ref - 20:sH_ref + 20, sH_ref - 20:sH_ref + 20].max())
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
            raise _RetryAttempt()

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

    return (band_results, theta_E, z_lens, z_source, mass,
            sigma_v, photom_mags, sigma_v_per_band)


# ── Parallel worker ──────────────────────────────────────────────────────

def _simulate_worker(args):
    if len(args) == 7:
        idx, lensed, seed, noise_seed, scene_idx, target_theta_range, eligible_scenes = args
    elif len(args) == 6:
        idx, lensed, seed, noise_seed, scene_idx, target_theta_range = args
        eligible_scenes = None
    else:
        idx, lensed, seed, noise_seed, scene_idx = args
        target_theta_range = None
        eligible_scenes = None
    # Catch all simulator-level failures here, INSIDE the worker process.
    # Raising would crash the multiprocessing pool's communication; returning
    # a "failed" marker keeps the pool healthy and lets the parent skip cleanly.
    try:
        result = simulate_one(lensed=lensed, seed=seed, scene_idx=scene_idx,
                              target_theta_range=target_theta_range,
                              eligible_scenes=eligible_scenes)
    except Exception as e:
        return {'idx': idx, 'failed': True, 'error': f'{type(e).__name__}: {e}',
                'target_theta_range': target_theta_range}
    (band_results, theta_E, z_lens, z_source, mass,
     sigma_v, photom_mags, sigma_v_per_band) = result

    rng_noise = np.random.default_rng(noise_seed)

    out = {'idx': idx, 'label': 1.0 if lensed else 0.0,
           'theta_E': theta_E, 'z_lens': z_lens, 'z_source': z_source, 'mass': mass,
           # v9: σ_v derived from cutout photometry, not random draw
           'sigma_v': float(sigma_v),
           'photom_mag_F115W': float(photom_mags.get('F115W', np.nan)),
           'photom_mag_F150W': float(photom_mags.get('F150W', np.nan)),
           'photom_mag_F277W': float(photom_mags.get('F277W', np.nan)),
           'sigma_v_F115W':    float(sigma_v_per_band.get('F115W', np.nan)),
           'sigma_v_F150W':    float(sigma_v_per_band.get('F150W', np.nan)),
           'sigma_v_F277W':    float(sigma_v_per_band.get('F277W', np.nan)),
           'images': {}, 'arcs': {}, 'galaxies': {}}

    for band in BANDS:
        src_part = band_results[band]['image_src_only']
        scene = band_results[band]['scene']
        # Poisson noise only on the NEW source photons we're injecting — the
        # scene already contains its real JWST observation noise.
        src_noisy = add_poisson_noise(src_part, band, rng=rng_noise)
        out['images'][band] = (scene + src_noisy).astype(np.float32)
        out['arcs'][band] = src_part
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

def make_preview(all_images, all_arcs, labels, theta_Es, z_lenses, z_sources, N):
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
    plt.suptitle(f'simulate_v9_consistent [{mode}, photom-σ_v] — {N} images', fontsize=12)
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
    parser.add_argument('--n', type=int, default=10, help='Total images (half lensed, half not) — legacy mode')
    parser.add_argument('--lensed', type=int, default=None,
                        help='Number of lensed images (overrides --n half if given)')
    parser.add_argument('--nonlensed', type=int, default=None,
                        help='Number of non-lensed images (overrides --n half if given)')
    parser.add_argument('--theta-e-min', type=float, default=0.5)
    parser.add_argument('--theta-e-max', type=float, default=2.5)
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Number of θ_E bins to balance the lensed sample across')
    parser.add_argument('--sersic', action='store_true', help='Force Sersic light profiles (v4 mode)')
    parser.add_argument('--out_dir', default=None, help='Output directory')
    parser.add_argument('--scenes', default='auto', choices=['auto', 'v8', 'v9', 'v10', 'v11', 'v12'],
                        help='Which scene catalog to use (default auto = prefer v12)')
    args = parser.parse_args()

    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Asymmetric counts: prefer explicit --lensed/--nonlensed; fall back to --n/2 each.
    if args.lensed is not None or args.nonlensed is not None:
        N_LENSED   = args.lensed if args.lensed is not None else args.n // 2
        N_NONLENSED = args.nonlensed if args.nonlensed is not None else args.n // 2
    else:
        N_LENSED = N_NONLENSED = args.n // 2
    N = N_LENSED + N_NONLENSED
    N_BINS = max(1, args.n_bins)
    BIN_EDGES = np.linspace(args.theta_e_min, args.theta_e_max, N_BINS + 1)
    PER_BIN = N_LENSED // N_BINS
    EXTRA = N_LENSED - PER_BIN * N_BINS  # any remainder goes into the first bins

    print(f"\nsimulate_v9_consistent — stratified θ_E generation")
    print(f"  {IMAGE_SIZE}x{IMAGE_SIZE} px @ {PIXEL_SCALE}\"/pix = {IMAGE_SIZE*PIXEL_SCALE:.1f}\" FoV")
    print(f"  total: {N} images ({N_LENSED} lensed across {N_BINS} bins, {N_NONLENSED} non-lensed)")
    print(f"  θ_E bins [{args.theta_e_min}, {args.theta_e_max}\"] → {PER_BIN} per bin (+{EXTRA} extras)")

    all_images = {}
    all_arcs = {}
    all_galaxies = {}

    rng_main = np.random.default_rng(99)
    # Non-lensed: unique scenes only (else two images are byte-identical).
    if N_NONLENSED > n_scenes:
        print(f'  WARN: requested {N_NONLENSED} non-lensed but only {n_scenes} unique scenes — capping to {n_scenes}')
        N_NONLENSED = n_scenes
        N = N_LENSED + N_NONLENSED
    scene_indices_nonlensed = rng_main.choice(n_scenes, size=N_NONLENSED, replace=False)
    scene_indices_lensed = rng_main.integers(n_scenes, size=N_LENSED)

    # Storage as numpy memmaps: 12 arrays × ~9 GB each would exceed RAM on a 48 GB Mac
    # if held in memory. memmap streams to/from disk under OS paging.
    print(f'  Allocating memmaps in {OUT_DIR}/ (each is ~{N * IMAGE_SIZE * IMAGE_SIZE * 4 / 1e9:.1f} GB)')
    for band in BANDS:
        all_images[band] = np.lib.format.open_memmap(
            OUT_DIR / f'images_{band}.npy', mode='w+',
            dtype=np.float32, shape=(N, IMAGE_SIZE, IMAGE_SIZE))
        all_arcs[band] = np.lib.format.open_memmap(
            OUT_DIR / f'arcs_{band}.npy', mode='w+',
            dtype=np.float32, shape=(N, IMAGE_SIZE, IMAGE_SIZE))
        all_galaxies[band] = np.lib.format.open_memmap(
            OUT_DIR / f'galaxies_{band}.npy', mode='w+',
            dtype=np.float32, shape=(N, IMAGE_SIZE, IMAGE_SIZE))
    labels = np.zeros(N); theta_Es = np.zeros(N); z_lenses = np.zeros(N)
    z_sources = np.zeros(N); masses = np.zeros(N)
    sigma_vs = np.zeros(N)
    photom_F115W = np.full(N, np.nan); photom_F150W = np.full(N, np.nan); photom_F277W = np.full(N, np.nan)
    sigma_v_F115W = np.full(N, np.nan); sigma_v_F150W = np.full(N, np.nan); sigma_v_F277W = np.full(N, np.nan)

    # Pre-compute each scene's predicted θ_E at a fiducial z_source.
    # This lets us pick scenes intelligently for each target bin instead of
    # randomly retrying.
    print('\nProfiling scene θ_E predictions (one-time, ~30s)...')
    scene_theta_predicted = np.full(n_scenes, np.nan)
    Z_SOURCE_REF = 2.5
    for s_idx in range(n_scenes):
        # Use measured σ_v if available, else photometry-derived
        if scenes_manifest is not None and 'has_measured_sigma_v' in scenes_manifest.columns \
                and bool(scenes_manifest['has_measured_sigma_v'].iloc[s_idx]):
            sigma_v = float(scenes_manifest['sigma_v_measured'].iloc[s_idx])
            z_lens_s = float(scenes_manifest['Z_PHOT_MEAN'].iloc[s_idx])
        else:
            try:
                mags = {b: cphot.cutout_ab_mag(
                    scenes[b][s_idx], aperture_radius_pix=PHOTOM_APERTURE_PX,
                    sky_annulus=PHOTOM_SKY_ANNULUS) for b in ('F115W','F150W','F277W')}
                z_lens_s = float(scenes_manifest['Z_PHOT_MEAN'].iloc[s_idx]) if scenes_manifest is not None else 0.5
                z_lens_s = float(np.clip(z_lens_s, ZLENS_LO, ZLENS_HI))
                sigma_v_pred, _ = cal.predict_sigma_v_multiband(mags, z_lens_s)
                if not np.isfinite(sigma_v_pred):
                    continue
                sigma_v = float(np.clip(sigma_v_pred, SIGMA_LO, SIGMA_HI))
            except Exception:
                continue
        try:
            zs = max(Z_SOURCE_REF, z_lens_s + 0.1)
            lc = LensCosmo(z_lens_s, zs)
            scene_theta_predicted[s_idx] = lc.sis_sigma_v2theta_E(sigma_v)
        except Exception:
            continue
    n_profiled = int(np.isfinite(scene_theta_predicted).sum())
    print(f'  profiled {n_profiled}/{n_scenes} scenes; θ_E range at z_s={Z_SOURCE_REF}: '
          f'{np.nanmin(scene_theta_predicted):.2f} – {np.nanmax(scene_theta_predicted):.2f}"')

    jobs = []
    # Non-lensed jobs
    for i in range(N_NONLENSED):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i, False, seed, noise_seed,
                     int(scene_indices_nonlensed[i]), None))
    # Lensed jobs — one per (bin, slot). For each bin, pre-filter scenes
    # whose predicted θ_E is "near" the bin (within a generous factor of 2).
    job_idx = N_NONLENSED
    for bin_i in range(N_BINS):
        count = PER_BIN + (1 if bin_i < EXTRA else 0)
        lo = float(BIN_EDGES[bin_i]); hi = float(BIN_EDGES[bin_i + 1])
        # Allow scenes whose predicted θ_E spans roughly half-to-double the target
        # (because z_source draws can shift θ_E by ~30-50% in practice).
        bin_mask = (scene_theta_predicted > lo * 0.5) & (scene_theta_predicted < hi * 2.0)
        eligible = np.where(bin_mask)[0]
        if len(eligible) == 0:
            # Fall back to all scenes with finite prediction in the upper tail
            eligible = np.where(np.isfinite(scene_theta_predicted))[0]
            order = np.argsort(scene_theta_predicted[eligible])[-50:]
            eligible = eligible[order]
            print(f'  bin {bin_i} [{lo:.2f},{hi:.2f}]: no plausible scenes; falling back to {len(eligible)} brightest')
        else:
            print(f'  bin {bin_i} [{lo:.2f},{hi:.2f}"]: {len(eligible)} eligible scenes')
        eligible_tuple = tuple(int(x) for x in eligible)
        for _ in range(count):
            seed = int(rng_main.integers(int(1e9)))
            noise_seed = int(rng_main.integers(int(1e9)))
            scene_idx = int(eligible[rng_main.integers(len(eligible))])
            jobs.append((job_idx, True, seed, noise_seed, scene_idx, (lo, hi), eligible_tuple))
            job_idx += 1

    # Parallel generation
    n_workers = mp.cpu_count()
    ctx = mp.get_context('fork')
    chunksize = max(1, min(50, N // (n_workers * 4)))
    print(f"  Using {n_workers} workers (fork), chunksize={chunksize}")

    done = 0
    t0 = time.time()

    completed_mask = np.zeros(N, dtype=bool)
    try:
        with ctx.Pool(n_workers) as pool:
            it = pool.imap_unordered(_simulate_worker, jobs, chunksize=chunksize)
            n_failed = 0
            while True:
                try:
                    result = next(it)
                except StopIteration:
                    break
                except Exception as e:
                    print(f'  WARN: pool delivered an exception: {type(e).__name__}: {e}')
                    continue
                if result.get('failed'):
                    n_failed += 1
                    if n_failed <= 5:  # only print first 5
                        print(f"  WARN: job {result['idx']} failed (bin={result.get('target_theta_range')}): {result.get('error')}")
                    continue
                idx = result['idx']
                for band in BANDS:
                    all_images[band][idx] = result['images'][band]
                    all_arcs[band][idx] = result['arcs'][band]
                    all_galaxies[band][idx] = result['galaxies'][band]
                labels[idx] = result['label']
                theta_Es[idx] = result['theta_E']
                z_lenses[idx] = result['z_lens']
                z_sources[idx] = result['z_source']
                masses[idx] = result['mass']
                sigma_vs[idx]      = result.get('sigma_v', np.nan)
                photom_F115W[idx]  = result.get('photom_mag_F115W', np.nan)
                photom_F150W[idx]  = result.get('photom_mag_F150W', np.nan)
                photom_F277W[idx]  = result.get('photom_mag_F277W', np.nan)
                sigma_v_F115W[idx] = result.get('sigma_v_F115W', np.nan)
                sigma_v_F150W[idx] = result.get('sigma_v_F150W', np.nan)
                sigma_v_F277W[idx] = result.get('sigma_v_F277W', np.nan)
                completed_mask[idx] = True

                done += 1
                if done % 10 == 0 or done == N:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (N - done) / rate if rate > 0 else 0
                    print(f'  [{done}/{N}] {elapsed:.0f}s elapsed, {rate:.1f} img/s, ETA {eta/60:.0f}m', flush=True)
    except KeyboardInterrupt:
        print('  Caught Ctrl+C — saving partial results')

    total = time.time() - t0
    n_kept = int(completed_mask.sum())
    print(f'\nDone — {n_kept}/{N} images completed in {total:.1f}s ({total/max(n_kept,1):.2f}s/image)')
    # Keep arrays at full N — write a completed_mask so callers know which entries are valid.
    np.save(str(OUT_DIR / 'completed_mask.npy'), completed_mask)

    # Print parameter summary
    lensed_mask = labels == 1
    if lensed_mask.any():
        print(f'\nParameter summary (lensed only):')
        print(f'  z_lens:   {z_lenses[lensed_mask].min():.2f} - {z_lenses[lensed_mask].max():.2f}  (mean {z_lenses[lensed_mask].mean():.2f})')
        print(f'  z_source: {z_sources[lensed_mask].min():.2f} - {z_sources[lensed_mask].max():.2f}  (mean {z_sources[lensed_mask].mean():.2f})')
        print(f'  theta_E:  {theta_Es[lensed_mask].min():.2f} - {theta_Es[lensed_mask].max():.2f}  (mean {theta_Es[lensed_mask].mean():.2f})')

    # Image arrays are already memmapped to disk; just flush + close.
    print(f'\nFlushing memmapped image arrays to disk ({OUT_DIR}/)')
    for band in BANDS:
        all_images[band].flush(); del all_images[band]
        all_arcs[band].flush();   del all_arcs[band]
        all_galaxies[band].flush(); del all_galaxies[band]
        sz = N * IMAGE_SIZE * IMAGE_SIZE * 4 / 1e6
        print(f'  images_{band}.npy  {sz:.1f} MB')

    np.save(str(OUT_DIR / 'lensed.npy'), labels)
    np.save(str(OUT_DIR / 'theta_Es.npy'), theta_Es)
    np.save(str(OUT_DIR / 'z_lens.npy'), z_lenses)
    np.save(str(OUT_DIR / 'z_source.npy'), z_sources)
    np.save(str(OUT_DIR / 'masses.npy'), masses)
    # v9: audit trail — what σ_v we derived from each cutout's photometry
    np.save(str(OUT_DIR / 'sigma_v.npy'), sigma_vs)
    np.save(str(OUT_DIR / 'photom_F115W.npy'), photom_F115W)
    np.save(str(OUT_DIR / 'photom_F150W.npy'), photom_F150W)
    np.save(str(OUT_DIR / 'photom_F277W.npy'), photom_F277W)
    np.save(str(OUT_DIR / 'sigma_v_F115W.npy'), sigma_v_F115W)
    np.save(str(OUT_DIR / 'sigma_v_F150W.npy'), sigma_v_F150W)
    np.save(str(OUT_DIR / 'sigma_v_F277W.npy'), sigma_v_F277W)

    meta = {
        'version': VERSION,
        'n_images': N,
        'image_size': IMAGE_SIZE,
        'pixel_scale': PIXEL_SCALE,
        'fov_arcsec': IMAGE_SIZE * PIXEL_SCALE,
        'bands': BANDS,
        'params': {
            'z_lens': (f'Per-cutout photo-z from prepped_scenes_v9 manifest, clipped to '
                       f'[{ZLENS_LO}, {ZLENS_HI}]'
                       if 'v9' in SCENES_SOURCE else
                       f'TruncNorm({ZLENS_LOC}, {ZLENS_SCALE}, [{ZLENS_LO}, {ZLENS_HI}])'),
            'scenes_source': SCENES_SOURCE,
            'z_source': f'TruncNorm({ZSRC_LOC}, {ZSRC_SCALE}, [{ZSRC_LO}, {ZSRC_HI}])',
            'sigma_v': f'Photometry-derived per cutout (FJ slope=-10, intercepts in fj_params.json); fallback range [{SIGMA_LO}, {SIGMA_HI}] km/s',
            'theta_E_filter': f'[{THETA_E_LO}, {THETA_E_HI}] arcsec',
            'sigma_consistency_max_dex': SIGMA_CONSISTENCY_MAX_DEX,
            'photom_aperture_px': PHOTOM_APERTURE_PX,
            'photom_sky_annulus': list(PHOTOM_SKY_ANNULUS),
            'calibration_params_path': str(V9_ROOT / 'fj_params.json'),
            'log_mass': f'U({LOGMASS_LO}, {LOGMASS_HI})',
        },
    }
    with open(OUT_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Preview
    make_preview(all_images, all_arcs, labels, theta_Es, z_lenses, z_sources, N)


if __name__ == '__main__':
    main()
