import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

# ── Paths ──────────────────────────────────────────────────────────────────
PREPPED_DIR   = 'prepped'
PSF_KERNEL_PATH = os.path.join(PREPPED_DIR, 'psf_median.npy')
OUT_DIR       = 'output/v2'

# ── Image config ──────────────────────────────────────────────────────────
PIXELS     = 125
PIXEL_SIZE = 0.031        # arcsec/pix — NIRCam SW channel
EXP_TIME   = 1380

# ── Real JWST backgrounds (F115W, jw01810) ────────────────────────────────
sum_to_flux  = 6.501853565914121
PIXAR_SR     = 2.29232933396454e-14
mjysr_to_sim = PIXAR_SR * 1e15 * sum_to_flux   # ≈ 149.0

real_bgs     = np.load(os.path.join(PREPPED_DIR, 'real_backgrounds.npy'))
real_bgs_sim = (real_bgs * mjysr_to_sim).astype(np.float32)
print(f'Loaded {len(real_bgs_sim)} real JWST background patches.')
print(f'Unit scale factor: {mjysr_to_sim:.2f}  (MJy/sr -> sim units)')

rng_bg = np.random.default_rng()

def get_real_background():
    return real_bgs_sim[rng_bg.integers(len(real_bgs_sim))]

print('Ready.')

# ── Empirical PSF ──────────────────────────────────────────────────────────
# Use the median stack of 174 clean PSF star stamps (63x63 px, 0.031"/px).
# Run `python build_psf.py` first to generate prepped/psf_median.npy.

def make_psf(kernel_path=PSF_KERNEL_PATH):
    """Load empirical PSF kernel and return lenstronomy PSF object."""
    kernel = np.load(kernel_path).astype(np.float64)
    kernel = np.clip(kernel, 0, None)  # remove tiny negatives from median stack
    kernel /= kernel.sum()   # ensure exact normalization
    return PSF(psf_type='PIXEL',
               kernel_point_source=kernel,
               kernel_point_source_normalisation=True)

# Verify PSF loads correctly
_test_psf = make_psf()
print(f'PSF kernel shape : {_test_psf.kernel_point_source.shape}')
print(f'PSF kernel sum   : {_test_psf.kernel_point_source.sum():.8f}')
peak = np.unravel_index(np.argmax(_test_psf.kernel_point_source),
                        _test_psf.kernel_point_source.shape)
print(f'PSF peak pixel   : {peak}  (expected near (31, 31))')
del _test_psf

# ── Lenstronomy kwargs helpers ─────────────────────────────────────────────

def make_kwargs_data():
    return {
        'background_rms': 0,
        'exposure_time': EXP_TIME,
        'ra_at_xy_0':  -PIXELS/2 * PIXEL_SIZE,
        'dec_at_xy_0': -PIXELS/2 * PIXEL_SIZE,
        'transform_pix2angle': np.array([[PIXEL_SIZE, 0.], [0., PIXEL_SIZE]]),
        'image_data': np.zeros((PIXELS, PIXELS))
    }

# supersampling_factor=3 for accurate sub-pixel PSF convolution
KWARGS_NUMERICS = {
    'supersampling_factor': 3,
    'supersampling_convolution': True
}

print('PSF and numerics settings ready.')

# ── Mass / stellar-mass helpers (unchanged from original notebook) ──────────

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

print('Helpers defined.')

# ── simulate_one v2 ────────────────────────────────────────────────────────
#
# Key improvements:
#   1. Empirical PIXEL PSF with supersampling_factor=3
#   2. SLACS-calibrated parameter distributions
#   3. SIE + SHEAR lens model (asymmetric arcs)
#   4. Source offset relative to Einstein radius
#   5. Arc/lens flux ratio floor: arc >= 1e-2 * lens light

# Minimum arc-to-lens-light total flux ratio (prevents undetectable arcs).
# Applied as a post-calibration amplitude boost — no re-simulation needed.
ARC_LENS_MIN_RATIO = 1e-2

def simulate_one_v2(lensed=True, seed=None):
    rng = np.random.default_rng(seed)

    # ── SLACS-calibrated redshifts ─────────────────────────────────────────
    z_lens = float(truncnorm.rvs(
        a=(0.05 - 0.3) / 0.15,
        b=(0.90 - 0.3) / 0.15,
        loc=0.3, scale=0.15,
        random_state=int(rng.integers(int(1e9)))
    ))

    z_src_min = max(0.6, z_lens + 0.05)
    for _ in range(100):
        z_source = float(truncnorm.rvs(
            a=(0.6 - 1.5) / 0.8,
            b=(3.0 - 1.5) / 0.8,
            loc=1.5, scale=0.8,
            random_state=int(rng.integers(int(1e9)))
        ))
        if z_source > z_lens + 0.05:
            break
    z_source = max(z_source, z_src_min)

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

    # ── Halo mass: galaxy-scale (SLACS ellipticals are NOT clusters) ───────
    # Real SLACS lenses: individual massive ellipticals, log M ~ 11.5–13.0
    # The old range (13.0–14.5) was cluster-scale → lens light drowned arcs.
    log_mass = float(rng.uniform(11.5, 13.0))
    mass     = 10**log_mass
    mStar    = stellar_mass(mass, z_lens)

    # ── Galaxy shape parameters ────────────────────────────────────────────
    e1, e2        = rng.normal(0, 0.15, size=2).clip(-0.5, 0.5)
    R_sersic_lens = float(truncnorm.rvs(0, 3, loc=0.3, scale=0.3,
                                        random_state=int(rng.integers(int(1e9)))))
    n_sersic_lens = float(rng.uniform(2, 6))
    R_sersic_src  = float(truncnorm.rvs(0, 3, loc=0.15, scale=0.15,
                                        random_state=int(rng.integers(int(1e9)))))
    n_sersic_src  = float(rng.uniform(1, 4))
    e1s, e2s      = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    if lensed and theta_E > 0:
        src_offset = float(rng.uniform(0.0, 0.3 * theta_E))
        src_angle  = float(rng.uniform(0, 2 * np.pi))
        center_x   = src_offset * np.cos(src_angle)
        center_y   = src_offset * np.sin(src_angle)
    else:
        center_x, center_y = rng.normal(0, 0.25, size=2)

    gamma_ext = float(rng.uniform(0.0, 0.08))
    psi_ext   = float(rng.uniform(0, np.pi))
    gamma1    = gamma_ext * np.cos(2 * psi_ext)
    gamma2    = gamma_ext * np.sin(2 * psi_ext)

    # ── Lenstronomy setup ──────────────────────────────────────────────────
    kwargs_data     = make_kwargs_data()
    data_class      = ImageData(**kwargs_data)
    psf_class       = make_psf()

    source_model_class     = LightModel(['SERSIC_ELLIPSE'])
    lens_light_model_class = LightModel(['SERSIC_ELLIPSE'])
    lens_model_class       = LensModel(['SIE', 'SHEAR'],
                                        z_lens=z_lens, z_source=z_source)

    image_model = ImageModel(
        data_class=data_class, psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=KWARGS_NUMERICS
    )

    kwargs_lens = [
        {'theta_E': theta_E, 'e1': float(e1), 'e2': float(e2),
         'center_x': 0., 'center_y': 0.},
        {'gamma1': gamma1, 'gamma2': gamma2}
    ]
    kwargs_lens_light = [{
        'amp': 1, 'R_sersic': R_sersic_lens,
        'n_sersic': n_sersic_lens, 'e1': float(e1), 'e2': float(e2),
        'center_x': 0., 'center_y': 0.
    }]
    kwargs_source = [{
        'amp': 1, 'R_sersic': R_sersic_src,
        'n_sersic': n_sersic_src, 'e1': float(e1s), 'e2': float(e2s),
        'center_x': float(center_x), 'center_y': float(center_y)
    }]

    # ── Calibrate source amplitude ─────────────────────────────────────────
    scale_up     = 10**float(rng.uniform(0, 2))
    src_flux_njy = float(truncnorm.rvs(0, 3, loc=50, scale=80,
                                       random_state=int(rng.integers(int(1e9)))))
    calc_sum_src = sum_to_flux * src_flux_njy   # target total arc flux (sim units)

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
    calc_sum_lens = sum_to_flux * lStar   # target total lens flux (sim units)

    img_lens = image_model.image(kwargs_lens, kwargs_source,
                                 kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                                 source_add=False, lens_light_add=True)
    s_lens = np.sum(img_lens)
    if s_lens <= 0:
        return simulate_one_v2(lensed=lensed, seed=int(rng.integers(int(1e9))))
    kwargs_lens_light[0]['amp'] = calc_sum_lens / s_lens

    # ── Arc/lens brightness floor ──────────────────────────────────────────
    if lensed and calc_sum_lens > 0:
        arc_flux  = calc_sum_src * scale_up
        lens_flux = calc_sum_lens
        if arc_flux < ARC_LENS_MIN_RATIO * lens_flux:
            boost = (ARC_LENS_MIN_RATIO * lens_flux) / arc_flux
            kwargs_source[0]['amp'] *= boost

    # ── Final images ────────────────────────────────────────────────────────
    image = image_model.image(kwargs_lens, kwargs_source,
                              kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                              source_add=True, lens_light_add=True)

    image_source = image_model.image(kwargs_lens, kwargs_source,
                                     kwargs_lens_light=kwargs_lens_light, kwargs_ps=None,
                                     source_add=True, lens_light_add=False)

    return image, image_source, theta_E, z_lens, z_source, mass, mStar

print('simulate_one_v2() defined.')
print(f'Halo mass range: 10^11.5 - 10^13.0 M_sun (galaxy-scale, SLACS-appropriate)')
print(f'Arc/lens floor : {ARC_LENS_MIN_RATIO:.0e}')

# ── Quick verification: generate 10 test images ────────────────────────────
# Checks that arcs are visible against lens light, PSF is non-circular,
# and theta_E peaks around 0.6-0.9"

import time

N_TEST = 10
N_EACH_TEST = N_TEST // 2

test_images        = np.zeros((N_TEST, PIXELS, PIXELS), dtype=np.float32)
test_image_sources = np.zeros_like(test_images)
test_labels        = np.zeros(N_TEST)
test_theta_Es      = np.zeros(N_TEST)
test_z_lenses      = np.zeros(N_TEST)
test_z_sources     = np.zeros(N_TEST)
test_arc_ratios    = np.zeros(N_TEST)

rng_main = np.random.default_rng(99)
jobs = ([(i, False) for i in range(N_EACH_TEST)] +
        [(i + N_EACH_TEST, True) for i in range(N_EACH_TEST)])

t0 = time.time()
for idx, lensed in jobs:
    label = 'lensed' if lensed else 'non-lensed'
    print(f'  [{idx+1}/{N_TEST}] {label}...', end=' ', flush=True)
    t1 = time.time()
    result = simulate_one_v2(lensed=lensed, seed=int(rng_main.integers(int(1e9))))
    image, image_source, theta_E, z_lens, z_source, mass, mStar = result
    bg = get_real_background()
    test_images[idx]        = image + bg
    test_image_sources[idx] = image_source
    test_labels[idx]        = 1.0 if lensed else 0.0
    test_theta_Es[idx]      = theta_E
    test_z_lenses[idx]      = z_lens
    test_z_sources[idx]     = z_source
    # arc/lens ratio from the raw (no-BG) image
    lens_only = image - image_source
    arc_sum   = np.sum(np.clip(image_source, 0, None))
    lens_sum  = np.sum(np.clip(lens_only, 0, None))
    test_arc_ratios[idx] = arc_sum / max(lens_sum, 1e-30) if lensed else np.nan
    ratio_str = f'  arc/lens={test_arc_ratios[idx]:.3f}' if lensed else ''
    print(f'{time.time()-t1:.1f}s  θE={theta_E:.2f}" zl={z_lens:.2f} zs={z_source:.2f}{ratio_str}')

print(f'\nTest run: {time.time()-t0:.1f}s total')
lensed_mask = test_labels == 1
lensed_tE = test_theta_Es[lensed_mask]
print(f'theta_E range   : {lensed_tE.min():.3f} - {lensed_tE.max():.3f}"  mean={lensed_tE.mean():.3f}"')
valid_ratios = test_arc_ratios[lensed_mask & np.isfinite(test_arc_ratios)]
if len(valid_ratios):
    print(f'arc/lens ratio  : min={valid_ratios.min():.3f}  median={np.median(valid_ratios):.3f}  max={valid_ratios.max():.3f}')
    n_visible = (valid_ratios >= 0.01).sum()
    print(f'visible arcs    : {n_visible}/{len(valid_ratios)} with arc/lens >= 1%')

# ── Preview: verify arcs, PSF shape, parameter distributions ──────────────

os.makedirs(OUT_DIR, exist_ok=True)

order_nl = np.where(test_labels == 0)[0]
order_l  = np.where(test_labels == 1)[0]
n_show   = min(5, len(order_nl), len(order_l))

fig, axs = plt.subplots(3, n_show, figsize=(4*n_show, 13), dpi=100)

def show(ax, img, title, cmap='gray', sqrt=False):
    d = np.sqrt(np.clip(img, 0, None)) if sqrt else img
    vmin, vmax = np.percentile(d, 1), np.percentile(d, 99.5)
    im = ax.imshow(d, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=7)
    ax.axis('off')

for col in range(n_show):
    idx = order_nl[col]
    show(axs[0, col], test_images[idx],
         f'non-lensed zl={test_z_lenses[idx]:.2f}', sqrt=True)

for col in range(n_show):
    idx = order_l[col]
    tE = test_theta_Es[idx]
    show(axs[1, col], test_images[idx],
         f'lensed θE={tE:.2f}" zl={test_z_lenses[idx]:.2f} zs={test_z_sources[idx]:.2f}',
         sqrt=True)

for col in range(n_show):
    idx = order_l[col]
    show(axs[2, col], test_image_sources[idx],
         f'arcs only θE={test_theta_Es[idx]:.2f}"', cmap='inferno')

axs[0, 0].set_ylabel('Non-lensed\n(√ stretch)', fontsize=10)
axs[1, 0].set_ylabel('Lensed\n(√ stretch)', fontsize=10)
axs[2, 0].set_ylabel('Arc-only\n(linear)', fontsize=10)

plt.suptitle('simulate_v2: empirical PSF + SLACS params + SIE+SHEAR', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/preview.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved -> {OUT_DIR}/preview.png')

# ── Generate 2,000 images (1,000 lensed + 1,000 non-lensed) ───────────────

N_EACH  = 1000
N_TOTAL = N_EACH * 2

images        = np.zeros((N_TOTAL, PIXELS, PIXELS), dtype=np.float32)
image_sources = np.zeros_like(images)
labels        = np.zeros(N_TOTAL)
theta_Es      = np.zeros(N_TOTAL)
z_lenses      = np.zeros(N_TOTAL)
z_sources     = np.zeros(N_TOTAL)
masses        = np.zeros(N_TOTAL)

rng_main = np.random.default_rng(42)
jobs = ([(i, False) for i in range(N_EACH)] +
        [(i + N_EACH, True) for i in range(N_EACH)])

t0 = time.time()
for idx, lensed in jobs:
    if idx % 100 == 0:
        elapsed = time.time() - t0
        print(f'  [{idx+1}/{N_TOTAL}]  elapsed={elapsed:.0f}s', flush=True)
    result = simulate_one_v2(lensed=lensed, seed=int(rng_main.integers(int(1e9))))
    image, image_source, theta_E, z_lens, z_source, mass, mStar = result
    bg = get_real_background()
    images[idx]        = image + bg
    image_sources[idx] = image_source + bg
    labels[idx]        = 1.0 if lensed else 0.0
    theta_Es[idx]      = theta_E
    z_lenses[idx]      = z_lens
    z_sources[idx]     = z_source
    masses[idx]        = mass

total = time.time() - t0
print(f'\nDone — {N_TOTAL} images in {total:.0f}s ({total/N_TOTAL:.2f}s/image)')
print(f'  Non-lensed: {int((labels==0).sum())}   Lensed: {int((labels==1).sum())}')

# ── Save outputs ───────────────────────────────────────────────────────────

np.save(f'{OUT_DIR}/images.npy',       images)
np.save(f'{OUT_DIR}/lensed.npy',       labels)
np.save(f'{OUT_DIR}/theta_Es.npy',     theta_Es)
np.save(f'{OUT_DIR}/z_lens.npy',       z_lenses)
np.save(f'{OUT_DIR}/z_source.npy',     z_sources)
np.save(f'{OUT_DIR}/masses.npy',       masses)
np.save(f'{OUT_DIR}/image_source.npy', image_sources)

print(f'Saved to {OUT_DIR}/')
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    if f.endswith('.npy'):
        print(f'  {f:<30} {os.path.getsize(path)/1e6:.1f} MB')

# ── Parameter distribution diagnostics ────────────────────────────────────

lensed_mask = labels == 1

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

axes[0].hist(theta_Es[lensed_mask], bins=30, color='steelblue', edgecolor='k')
axes[0].set_xlabel('Einstein radius θ_E (arcsec)')
axes[0].set_ylabel('Count')
axes[0].set_title('θ_E distribution (lensed only)')

axes[1].hist(z_lenses[lensed_mask], bins=30, color='coral', edgecolor='k', label='z_lens')
axes[1].hist(z_sources[lensed_mask], bins=30, color='green', alpha=0.6, edgecolor='k', label='z_source')
axes[1].set_xlabel('Redshift')
axes[1].set_title('Redshift distributions (lensed)')
axes[1].legend()

axes[2].hist(np.log10(masses[lensed_mask]), bins=30, color='purple', edgecolor='k')
axes[2].set_xlabel('log10(M_halo / M_sun)')
axes[2].set_title('Halo mass distribution (lensed)')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/param_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print(f'Saved -> {OUT_DIR}/param_distributions.png')

# ── Multi-band stub ────────────────────────────────────────────────────────
# Demonstrates how to load per-band backgrounds with their PIXAR_SR values
# once prep_multifield.py has been run.
#
# The simulation renders at F115W; other bands can be appended as additional
# channels once lenstronomy multi-band rendering is wired up.

import json
from pathlib import Path

PREPPED_V2 = Path('prepped_v2')
PIXAR_SR_JSON = PREPPED_V2 / 'pixar_sr.json'

if PIXAR_SR_JSON.exists():
    with open(PIXAR_SR_JSON) as f:
        pixar_sr_map = json.load(f)
    print('PIXAR_SR per band:')
    for band, val in pixar_sr_map.items():
        scale = val * 1e15 * sum_to_flux
        print(f'  {band}: PIXAR_SR={val:.4e} sr  ->  mjysr_to_sim={scale:.2f}')

    # Load F115W backgrounds from prepped_v2 if available
    f115w_v2 = PREPPED_V2 / 'jw01810' / 'F115W' / 'backgrounds.npy'
    if f115w_v2.exists():
        bgs_v2 = np.load(str(f115w_v2))
        print(f'  Loaded prepped_v2 F115W: {bgs_v2.shape}')
    else:
        print(f'  prepped_v2 F115W not found — run prep_multifield.py first.')
else:
    print('prepped_v2/pixar_sr.json not found — run prep_multifield.py first.')
    print('Using original prepped/ backgrounds for this run.')
