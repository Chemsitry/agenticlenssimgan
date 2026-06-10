"""
validate_f115w.py — Multi-band validation against a real COWLS II lens.

Extracts real cutouts from the DR0.5 mosaics in all 4 NIRCam bands,
simulates the same system with matching parameters, and produces a
side-by-side comparison figure per band.

Default target: Lens E (COSJ095950+022057) from COWLS II (Mahler et al. 2025)
  theta_E = 1.04", z_lens = 0.939, z_source ~ 2.5 (assumed)

Usage:
    .venv/bin/python3 validate_f115w.py
    .venv/bin/python3 validate_f115w.py --seed 123
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from astropy.io import fits
from astropy.wcs import WCS

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

# ── Configuration ────────────────────────────────────────────────────────

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
IMAGE_SIZE = 630
PIXEL_SCALE = 0.03  # arcsec/pix  (630 * 0.03 = 18.9" FoV, matches COWLS II)
PREPPED_DIR = Path('prepped_mosaic_630')
MOSAIC_DIR = Path('raw_data/1727_mosaic')
OUT_DIR = Path('output/validation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sum_to_flux = 6.501853565914121
gain = 2.05

# Lens E preset (COWLS II Table 1)
LENS_E = {
    'name': 'COSJ095950+022057',
    'label': 'E',
    'ra': 149.961430320,
    'dec': 2.349411544,
    'theta_E': 1.04,
    'z_lens': 0.939,
    'z_source': 2.5,  # assumed (not measured)
    'log_mstar': 11.150,
}

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
        'pixar_sr': pixar_sr,
        'photmjsr': photmjsr,
        'xposure': xposure,
        'bg_median': info['bg_median'],
        'mjysr_to_sim': mjysr_to_sim,
        'sim_to_elec': sim_to_elec,
    }

    # PSF
    psf = np.load(str(PREPPED_DIR / band / 'psf_median.npy')).astype(np.float64)
    psf = np.clip(psf, 0, None)
    psf /= psf.sum()
    psf_kernels[band] = psf

    # Backgrounds
    bg_raw = np.load(str(PREPPED_DIR / band / 'backgrounds.npy'))
    bg_raw = np.nan_to_num(bg_raw, nan=0.0)
    backgrounds[band] = (bg_raw * mjysr_to_sim).astype(np.float32)

    print(f"  {band}: mjysr_to_sim={mjysr_to_sim:.2f}  sim_to_elec={sim_to_elec:.2f}  "
          f"bg_median={info['bg_median']:.6f}  PSF={psf.shape}")


# ── Helper functions (from simulate_v3.py) ───────────────────────────────

def elliptical_color_ratios(z_lens):
    f150w = 1.3 + 0.4 * z_lens
    f277w = 1.8 + 1.2 * z_lens
    f444w = 2.0 + 1.8 * z_lens
    return {'F115W': 1.0, 'F150W': f150w, 'F277W': f277w, 'F444W': f444w}


def starforming_color_ratios(z_source, uv_slope=0.5):
    ly_break_um = 0.1216 * (1 + z_source)
    band_waves = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}
    ratios = {}
    for band, lam_eff in band_waves.items():
        if lam_eff < ly_break_um:
            suppression = max(0.0, (lam_eff / ly_break_um) ** 3)
            ratio = suppression * 0.05
        else:
            ratio = (1.15 / lam_eff) ** uv_slope
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
    mM10 = 11.88; mu = 0.019; mM00 = 0.0282; nu = -0.72
    gamma0 = 0.556; gamma1 = -0.26; beta0 = 1.06; beta1 = 0.17
    M1 = 10**(11.88 * (z + 1)**mu)
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


# ── Step 1: Extract real cutouts (all bands) ─────────────────────────────

def extract_real_cutouts(ra, dec):
    """Extract 224x224 cutouts from DR0.5 mosaics at given RA/Dec for all bands."""
    import warnings
    warnings.filterwarnings('ignore', message='.*datfix.*')
    warnings.filterwarnings('ignore', message='.*obsfix.*')

    cutouts = {}
    for band in BANDS:
        fits_files = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
        if not fits_files:
            print(f"  {band}: ERROR — no mosaic FITS file found")
            cutouts[band] = None
            continue

        hdul = fits.open(str(fits_files[0]), memmap=True)
        wcs = WCS(hdul[1].header)
        sci = hdul[1].data
        ny, nx = sci.shape

        px, py = wcs.all_world2pix(ra, dec, 0)
        px, py = int(np.round(px)), int(np.round(py))
        half = IMAGE_SIZE // 2

        if not (half <= px < nx - half and half <= py < ny - half):
            print(f"  {band}: outside mosaic bounds (px={px}, py={py})")
            hdul.close()
            cutouts[band] = None
            continue

        cutout = sci[py - half:py + half, px - half:px + half].copy()
        hdul.close()

        valid = np.isfinite(cutout) & (cutout != 0)
        valid_frac = valid.mean()

        bg_med = BAND_CONFIG[band]['bg_median']
        mts = BAND_CONFIG[band]['mjysr_to_sim']
        cutout_clean = np.where(valid, cutout - bg_med, 0.0)
        cutout_sim = (cutout_clean * mts).astype(np.float32)

        print(f"  {band}: valid={valid_frac:.3f}  range=[{cutout_sim.min():.1f}, {cutout_sim.max():.1f}] sim")
        cutouts[band] = cutout_sim

    return cutouts


# ── Step 2: Simulate all bands ───────────────────────────────────────────

def simulate_multiband(params, real_cutouts=None, seed=42):
    """Simulate a lens system in all 4 NIRCam bands."""
    rng = np.random.default_rng(seed)

    z_lens = params['z_lens']
    z_source = params['z_source']
    theta_E = params['theta_E']
    log_mstar = params['log_mstar']

    # Shared lens mass model
    e1, e2 = 0.05, 0.02
    gamma1, gamma2 = 0.02, 0.01

    lens_model = LensModel(['SIE', 'SHEAR'], z_lens=z_lens, z_source=z_source)
    kwargs_lens = [
        {'theta_E': theta_E, 'e1': e1, 'e2': e2,
         'center_x': 0., 'center_y': 0.},
        {'gamma1': gamma1, 'gamma2': gamma2}
    ]

    # Shared source position
    src_offset = 0.3 * theta_E
    src_angle = float(rng.uniform(0, 2 * np.pi))
    center_x = src_offset * np.cos(src_angle)
    center_y = src_offset * np.sin(src_angle)
    e1s, e2s = rng.normal(0, 0.2, size=2).clip(-0.6, 0.6)

    # SED color ratios
    lens_colors = elliptical_color_ratios(z_lens)
    uv_slope = float(rng.normal(-0.1, 0.7))
    src_colors = starforming_color_ratios(z_source, uv_slope)

    print(f"\n  Lens SED: {', '.join(f'{b}={v:.2f}' for b, v in lens_colors.items())}")
    print(f"  Src  SED: {', '.join(f'{b}={v:.2f}' for b, v in src_colors.items())} (uv_slope={uv_slope:.2f})")

    # Light model templates (amp=1, calibrated per band below)
    kwargs_lens_light = [{
        'amp': 1, 'R_sersic': 0.4, 'n_sersic': 4,
        'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0.
    }]
    kwargs_source = [{
        'amp': 1, 'R_sersic': 0.15, 'n_sersic': 1.5,
        'e1': float(e1s), 'e2': float(e2s),
        'center_x': float(center_x), 'center_y': float(center_y)
    }]

    # Render all bands — calibrate lens brightness per band against real data
    bg_idx = int(rng.integers(len(backgrounds['F115W'])))
    results = {}
    target_ratio = 0.25

    for band in BANDS:
        psf_class_b = make_psf_obj(band)
        data_class_b = ImageData(**make_kwargs_data())
        lens_light_b = LightModel(['SERSIC_ELLIPSE'])
        source_b = LightModel(['SERSIC_ELLIPSE'])

        im = ImageModel(
            data_class=data_class_b, psf_class=psf_class_b,
            lens_model_class=lens_model,
            source_model_class=source_b,
            lens_light_model_class=lens_light_b,
            kwargs_numerics=KWARGS_NUMERICS)

        # Calibrate lens amp: render at amp=1, measure peak, scale to match real
        kw_ll_cal = [{**kwargs_lens_light[0], 'amp': 1.0}]
        kw_src_cal = [{**kwargs_source[0], 'amp': 1.0}]

        img_lens_unit = im.image(kwargs_lens, kw_src_cal,
                                  kwargs_lens_light=kw_ll_cal,
                                  source_add=False, lens_light_add=True)
        peak_unit = float(np.max(img_lens_unit))

        if (real_cutouts is not None and real_cutouts.get(band) is not None
                and peak_unit > 0):
            real_peak = float(np.max(real_cutouts[band]))
            amp_lens = real_peak / peak_unit
        else:
            mStar = 10**log_mstar
            lStar = mStar / ML_ratio(max(z_lens, 0.01))
            amp_lens = (sum_to_flux * lStar * lens_colors[band]) / np.sum(img_lens_unit)

        # Calibrate source amp via arc/lens ratio
        img_src_unit = im.image(kwargs_lens, kw_src_cal,
                                 kwargs_lens_light=kw_ll_cal,
                                 source_add=True, lens_light_add=False)
        sum_src_unit = float(np.sum(img_src_unit))
        sum_lens = float(np.sum(img_lens_unit)) * amp_lens
        amp_src = (sum_lens * target_ratio / sum_src_unit) if sum_src_unit > 0 else 1.0

        # Final render with calibrated amps
        kw_ll = [{**kwargs_lens_light[0], 'amp': amp_lens}]
        kw_src = [{**kwargs_source[0], 'amp': amp_src}]

        image_full = im.image(kwargs_lens, kw_src,
                               kwargs_lens_light=kw_ll,
                               source_add=True, lens_light_add=True)
        image_arcs = im.image(kwargs_lens, kw_src,
                               kwargs_lens_light=kw_ll,
                               source_add=True, lens_light_add=False)

        image_noisy = add_poisson_noise(image_full, band, rng=rng)
        bg = backgrounds[band][bg_idx]
        image_final = (image_noisy + bg).astype(np.float32)

        results[band] = {
            'final': image_final,
            'arcs': image_arcs.astype(np.float32),
        }
        # Report center peak (lens), not global max (could be a bright bg galaxy)
        ctr = IMAGE_SIZE // 2
        sim_center = float(image_full[ctr, ctr])
        real_center = float(real_cutouts[band][ctr, ctr]) if real_cutouts and real_cutouts.get(band) is not None else 0
        print(f"  {band}: lens_peak={sim_center:.1f}  real_peak={real_center:.1f}  arc_peak={image_arcs.max():.1f}")

    return results


# ── RGB rendering ────────────────────────────────────────────────────────

def make_rgb(r, g, b, smooth=3.5, gamma=0.30, lum_gate=0.08):
    """Percentile+gamma RGB composite tuned to match COWLS II rendering.

    Channels should be in MJy/sr (or any consistent surface brightness unit).
    Warm channel weighting (R*1.5, B*0.5) should be applied before calling.
    """
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
    # Luminance-based noise gate (push low-signal pixels to black)
    if lum_gate > 0:
        lum = np.max(out, axis=2)
        gate = np.clip((lum - lum_gate) / lum_gate, 0, 1)
        for i in range(3):
            out[:, :, i] *= gate
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def build_rgb(img_dict):
    """Build RGB from 4-band dict. R=F444W, G=avg(F277W,F150W), B=F115W.
    Warm weighting to match COWLS II golden tone."""
    norm = {band: BAND_CONFIG[band]['mjysr_to_sim'] for band in BANDS}
    r = img_dict['F444W'] / norm['F444W'] * 1.5
    g = 0.5 * (img_dict['F277W'] / norm['F277W'] +
                img_dict['F150W'] / norm['F150W'])
    b = img_dict['F115W'] / norm['F115W'] * 0.5
    return make_rgb(r, g, b)


# ── Step 3: Comparison figure ────────────────────────────────────────────

def make_comparison(real_cutouts, sim_results, params, save_path):
    """5-row comparison: 4 bands + RGB row."""

    fig, axs = plt.subplots(5, 3, figsize=(14, 22), dpi=150)

    for i, band in enumerate(BANDS):
        real = real_cutouts.get(band)
        sim_final = sim_results[band]['final']
        sim_arcs = sim_results[band]['arcs']

        # Matched stretch from real data
        if real is not None:
            ref = np.nan_to_num(np.clip(real, 0, None), nan=0.0)
        else:
            ref = np.nan_to_num(np.clip(sim_final, 0, None), nan=0.0)

        vmin = float(np.percentile(ref, 0.5))
        vmax = float(np.percentile(ref, 99.9))
        if vmax <= vmin:
            vmax = vmin + 1.0
        lw = max((vmax - vmin) * 0.001, 1e-6)
        norm = AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)

        def show(ax, img, title, use_norm=True, cmap='gray'):
            d = np.nan_to_num(np.clip(img, 0, None), nan=0.0)
            if use_norm:
                im = ax.imshow(d, norm=norm, origin='lower', cmap=cmap)
            else:
                v1 = float(np.percentile(d, 0.5))
                v2 = float(np.percentile(d, 99.9))
                if v2 <= v1:
                    v2 = v1 + 1.0
                lw2 = max((v2 - v1) * 0.01, 1e-6)
                n2 = AsinhNorm(linear_width=lw2, vmin=v1, vmax=v2)
                im = ax.imshow(d, norm=n2, origin='lower', cmap=cmap)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # Col 0: Real
        if real is not None:
            show(axs[i, 0], real, f"Real {band}")
        else:
            axs[i, 0].text(0.5, 0.5, f'No data', ha='center', va='center',
                           transform=axs[i, 0].transAxes, fontsize=12)
            axs[i, 0].set_title(f"Real {band}")
            axs[i, 0].axis('off')

        # Col 1: Sim full
        show(axs[i, 1], sim_final, f"Sim {band}")

        # Col 2: Sim arcs
        show(axs[i, 2], sim_arcs, f"Sim arcs {band}", use_norm=False, cmap='inferno')

    # Row 4: RGB composites
    def add_annotations(ax, pixel_scale=PIXEL_SCALE, img_size=IMAGE_SIZE,
                        bar_arcsec=5.0, is_paper=False):
        """Add scale bar and N/E compass to an RGB panel."""
        bar_px = bar_arcsec / pixel_scale
        # Scale bar — bottom-left
        margin = img_size * 0.06
        y_bar = margin
        x_bar_start = margin
        x_bar_end = margin + bar_px
        ax.plot([x_bar_start, x_bar_end], [y_bar, y_bar],
                color='white', linewidth=2.5, solid_capstyle='butt')
        ax.text((x_bar_start + x_bar_end) / 2, y_bar + img_size * 0.03,
                f'{bar_arcsec:.0f}"', color='white', fontsize=9,
                ha='center', va='bottom', fontweight='bold')

        if not is_paper:
            # N/E compass — top-left
            arrow_len = img_size * 0.08
            cx = margin + arrow_len
            cy = img_size - margin - arrow_len
            # N arrow (up)
            ax.annotate('', xy=(cx, cy + arrow_len), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
            ax.text(cx, cy + arrow_len + img_size * 0.02, 'N',
                    color='white', fontsize=8, ha='center', va='bottom', fontweight='bold')
            # E arrow (left, since RA increases to the left in standard orientation)
            ax.annotate('', xy=(cx - arrow_len, cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
            ax.text(cx - arrow_len - img_size * 0.02, cy, 'E',
                    color='white', fontsize=8, ha='right', va='center', fontweight='bold')

    # Real RGB: use the COWLS II paper figure (expert rendering)
    cowls_panel_path = OUT_DIR / 'cowls2_lens_e.png'
    if cowls_panel_path.exists():
        from PIL import Image
        cowls_rgb = np.array(Image.open(str(cowls_panel_path)))
        axs[4, 0].imshow(cowls_rgb)
        axs[4, 0].set_title("Real RGB (COWLS II, Mahler+2025)", fontsize=10)
    else:
        real_rgb = build_rgb(real_cutouts)
        axs[4, 0].imshow(real_rgb, origin='lower')
        axs[4, 0].set_title("Real RGB (from mosaic)", fontsize=10)
        add_annotations(axs[4, 0])
    axs[4, 0].axis('off')

    sim_rgb = build_rgb({band: sim_results[band]['final'] for band in BANDS})
    axs[4, 1].imshow(sim_rgb, origin='lower')
    axs[4, 1].set_title("Sim RGB", fontsize=10)
    axs[4, 1].axis('off')

    sim_arcs_rgb = build_rgb({band: sim_results[band]['arcs'] for band in BANDS})
    axs[4, 2].imshow(sim_arcs_rgb, origin='lower')
    axs[4, 2].set_title("Sim arcs RGB", fontsize=10)
    axs[4, 2].axis('off')

    plt.suptitle(
        f"Multi-band Validation: {params['name']} (Lens {params['label']})\n"
        f"$\\theta_E$ = {params['theta_E']}\"  "
        f"$z_{{lens}}$ = {params['z_lens']}  "
        f"$z_{{source}}$ = {params['z_source']} (assumed)",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved -> {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-band validation against COWLS II lens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    params = LENS_E

    print(f"\n{'='*60}")
    print(f"Multi-band Validation: {params['name']} (Lens {params['label']})")
    print(f"  theta_E = {params['theta_E']}\", z_lens = {params['z_lens']}, "
          f"z_source = {params['z_source']}")
    print(f"{'='*60}")

    # Step 1: Extract real cutouts
    print(f"\nStep 1: Extracting real cutouts from DR0.5 mosaics...")
    real_cutouts = extract_real_cutouts(params['ra'], params['dec'])

    # Step 2: Simulate all bands (calibrate lens brightness to real data)
    print(f"\nStep 2: Simulating all bands...")
    sim_results = simulate_multiband(params, real_cutouts=real_cutouts, seed=args.seed)

    # Step 3: Compare
    print(f"\nStep 3: Generating comparison figure...")
    save_path = OUT_DIR / 'validate_multiband.png'
    make_comparison(real_cutouts, sim_results, params, save_path)


if __name__ == '__main__':
    main()
