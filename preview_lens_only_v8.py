"""
preview_lens_only_v8.py — render just lens galaxy + real background (no source,
no lensing). Diagnostic to validate that part of the pipeline in isolation.

For each of N systems:
  1. sample a random lens stamp from lenses_v8
  2. place it at image center via INTERPOL (same pipeline path as simulate_v8)
  3. scale to the standard lens amplitude (same peak-matching as simulate_v8)
  4. add Poisson + read noise
  5. add a random real background cutout

Output: output/v8/lens_only_preview.png (4 bands + RGB, 5 systems)
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.ndimage import gaussian_filter, rotate

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel


BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
IMAGE_SIZE = 630
PIXEL_SCALE = 0.03
PREPPED_DIR = Path('prepped_mosaic_630')
OUT_DIR = Path('output/v8')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sum_to_flux = 6.501853565914121
gain = 2.05
N = 5
SEED = 17

with open(PREPPED_DIR / 'band_info.json') as f:
    all_band_info = json.load(f)

BAND_CONFIG, backgrounds, psf_kernels = {}, {}, {}
for band in BANDS:
    info = all_band_info[band]
    mjysr_to_sim = info['pixar_sr'] * 1e15 * sum_to_flux
    sim_to_elec = (1.0 / mjysr_to_sim) / info['photmjsr'] * info['xposure'] * gain
    BAND_CONFIG[band] = {'mjysr_to_sim': mjysr_to_sim, 'sim_to_elec': sim_to_elec}
    psf = np.load(str(PREPPED_DIR / band / 'psf_median.npy')).astype(np.float64)
    psf = np.clip(psf, 0, None); psf /= psf.sum()
    psf_kernels[band] = psf
    bg_raw = np.load(str(PREPPED_DIR / band / 'backgrounds.npy'))
    backgrounds[band] = (np.nan_to_num(bg_raw) * mjysr_to_sim).astype(np.float32)

lens_stamps = {b: np.load(str(PREPPED_DIR / 'lenses_v8' / f'stamps_{b}.npy')) for b in BANDS}
with open(PREPPED_DIR / 'lenses_v8' / 'lens_info.json') as f:
    lens_info = json.load(f)
LENS_STAMP_SCALE = lens_info.get('pixel_scale', 0.03)
print(f'Lenses: {lens_stamps[BANDS[0]].shape}, pixel_scale={LENS_STAMP_SCALE}')


def make_kwargs_data():
    ra_at_xy0 = -(IMAGE_SIZE - 1) / 2 * PIXEL_SCALE
    dec_at_xy0 = -(IMAGE_SIZE - 1) / 2 * PIXEL_SCALE
    return {
        'image_data': np.zeros((IMAGE_SIZE, IMAGE_SIZE)),
        'ra_at_xy_0': ra_at_xy0, 'dec_at_xy_0': dec_at_xy0,
        'transform_pix2angle': np.array([[PIXEL_SCALE, 0], [0, PIXEL_SCALE]]),
        'background_rms': 0.1, 'exposure_time': 1.0,
    }


def make_delta_psf():
    kernel = np.zeros((5, 5)); kernel[2, 2] = 1.0
    return PSF(psf_type='PIXEL', kernel_point_source=kernel)


def augment_stamp(stamp, angle, do_flip):
    """Absolute-calibration rotate/flip. No clip, no normalize — preserves
    natural sky noise around the galaxy."""
    out = rotate(stamp, angle, reshape=False, order=1, mode='constant', cval=0.0)
    if do_flip:
        out = np.ascontiguousarray(np.fliplr(out))
    return out


def add_poisson_noise(img_sim, band, rng):
    conv = BAND_CONFIG[band]['sim_to_elec']
    elec = np.clip(img_sim * conv, 0, None)
    noisy = rng.poisson(elec).astype(np.float32) / conv
    return noisy + rng.normal(0, 0.5, size=img_sim.shape).astype(np.float32) / conv


def render_lens_only(seed):
    rng = np.random.default_rng(seed)
    n_lens = len(lens_stamps[BANDS[0]])
    lens_idx = int(rng.integers(n_lens))
    angle = float(rng.uniform(0, 360))
    flip = bool(rng.random() > 0.5)
    bg_idx = int(rng.integers(len(backgrounds[BANDS[0]])))

    out = {}
    for band in BANDS:
        data_class = ImageData(**make_kwargs_data())
        psf_delta = make_delta_psf()
        stamp = augment_stamp(lens_stamps[band][lens_idx].copy(), angle, flip)
        light = LightModel(['INTERPOL'])
        kw_ll = [{'image': stamp.astype(np.float64), 'amp': 1.0,
                  'center_x': 0., 'center_y': 0., 'phi_G': 0.0,
                  'scale': LENS_STAMP_SCALE}]
        # Trivial lens model (not used since source_add=False)
        lens_model = LensModel(['SIE', 'SHEAR'])
        dummy_kwargs_lens = [
            {'theta_E': 1.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0., 'center_y': 0.},
            {'gamma1': 0.0, 'gamma2': 0.0},
        ]
        source_light = LightModel(['SERSIC_ELLIPSE'])
        kw_src = [{'amp': 0.0, 'R_sersic': 0.1, 'n_sersic': 1.0,
                   'e1': 0.0, 'e2': 0.0, 'center_x': 0., 'center_y': 0.}]
        im = ImageModel(data_class=data_class, psf_class=psf_delta,
                         lens_model_class=lens_model,
                         source_model_class=source_light,
                         lens_light_model_class=light,
                         kwargs_numerics={'supersampling_factor': 1,
                                          'supersampling_convolution': False,
                                          'compute_mode': 'regular'})
        # Absolute-calibration v8 stamps: amp=1 preserves natural brightness.
        img = im.image(dummy_kwargs_lens, kw_src,
                        kwargs_lens_light=kw_ll,
                        source_add=False, lens_light_add=True)
        rng_noise = np.random.default_rng(seed * 13 + hash(band) % 1000)
        noisy = add_poisson_noise(img, band, rng_noise)
        bg = backgrounds[band][bg_idx]
        final = (noisy + bg).astype(np.float32)
        out[band] = {'lens_only': img.astype(np.float32),
                     'lens_noisy': noisy,
                     'background': bg,
                     'final': final}
    return {'lens_idx': lens_idx, 'bg_idx': bg_idx, 'bands': out}


def make_rgb(r, g, b, gamma=0.5, smooth=0.8):
    r = gaussian_filter(np.nan_to_num(r).astype(np.float64), smooth)
    g = gaussian_filter(np.nan_to_num(g).astype(np.float64), smooth)
    b = gaussian_filter(np.nan_to_num(b).astype(np.float64), smooth)
    out = np.zeros((*r.shape, 3))
    for i, ch in enumerate([r, g, b]):
        bg = np.percentile(ch, 30); ch = ch - bg; ch[ch < 0] = 0
        vlo = np.percentile(ch, 1); vhi = np.percentile(ch, 99.8)
        if vhi <= vlo: vhi = vlo + 1
        out[:, :, i] = np.clip((ch - vlo) / (vhi - vlo), 0, 1)
    out = np.power(out, gamma)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for i in range(N):
        seed = int(rng.integers(int(1e9)))
        r = render_lens_only(seed)
        if r is not None:
            results.append(r)
        print(f'  rendered {len(results)}/{N}')

    fig, axs = plt.subplots(5, len(results), figsize=(4 * len(results), 20), dpi=100)
    if len(results) == 1:
        axs = axs[:, np.newaxis]
    for col, r in enumerate(results):
        for row, band in enumerate(BANDS):
            img = r['bands'][band]['final']
            d = np.nan_to_num(np.clip(img, 0, None))
            vmin = float(np.percentile(d, 0.5))
            vmax = float(np.percentile(d, 99.9))
            lw = max((vmax - vmin) * 0.001, 1e-6)
            norm = AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)
            axs[row, col].imshow(d, norm=norm, origin='lower', cmap='gray')
            axs[row, col].set_title(
                f'{band} lens#{r["lens_idx"]} bg#{r["bg_idx"]}', fontsize=8)
            axs[row, col].axis('off')
        rgb = make_rgb(r['bands']['F444W']['final'],
                        r['bands']['F277W']['final'],
                        r['bands']['F150W']['final'])
        axs[4, col].imshow(rgb, origin='lower')
        axs[4, col].set_title('RGB (444/277/150)', fontsize=8)
        axs[4, col].axis('off')
    for row, band in enumerate(BANDS):
        axs[row, 0].set_ylabel(band, fontsize=10)
    axs[4, 0].set_ylabel('RGB', fontsize=10)
    plt.suptitle(f'v8 lens-only diagnostic — {len(results)} systems (lens light + real bg + noise, no source, no lensing)', fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / 'lens_only_preview.png'
    plt.savefig(str(path), dpi=100, bbox_inches='tight')
    print(f'Saved -> {path}')


if __name__ == '__main__':
    main()
