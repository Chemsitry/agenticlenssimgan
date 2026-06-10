"""
normalize.py — Fixed asinh normalization shared by all GAN scripts.

Why fixed (not per-image):
  The old GAN used per-image min-max, which destroyed physical flux, sky RMS,
  and color information — the discriminator couldn't use any of those signals.
  Here we compute statistics ONCE from the real cutouts and reuse them for every
  image (sim or real).  This way the GAN can learn that sim and real have
  different noise levels, sky colors, or brightness distributions.

The stretch: arcsinh((image - sky_med) / (k * sky_sigma))
  - Near zero (sky pixels): behaves like (image - sky_med) / (k * sky_sigma) — linear
  - Bright pixels: behaves like log — compresses dynamic range without clipping
  - Invertible: physical units = sky_med + k * sky_sigma * sinh(stretched)

Usage:
    from gan.data.normalize import compute_stats, normalize, save_stats, load_stats

    # Once, from real cutouts (dict: band -> (N, H, W) array in sky-subtracted MJy/sr):
    stats = compute_stats(real_cutouts_dict)
    save_stats('output/gan/real_cutouts/normalization.json', stats)

    # Then for any image:
    stats = load_stats('output/gan/real_cutouts/normalization.json')
    x_norm = normalize(image_array, 'F115W', stats)
"""

import json
import numpy as np
from astropy.stats import sigma_clipped_stats

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
K = 3.0  # softening: divides by K * sky_sigma before arcsinh


def compute_stats(real_cutouts: dict) -> dict:
    """
    Compute per-band sky_med and sky_sigma from real cutouts.

    Parameters
    ----------
    real_cutouts : dict  {band: np.ndarray shape (N, H, W)}
        Sky-subtracted real cutouts in MJy/sr.  Most pixels are sky so
        sigma-clipped stats give the sky residual and noise level.

    Returns
    -------
    dict  {band: {'sky_med': float, 'sky_sigma': float, 'k': float}}
    """
    stats = {}
    for band, cutouts in real_cutouts.items():
        flat = cutouts.ravel().astype(np.float64)
        # sigma_clipped_stats returns (mean, median, std)
        _, sky_med, sky_sigma = sigma_clipped_stats(flat, sigma=3, maxiters=5)
        stats[band] = {
            'sky_med': float(sky_med),
            'sky_sigma': float(sky_sigma),
            'k': K,
        }
        print(f'  {band}: sky_med={sky_med:.6f}  sky_sigma={sky_sigma:.6f}')
    return stats


def normalize(image: np.ndarray, band: str, stats: dict) -> np.ndarray:
    """
    Apply fixed arcsinh stretch to a single image or batch.

    Parameters
    ----------
    image : np.ndarray  any shape, float32/64
    band  : str  one of BANDS
    stats : dict  from load_stats()

    Returns
    -------
    np.ndarray  same shape, float32
    """
    s = stats[band]
    return np.arcsinh((image - s['sky_med']) / (s['k'] * s['sky_sigma'])).astype(np.float32)


def denormalize(stretched: np.ndarray, band: str, stats: dict) -> np.ndarray:
    """Invert normalize() — useful for sanity checks and display."""
    s = stats[band]
    return (np.sinh(stretched) * s['k'] * s['sky_sigma'] + s['sky_med']).astype(np.float32)


def normalize_stack(stack: np.ndarray, band: str, stats: dict) -> np.ndarray:
    """Normalize an (N, H, W) stack in one call."""
    return normalize(stack, band, stats)


def normalize_4band(images_dict: dict, stats: dict) -> np.ndarray:
    """
    Normalize a dict of 4-band stacks and return a single (N, 4, H, W) tensor.

    Parameters
    ----------
    images_dict : dict  {band: np.ndarray (N, H, W)}
    stats       : dict  from load_stats()

    Returns
    -------
    np.ndarray  (N, 4, H, W)  float32
    """
    bands_ordered = BANDS
    channels = [normalize_stack(images_dict[b], b, stats) for b in bands_ordered]
    return np.stack(channels, axis=1)  # (N, 4, H, W)


def save_stats(path: str, stats: dict) -> None:
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved normalization stats -> {path}')


def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
