"""
prep_gan_data.py — Prepare training data for SimGAN refinement.

Simulated: Center-crop 128x128 from galaxy-only images (no background).
Real: Find brightest galaxy in each background patch, crop 128x128, subtract
      local background estimate to isolate galaxy morphology.

Both domains contain isolated galaxy light only, so the discriminator must
learn morphology/texture differences — not background or brightness cues.

Usage:
    .venv/bin/python3 prep_gan_data.py
"""

import argparse
import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']


def center_crop(arr, crop_size):
    """Center-crop (N, H, W) to (N, crop_size, crop_size)."""
    _, h, w = arr.shape
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return arr[:, y0:y0 + crop_size, x0:x0 + crop_size]


def find_galaxy_crops(bg_band, crop_size, margin=10):
    """Find the brightest galaxy in each background patch, return crop coords."""
    n, h, w = bg_band.shape
    half = crop_size // 2
    coords = []

    for i in range(n):
        patch = bg_band[i]
        smoothed = gaussian_filter(patch, sigma=3.0)
        smoothed[:half + margin, :] = 0
        smoothed[-(half + margin):, :] = 0
        smoothed[:, :half + margin] = 0
        smoothed[:, -(half + margin):] = 0
        y, x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        coords.append((y, x))

    return coords


def crop_at_coords(arr, coords, crop_size):
    """Crop (N, H, W) at given (y, x) centers."""
    half = crop_size // 2
    n = arr.shape[0]
    out = np.zeros((n, crop_size, crop_size), dtype=arr.dtype)
    for i, (y, x) in enumerate(coords):
        out[i] = arr[i, y - half:y + half, x - half:x + half]
    return out


def subtract_background(crops):
    """Subtract local background from each crop using edge pixels.

    Estimates background as the median of a 10px border around the crop,
    then subtracts it. This isolates the galaxy light from the sky.
    """
    n, h, w = crops.shape
    out = np.zeros_like(crops)
    for i in range(n):
        patch = crops[i]
        # Use 10px border as background estimate
        border = np.concatenate([
            patch[:10, :].ravel(),
            patch[-10:, :].ravel(),
            patch[:, :10].ravel(),
            patch[:, -10:].ravel(),
        ])
        bg_level = np.median(border)
        out[i] = patch - bg_level
    return out


def normalize(data, percentile_low=1, percentile_high=99):
    """Per-image, per-band percentile normalization to [-1, 1]."""
    n_images, n_bands = data.shape[:2]
    stats = {}
    for b in range(n_bands):
        stats[b] = {
            'p_lo': float(np.percentile(data[:, b], percentile_low)),
            'p_hi': float(np.percentile(data[:, b], percentile_high)),
        }

    for i in range(n_images):
        for b in range(n_bands):
            patch = data[i, b]
            p_lo = np.percentile(patch, percentile_low)
            p_hi = np.percentile(patch, percentile_high)
            if p_hi - p_lo < 1e-8:
                data[i, b] = 0
            else:
                data[i, b] = np.clip((patch - p_lo) / (p_hi - p_lo), 0, 1) * 2 - 1

    return data, stats


def main():
    parser = argparse.ArgumentParser(description='Prepare SimGAN training data')
    parser.add_argument('--sim_dir', default='output/v4')
    parser.add_argument('--bg_dir', default='prepped_mosaic_630')
    parser.add_argument('--out_dir', default='output/gan')
    parser.add_argument('--crop', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Simulated: center crops of galaxy-only (no background) ───────────
    print(f'Loading galaxy-only images from {args.sim_dir}...')
    sim_arrays = []
    for band in BANDS:
        path = os.path.join(args.sim_dir, f'galaxies_{band}.npy')
        data = np.load(path, mmap_mode='r')
        print(f'  {band}: {data.shape}')
        sim_arrays.append(center_crop(data, args.crop))
    sim_data = np.stack(sim_arrays, axis=1).astype(np.float32)
    print(f'  Sim stacked: {sim_data.shape}')

    # Apply sqrt stretch to compress dynamic range
    sim_data = np.sqrt(np.clip(sim_data, 0, None))

    # ── Real: find brightest galaxy, crop, subtract background ───────────
    print(f'\nLoading real backgrounds from {args.bg_dir}...')
    bg_f277w = np.load(os.path.join(args.bg_dir, 'F277W', 'backgrounds.npy'), mmap_mode='r')
    print(f'  Finding brightest galaxy in each of {bg_f277w.shape[0]} patches...')
    coords = find_galaxy_crops(bg_f277w, args.crop)

    # Check quality
    peaks = []
    for i, (y, x) in enumerate(coords):
        half = args.crop // 2
        crop = bg_f277w[i, y - half:y + half, x - half:x + half]
        peaks.append(np.max(crop))
    peaks = np.array(peaks)
    print(f'  Galaxy peaks: median={np.median(peaks):.3f}, '
          f'min={np.min(peaks):.3f}, max={np.max(peaks):.3f}')

    # Filter out empty patches
    threshold = np.percentile(peaks, 10)
    good_idx = np.where(peaks > threshold)[0]
    coords_good = [coords[i] for i in good_idx]
    print(f'  Keeping {len(good_idx)} patches with galaxies')

    # Crop all bands at galaxy locations, then subtract background
    real_arrays = []
    for band in BANDS:
        bg = np.load(os.path.join(args.bg_dir, band, 'backgrounds.npy'), mmap_mode='r')
        bg_good = bg[good_idx]
        cropped = crop_at_coords(bg_good, coords_good, args.crop)
        # Subtract local background to isolate galaxy light
        cropped = subtract_background(cropped)
        real_arrays.append(cropped)
        print(f'  {band}: cropped + bg-subtracted {cropped.shape}')
    real_data = np.stack(real_arrays, axis=1).astype(np.float32)

    # Apply sqrt stretch (clip negatives from bg subtraction)
    real_data = np.sqrt(np.clip(real_data, 0, None))
    print(f'  Real stacked: {real_data.shape}')

    # ── Normalize together ────────────────────────────────────────────────
    print('\nComputing normalization stats...')
    combined = np.concatenate([sim_data, real_data], axis=0)
    combined_norm, norm_stats = normalize(combined)

    n_sim = sim_data.shape[0]
    sim_norm = combined_norm[:n_sim]
    real_norm = combined_norm[n_sim:]

    print(f'  Sim range:  [{sim_norm.min():.3f}, {sim_norm.max():.3f}]')
    print(f'  Real range: [{real_norm.min():.3f}, {real_norm.max():.3f}]')

    # ── Save ──────────────────────────────────────────────────────────────
    print(f'\nSaving to {args.out_dir}...')
    np.save(os.path.join(args.out_dir, 'sim_train.npy'), sim_norm)
    np.save(os.path.join(args.out_dir, 'real_train.npy'), real_norm)

    for name in ['lensed', 'theta_Es', 'z_lens', 'z_source', 'masses']:
        path = os.path.join(args.sim_dir, f'{name}.npy')
        if os.path.exists(path):
            np.save(os.path.join(args.out_dir, f'{name}.npy'), np.load(path))

    with open(os.path.join(args.out_dir, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    info = {
        'n_sim': int(n_sim),
        'n_real': int(real_data.shape[0]),
        'crop_size': args.crop,
        'bands': BANDS,
        'sim_dir': args.sim_dir,
        'bg_dir': args.bg_dir,
        'note': 'Sim=galaxy-only center crops (no background), Real=bg-subtracted galaxy crops',
    }
    with open(os.path.join(args.out_dir, 'data_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f'\nDone! {n_sim} sim + {real_data.shape[0]} real galaxy crops prepared.')
    print(f'  sim_train.npy:  {sim_norm.shape} ({sim_norm.nbytes / 1e6:.1f} MB)')
    print(f'  real_train.npy: {real_norm.shape} ({real_norm.nbytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
