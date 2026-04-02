"""
refine_dataset.py — Apply trained SimGAN refiner to simulated images.

Takes v4 simulated images, passes them through the trained refiner,
and saves the refined images as v5.

Usage:
    .venv/bin/python3 refine_dataset.py
    .venv/bin/python3 refine_dataset.py --input output/v4 --output output/v5
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BANDS = ['F115W', 'F150W', 'F277W', 'F444W']


def main():
    parser = argparse.ArgumentParser(description='Refine simulated images with trained SimGAN')
    parser.add_argument('--input', default='output/v4', help='Input simulated images directory')
    parser.add_argument('--output', default='output/v5', help='Output refined images directory')
    parser.add_argument('--model', default='output/v5/simgan_final.pt', help='Trained model path')
    parser.add_argument('--norm_stats', default='output/gan/norm_stats.json', help='Normalization stats')
    parser.add_argument('--crop', type=int, default=128, help='Center crop size used in training')
    parser.add_argument('--batch', type=int, default=16, help='Inference batch size')
    args = parser.parse_args()

    import torch

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────────────────
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    n_channels = checkpoint.get('n_channels', 4)
    bands_trained = checkpoint.get('bands', BANDS)
    print(f'Model trained on: {bands_trained} ({n_channels} channels)')

    # Rebuild refiner (import from train_simgan)
    from train_simgan import build_models
    refiner, _ = build_models(n_channels, device)
    refiner.load_state_dict(checkpoint['refiner'])
    refiner.eval()

    # ── Load normalization stats ──────────────────────────────────────────
    with open(args.norm_stats) as f:
        norm_stats = json.load(f)

    # ── Load input images ─────────────────────────────────────────────────
    print(f'\nLoading images from {args.input}...')
    images = {}
    for band in BANDS:
        path = os.path.join(args.input, f'images_{band}.npy')
        images[band] = np.load(path)
        print(f'  {band}: {images[band].shape}')

    n_images = images[BANDS[0]].shape[0]
    h, w = images[BANDS[0]].shape[1:]

    # ── Process: center-crop, normalize, refine, denormalize, paste back ──
    crop = args.crop
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2

    # Determine which bands to refine
    if set(bands_trained) == set(BANDS):
        band_idx = list(range(4))
    else:
        band_idx = [BANDS.index(b) for b in bands_trained]

    print(f'\nRefining {n_images} images (center {crop}x{crop} region)...')

    # Prepare output arrays (copy originals, then paste refined centers)
    refined_images = {b: images[b].copy() for b in BANDS}

    for start in range(0, n_images, args.batch):
        end = min(start + args.batch, n_images)
        batch_size = end - start

        # Extract center crops and stack selected bands
        crops = np.zeros((batch_size, n_channels, crop, crop), dtype=np.float32)
        for ci, bi in enumerate(band_idx):
            band = BANDS[bi]
            crop_data = images[band][start:end, y0:y0 + crop, x0:x0 + crop]
            # Apply same normalization as training
            crop_data = np.sqrt(np.clip(crop_data, 0, None))
            p_lo = norm_stats[str(ci)]['p_lo']
            p_hi = norm_stats[str(ci)]['p_hi']
            crop_data = np.clip((crop_data - p_lo) / (p_hi - p_lo + 1e-8), 0, 1) * 2 - 1
            crops[:, ci] = crop_data

        # Refine
        with torch.no_grad():
            t_in = torch.tensor(crops, dtype=torch.float32).to(device)
            t_out = refiner(t_in).cpu().numpy()

        # Denormalize and paste back
        for ci, bi in enumerate(band_idx):
            band = BANDS[bi]
            p_lo = norm_stats[str(ci)]['p_lo']
            p_hi = norm_stats[str(ci)]['p_hi']
            # [-1,1] -> [0,1] -> original scale
            refined_crop = (t_out[:, ci] + 1) / 2
            refined_crop = refined_crop * (p_hi - p_lo) + p_lo
            # Undo sqrt stretch
            refined_crop = refined_crop ** 2
            refined_images[band][start:end, y0:y0 + crop, x0:x0 + crop] = refined_crop

        print(f'  [{end}/{n_images}]', end='\r')

    print()

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    for band in BANDS:
        out_path = os.path.join(args.output, f'images_{band}.npy')
        np.save(out_path, refined_images[band])
        print(f'Saved {out_path}  {refined_images[band].shape}')

    # Copy labels and sources unchanged
    for name in ['lensed', 'theta_Es', 'z_lens', 'z_source', 'masses',
                 'sources_F115W', 'sources_F150W', 'sources_F277W', 'sources_F444W']:
        src = os.path.join(args.input, f'{name}.npy')
        if os.path.exists(src):
            data = np.load(src)
            np.save(os.path.join(args.output, f'{name}.npy'), data)

    # Copy metadata and add refinement info
    meta_path = os.path.join(args.input, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta['version'] = 'v5'
    meta['refined'] = True
    meta['refiner_model'] = args.model
    meta['refined_region'] = f'center {crop}x{crop}'
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Preview ───────────────────────────────────────────────────────────
    print('\nGenerating preview...')
    n_show = min(5, n_images)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for i in range(n_show):
        # Show F277W (usually the most informative)
        orig = images['F277W'][i, y0:y0+crop, x0:x0+crop]
        ref = refined_images['F277W'][i, y0:y0+crop, x0:x0+crop]

        vmin = np.percentile(orig[orig > 0], 1) if np.any(orig > 0) else 0
        vmax = np.percentile(orig[orig > 0], 99.5) if np.any(orig > 0) else 1

        axes[0, i].imshow(np.sqrt(np.clip(orig, 0, None)), cmap='gray',
                          origin='lower', vmin=np.sqrt(max(vmin,0)), vmax=np.sqrt(max(vmax,0)))
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        if i == 0:
            axes[0, i].set_ylabel('Original (v4)', fontsize=11)

        axes[1, i].imshow(np.sqrt(np.clip(ref, 0, None)), cmap='gray',
                          origin='lower', vmin=np.sqrt(max(vmin,0)), vmax=np.sqrt(max(vmax,0)))
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel('Refined (v5)', fontsize=11)

    plt.suptitle('SimGAN Refinement: v4 → v5 (F277W center crop)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'refinement_preview.png'), dpi=150)
    print(f'Saved {args.output}/refinement_preview.png')

    print(f'\nDone! Refined images saved to {args.output}/')


if __name__ == '__main__':
    main()
