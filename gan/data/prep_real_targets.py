"""
prep_real_targets.py — Stage 0: extract ~30k random COSMOS-Web cutouts.

Unlike prep_mosaic.py's backgrounds.npy (which avoids galaxies), these
cutouts sample the FULL sky distribution — bright galaxies included.
That is essential: a discriminator trained against galaxy-free "real" sky
would trivially win by detecting the lens galaxy in sim images.

Each cutout is a 4-band (125x125 or 224x224) patch extracted at the SAME
sky location in all four bands (same 30mas grid, so they're aligned).

Rejection criteria (same as prep_mosaic.py, so results are consistent):
  - Patch touches the mosaic edge
  - Any band has fewer than 98% finite, non-zero pixels

Output (relative to agenticlenssimgan/ working directory):
    output/gan/real_cutouts/cutouts_{band}.npy   (N, size, size) float32
    output/gan/real_cutouts/cutout_info.json     center pixel coords + acceptance rate
    output/gan/real_cutouts/normalization.json   sky_med, sky_sigma, k per band

Usage:
    # From agenticlenssimgan/:
    python -m gan.data.prep_real_targets
    python -m gan.data.prep_real_targets --n 30000 --size 125 --seed 42
    python -m gan.data.prep_real_targets --smoke-test   # 50 cutouts, quick sanity check
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from gan.data.normalize import compute_stats, save_stats

# ── Constants (must match prep_mosaic.py) ────────────────────────────────────

MOSAIC_DIR = Path('raw_data/1727_mosaic')
BANDS      = ['F115W', 'F150W', 'F277W', 'F444W']
VALID_FRAC = 0.98    # require 98% valid pixels per patch per band
EDGE_MARGIN = 100    # pixels from mosaic edge to avoid


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n',          type=int, default=30000,
                   help='Target number of cutouts (default: 30000)')
    p.add_argument('--size',       type=int, default=125,
                   help='Cutout size in pixels (default: 125; use 224 for larger field)')
    p.add_argument('--seed',       type=int, default=99,
                   help='Random seed (default: 99; different from prep_mosaic seed=42)')
    p.add_argument('--prepped-dir', default=None,
                   help='prepped_mosaic/ dir to read band_info.json from. '
                        'Auto-detected from --size if not set.')
    p.add_argument('--out-dir',    default=None,
                   help='Output directory (default: output/gan/real_cutouts)')
    p.add_argument('--smoke-test', action='store_true',
                   help='Extract only 50 cutouts to verify data paths and shapes')
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    n_target = 50 if args.smoke_test else args.n
    size     = args.size
    half     = size // 2

    prepped_dir = Path(args.prepped_dir) if args.prepped_dir else (
        Path('prepped_mosaic') if size == 125 else Path(f'prepped_mosaic_{size}'))
    out_dir = Path(args.out_dir) if args.out_dir else Path('output/gan/real_cutouts')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ───────────────────────────────────────────────────────

    band_info_path = prepped_dir / 'band_info.json'
    if not band_info_path.exists():
        raise FileNotFoundError(
            f'band_info.json not found at {band_info_path}. '
            f'Run prep_mosaic.py --size {size} first.')

    with open(band_info_path) as f:
        band_info = json.load(f)

    for band in BANDS:
        mosaic_glob = list((MOSAIC_DIR / band).glob('mosaic*.fits'))
        if not mosaic_glob:
            raise FileNotFoundError(
                f'No mosaic*.fits in {MOSAIC_DIR / band}. '
                f'Check that raw_data/1727_mosaic/{band}/ exists.')

    print(f'prep_real_targets: size={size}  n_target={n_target}  seed={args.seed}')
    print(f'  mosaic_dir : {MOSAIC_DIR}')
    print(f'  prepped_dir: {prepped_dir}')
    print(f'  out_dir    : {out_dir}')
    if args.smoke_test:
        print('  *** SMOKE TEST MODE: extracting 50 cutouts only ***')

    # ── Open all four mosaics (memory-mapped — only reads pages we access) ────

    print('\nOpening mosaics (memory-mapped)...')
    band_data = {}
    for band in BANDS:
        fits_path = sorted((MOSAIC_DIR / band).glob('mosaic*.fits'))[0]
        hdul = fits.open(str(fits_path), memmap=True)
        sci  = hdul[1].data   # SCI extension; shape (ny, nx)

        # Try to open WHT extension for per-pixel weight checking (JWST Level 3).
        # WHT=0 pixels are unobserved; not all mosaic versions have this extension.
        wht = None
        if len(hdul) > 3:
            try:
                wht = hdul[3].data
                print(f'  {band}: SCI {sci.shape}  WHT found at ext 3')
            except Exception:
                pass
        if wht is None:
            print(f'  {band}: SCI {sci.shape}  (no WHT ext — using finite+nonzero check)')

        bg_median = band_info[band]['bg_median']
        band_data[band] = {
            'hdul': hdul, 'sci': sci, 'wht': wht,
            'ny': sci.shape[0], 'nx': sci.shape[1],
            'bg_median': bg_median,
        }

    ref = band_data[BANDS[0]]
    ny, nx = ref['ny'], ref['nx']
    print(f'\nMosaic size: {ny} x {nx} pixels')
    print(f'Sampling {n_target} cutouts of {size}x{size}...')

    # ── Sample random patch centers ───────────────────────────────────────────

    rng = np.random.default_rng(args.seed)
    patches   = {band: [] for band in BANDS}
    centers   = []   # (cy, cx) pixel coords
    attempts  = 0
    max_attempts = n_target * 50   # give up after this many tries

    t0 = time.time()
    while len(patches[BANDS[0]]) < n_target and attempts < max_attempts:
        attempts += 1

        cy = int(rng.integers(half + EDGE_MARGIN, ny - half - EDGE_MARGIN))
        cx = int(rng.integers(half + EDGE_MARGIN, nx - half - EDGE_MARGIN))
        y0, x0 = cy - half, cx - half

        # Check validity across ALL bands before accepting
        all_valid = True
        for band in BANDS:
            sci = band_data[band]['sci']
            patch = sci[y0:y0 + size, x0:x0 + size]

            if patch.shape != (size, size):
                all_valid = False
                break

            valid_mask = np.isfinite(patch) & (patch != 0)
            if valid_mask.mean() < VALID_FRAC:
                all_valid = False
                break

            wht = band_data[band]['wht']
            if wht is not None:
                wht_patch = wht[y0:y0 + size, x0:x0 + size]
                if (wht_patch == 0).mean() > (1 - VALID_FRAC):
                    all_valid = False
                    break

        if not all_valid:
            continue

        # Accept: extract sky-subtracted cutout (matches how backgrounds.npy is saved)
        for band in BANDS:
            sci = band_data[band]['sci']
            patch = sci[y0:y0 + size, x0:x0 + size].copy().astype(np.float32)
            patch -= band_data[band]['bg_median']   # sky-subtract (same as prep_mosaic.py)
            patches[band].append(patch)

        centers.append([int(cy), int(cx)])

        n_done = len(patches[BANDS[0]])
        if n_done % 2000 == 0 or n_done == n_target:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            accept = n_done / attempts * 100
            print(f'  {n_done:5d}/{n_target}  ({attempts} attempts, '
                  f'{accept:.1f}% accept, {rate:.0f} patches/s)')

    # ── Save cutouts ──────────────────────────────────────────────────────────

    n_got = len(patches[BANDS[0]])
    if n_got < n_target:
        print(f'\nWARNING: only got {n_got}/{n_target} cutouts after {attempts} attempts.')
        print('  Consider: lower --n, wider mosaic, or lower VALID_FRAC.')

    print(f'\nSaving {n_got} cutouts...')
    for band in BANDS:
        arr = np.array(patches[band], dtype=np.float32)   # (N, size, size)
        path = out_dir / f'cutouts_{band}.npy'
        np.save(str(path), arr)
        print(f'  {band}: {arr.shape}  {arr.nbytes / 1e6:.1f} MB  -> {path}')

    # ── Compute and save normalization stats ──────────────────────────────────
    # stats are computed from the sky-subtracted real cutouts; sky_med ≈ 0,
    # sky_sigma ≈ the noise RMS — this is the scale we normalise by.

    print('\nComputing normalization stats...')
    cutout_arrays = {band: np.array(patches[band], dtype=np.float32) for band in BANDS}
    stats = compute_stats(cutout_arrays)
    save_stats(str(out_dir / 'normalization.json'), stats)

    # ── Save cutout metadata ──────────────────────────────────────────────────

    info = {
        'n_cutouts':      n_got,
        'size':           size,
        'seed':           args.seed,
        'mosaic_dir':     str(MOSAIC_DIR),
        'valid_frac':     VALID_FRAC,
        'attempts':       attempts,
        'accept_rate':    round(n_got / attempts, 4) if attempts else 0,
        'bands':          BANDS,
        'units':          'MJy/sr, sky-subtracted (bg_median removed per band_info.json)',
        'centers_pix':    centers,   # list of [cy, cx]
    }
    with open(out_dir / 'cutout_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f'Saved -> {out_dir}/cutout_info.json')

    # ── Visual smoke test ─────────────────────────────────────────────────────

    if args.smoke_test:
        _save_smoke_test_figure(cutout_arrays, out_dir, size)

    # ── Summary ───────────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  {n_got} cutouts, {attempts} attempts ({n_got/attempts*100:.1f}% accept rate)')
    print(f'  Output: {out_dir}/')
    print(f'  Next: run audit_distributions.py to compare sim vs real pixel stats')


def _save_smoke_test_figure(cutout_arrays: dict, out_dir: Path, size: int) -> None:
    """Save a 4x4 grid of RGB thumbnails for visual inspection."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm

        n_show = min(16, len(cutout_arrays[BANDS[0]]))
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()

        # Simple RGB: R=F444W, G=F277W, B=mean(F115W,F150W)  (COSMOS-Web convention)
        r = cutout_arrays['F444W']
        g = cutout_arrays['F277W']
        b = (cutout_arrays['F115W'] + cutout_arrays['F150W']) / 2.0

        for i in range(n_show):
            rgb = np.stack([r[i], g[i], b[i]], axis=-1)
            vmax = np.nanpercentile(np.abs(rgb), 99.5)
            norm = AsinhNorm(linear_width=vmax * 0.02, vmin=-vmax * 0.1, vmax=vmax)
            rgb_norm = np.clip(np.stack([norm(rgb[..., c]) for c in range(3)], axis=-1), 0, 1)
            axes[i].imshow(rgb_norm, origin='lower')
            axes[i].axis('off')

        for i in range(n_show, 16):
            axes[i].axis('off')

        plt.suptitle(f'Real COSMOS-Web cutouts (smoke test, {size}x{size})', fontsize=14)
        plt.tight_layout()
        fig_path = out_dir / 'smoke_test_cutouts.png'
        plt.savefig(str(fig_path), dpi=100)
        plt.close()
        print(f'  Smoke test figure -> {fig_path}')
    except Exception as e:
        print(f'  WARNING: could not save smoke test figure: {e}')


if __name__ == '__main__':
    main()
