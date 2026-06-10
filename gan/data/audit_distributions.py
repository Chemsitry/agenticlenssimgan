"""
audit_distributions.py — Stage 0: compare sim vs real pixel statistics.

Produces four diagnostic figures that answer: "Does our simulator look
anything like real JWST data, and if not, where does it fail?"

Figures produced (all in --out-dir):
  1. pixel_histograms.png   — per-band pixel value distributions, sim vs real.
                              Log y-axis so tails are visible.  If the two
                              histograms overlap, the marginal distribution is OK.
                              If they don't, we have an obvious normalization bug.

  2. radial_profiles.png    — average radial surface-brightness profile for the
                              brightest object in each stamp (sim: the lens galaxy;
                              real: the brightest blob, which is usually a galaxy).
                              A profile mismatch tells us the PSF or size distribution
                              is wrong.

  3. centroid_positions.png — where is the brightest pixel (argmax) in each stamp?
                              For sim: should be near (62, 62) because the lens is
                              hardcoded at center_x=0.  For real: spread across
                              the stamp.  A spike at center = "centered-lens shortcut"
                              that the discriminator will trivially exploit.

  4. power_spectra.png      — azimuthally averaged 2D power spectrum per band.
                              Mismatch at high spatial frequencies = PSF or noise model
                              is wrong.  Mismatch at low frequencies = large-scale
                              structure (e.g., galaxy size distribution) is wrong.

Usage:
    python -m gan.data.audit_distributions
    python -m gan.data.audit_distributions --sim-dir output/v3 --size 125
    python -m gan.data.audit_distributions --n-sim 2000 --n-real 2000  # quick run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sim-dir',  default='output/v3',
                   help='Directory with images_{band}.npy from simulate_v3.py (default: output/v3)')
    p.add_argument('--real-dir', default='output/gan/real_cutouts',
                   help='Directory with cutouts_{band}.npy from prep_real_targets.py')
    p.add_argument('--out-dir',  default='output/gan/baselines',
                   help='Where to save figures (default: output/gan/baselines)')
    p.add_argument('--size',     type=int, default=125)
    p.add_argument('--n-sim',    type=int, default=5000,
                   help='Max sim images to use (default: 5000; set lower for speed)')
    p.add_argument('--n-real',   type=int, default=5000,
                   help='Max real images to use (default: 5000)')
    p.add_argument('--norm',     action='store_true',
                   help='Apply arcsinh normalization before plotting histograms')
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sim(sim_dir: Path, n: int) -> dict:
    """Load up to n sim images per band.  Returns dict {band: (N, H, W)}."""
    data = {}
    for band in BANDS:
        path = sim_dir / f'images_{band}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'Sim images not found at {path}.\n'
                f'Run: python simulate_v3.py --n 5000  (or more)')
        arr = np.load(str(path), mmap_mode='r')
        arr = np.array(arr[:n], dtype=np.float32)
        data[band] = arr
    n_loaded = len(data[BANDS[0]])
    print(f'Loaded {n_loaded} sim images from {sim_dir}/')
    return data


def load_real(real_dir: Path, n: int) -> dict:
    """Load up to n real cutouts per band.  Returns dict {band: (N, H, W)}."""
    data = {}
    for band in BANDS:
        path = real_dir / f'cutouts_{band}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'Real cutouts not found at {path}.\n'
                f'Run: python -m gan.data.prep_real_targets')
        arr = np.load(str(path), mmap_mode='r')
        arr = np.array(arr[:n], dtype=np.float32)
        data[band] = arr
    n_loaded = len(data[BANDS[0]])
    print(f'Loaded {n_loaded} real cutouts from {real_dir}/')
    return data


# ── Figure 1: Pixel histograms ────────────────────────────────────────────────

def plot_pixel_histograms(sim: dict, real: dict, out_path: Path,
                          norm_stats: dict = None) -> None:
    """
    Per-band pixel value distribution, log y-axis, sim vs real overlay.
    If norm_stats is provided, apply arcsinh normalization first.
    """
    n_bands = len(BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4), sharey=False)

    for ax, band in zip(axes, BANDS):
        s = sim[band].ravel()
        r = real[band].ravel()

        if norm_stats:
            from gan.data.normalize import normalize
            s = normalize(s, band, norm_stats)
            r = normalize(r, band, norm_stats)
            xlabel = 'arcsinh-normalized flux'
        else:
            xlabel = 'Flux (sky-subtracted MJy/sr)'

        # Use the same bin edges for both so they're directly comparable
        lo = float(np.percentile(np.concatenate([s, r]), 0.5))
        hi = float(np.percentile(np.concatenate([s, r]), 99.5))
        bins = np.linspace(lo, hi, 100)

        ax.hist(s, bins=bins, histtype='step', color='steelblue',
                linewidth=1.5, label=f'sim (n={len(sim[band])})', density=True)
        ax.hist(r, bins=bins, histtype='step', color='tomato',
                linewidth=1.5, label=f'real (n={len(real[band])})', density=True)
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('density' if band == BANDS[0] else '', fontsize=9)
        ax.set_title(band, fontsize=11)
        ax.legend(fontsize=8)

        # Annotate median and std for quick diagnosis
        for arr, label, color in [(s, 'sim', 'steelblue'), (r, 'real', 'tomato')]:
            ax.axvline(float(np.median(arr)), color=color, linestyle='--',
                       linewidth=0.8, alpha=0.7)

    plt.suptitle('Pixel value distribution: sim vs real  (dashed = median)', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f'  Saved -> {out_path}')


# ── Figure 2: Radial profiles ─────────────────────────────────────────────────

def _radial_profile(images: np.ndarray, center: tuple = None) -> np.ndarray:
    """
    Average azimuthal radial profile of images.

    For each image, find the brightness peak (or use 'center' if given),
    then bin pixels by integer radius and average.

    Returns (max_r,) array of mean brightness per radial bin.
    """
    n, h, w = images.shape
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center

    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = int(r.max())

    # Accumulate mean per radius bin
    radial_mean = np.zeros((n, max_r + 1), dtype=np.float32)
    for ri in range(max_r + 1):
        mask = (r == ri)
        if mask.sum() == 0:
            continue
        radial_mean[:, ri] = images[:, mask].mean(axis=1)

    return radial_mean.mean(axis=0)   # average over images


def plot_radial_profiles(sim: dict, real: dict, out_path: Path) -> None:
    """
    Average radial profile of the brightest source in each stamp.
    For sim, the brightest source is the centered lens galaxy.
    For real, we measure from the image center (comparing like-for-like geometry).
    """
    n_bands = len(BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4), sharey=False)

    for ax, band in zip(axes, BANDS):
        s_imgs = sim[band]
        r_imgs = real[band]

        h, w = s_imgs.shape[1:]
        center = (h // 2, w // 2)

        prof_sim  = _radial_profile(s_imgs,  center=center)
        prof_real = _radial_profile(r_imgs, center=center)

        radii = np.arange(len(prof_sim))
        ax.semilogy(radii, np.abs(prof_sim)  + 1e-12, color='steelblue', label='sim')
        ax.semilogy(radii, np.abs(prof_real) + 1e-12, color='tomato',    label='real')
        ax.set_xlabel('Radius (pixels)', fontsize=9)
        ax.set_ylabel('Mean flux' if band == BANDS[0] else '', fontsize=9)
        ax.set_title(band, fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle('Average radial profile from image center: sim vs real', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f'  Saved -> {out_path}')


# ── Figure 3: Argmax (centroid) positions ─────────────────────────────────────

def plot_centroid_positions(sim: dict, real: dict, out_path: Path) -> None:
    """
    2D histogram of where the brightest pixel sits in each stamp.

    For sim images: if there's a spike at the image center, the lens is always
    centered — that's the shortcut Stage 1 tries to fix.

    For real images: should be spread across the stamp because the bright galaxy
    can be anywhere in a random cutout.
    """
    band = 'F115W'   # one band is enough to see the centroid shortcut
    s_imgs = sim[band]
    r_imgs = real[band]
    h, w   = s_imgs.shape[1:]

    def get_centroids(imgs):
        flat = imgs.reshape(len(imgs), -1)
        idx  = flat.argmax(axis=1)
        rows = idx // w
        cols = idx % w
        return rows, cols

    ys_sim,  xs_sim  = get_centroids(s_imgs)
    ys_real, xs_real = get_centroids(r_imgs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bins_y = np.arange(h + 1)
    bins_x = np.arange(w + 1)

    for ax, ys, xs, label, color in [
        (axes[0], ys_sim,  xs_sim,  'sim',  'steelblue'),
        (axes[1], ys_real, xs_real, 'real', 'tomato'),
    ]:
        h2d, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y])
        im = ax.imshow(h2d.T, origin='lower', cmap='hot',
                       extent=[0, w, 0, h], aspect='equal')
        ax.set_title(f'{label}: argmax distribution ({band})', fontsize=11)
        ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='count')
        # Mark image center
        ax.axvline(w // 2, color='cyan', linewidth=1, linestyle='--', alpha=0.7)
        ax.axhline(h // 2, color='cyan', linewidth=1, linestyle='--', alpha=0.7)

    plt.suptitle('Brightest-pixel position: centered spike in sim = lens shortcut', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f'  Saved -> {out_path}')


# ── Figure 4: 2D power spectra ────────────────────────────────────────────────

def _mean_power_spectrum(images: np.ndarray) -> tuple:
    """
    Compute the azimuthally averaged 2D power spectrum averaged over images.

    Returns (freq_bins, mean_power) both 1D arrays.
    """
    n, h, w = images.shape
    # Hanning window reduces spectral leakage at edges
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    window /= window.mean()

    ps_sum = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        fft2 = np.fft.fft2((images[i] * window).astype(np.float64))
        ps_sum += np.abs(np.fft.fftshift(fft2)) ** 2
    ps_mean = ps_sum / n

    # Azimuthal average
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = min(cy, cx)

    freq  = np.arange(max_r)
    power = np.array([ps_mean[r == ri].mean() if (r == ri).any() else 0
                      for ri in freq])
    # Convert pixel-frequency to 1/arcsec using 30mas pixel scale
    pixel_scale_arcsec = 0.03
    freq_arcsec = freq / (h * pixel_scale_arcsec)

    return freq_arcsec[1:], power[1:]   # skip DC term


def plot_power_spectra(sim: dict, real: dict, out_path: Path) -> None:
    n_bands = len(BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4), sharey=False)

    for ax, band in zip(axes, BANDS):
        freq_s, ps_s  = _mean_power_spectrum(sim[band])
        freq_r, ps_r  = _mean_power_spectrum(real[band])

        ax.loglog(freq_s, ps_s, color='steelblue', label='sim')
        ax.loglog(freq_r, ps_r, color='tomato',    label='real')
        ax.set_xlabel('Spatial freq (1/arcsec)', fontsize=9)
        ax.set_ylabel('Power' if band == BANDS[0] else '', fontsize=9)
        ax.set_title(band, fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle('Mean 2D power spectrum: sim vs real\n'
                 'High-freq mismatch → PSF/noise wrong; '
                 'low-freq mismatch → galaxy size distribution wrong', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f'  Saved -> {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    sim_dir  = Path(args.sim_dir)
    real_dir = Path(args.real_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('audit_distributions.py')
    print(f'  sim_dir : {sim_dir}')
    print(f'  real_dir: {real_dir}')
    print(f'  out_dir : {out_dir}')

    # Load data
    sim  = load_sim(sim_dir, args.n_sim)
    real = load_real(real_dir, args.n_real)

    # Load normalization stats if requested
    norm_stats = None
    if args.norm:
        from gan.data.normalize import load_stats
        stats_path = real_dir / 'normalization.json'
        if stats_path.exists():
            norm_stats = load_stats(str(stats_path))
            print('  Normalization stats loaded; applying arcsinh before histograms')
        else:
            print('  WARNING: --norm requested but normalization.json not found; skipping')

    print('\nFigure 1: pixel histograms...')
    plot_pixel_histograms(sim, real, out_dir / 'pixel_histograms.png', norm_stats)

    print('Figure 2: radial profiles...')
    plot_radial_profiles(sim, real, out_dir / 'radial_profiles.png')

    print('Figure 3: centroid positions...')
    plot_centroid_positions(sim, real, out_dir / 'centroid_positions.png')

    print('Figure 4: power spectra...')
    plot_power_spectra(sim, real, out_dir / 'power_spectra.png')

    # Save a brief text summary that can be read before looking at figures
    summary_lines = []
    for band in BANDS:
        s = sim[band].ravel()
        r = real[band].ravel()
        summary_lines.append(
            f'{band}:  sim  median={np.median(s):.4f}  std={np.std(s):.4f} '
            f'| real median={np.median(r):.4f}  std={np.std(r):.4f}'
        )
    summary_path = out_dir / 'pixel_stats_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('Per-band pixel statistics (sky-subtracted MJy/sr)\n')
        f.write('=' * 60 + '\n')
        f.write(f'sim_dir : {sim_dir}\n')
        f.write(f'real_dir: {real_dir}\n')
        f.write(f'n_sim   : {len(sim[BANDS[0]])}\n')
        f.write(f'n_real  : {len(real[BANDS[0]])}\n\n')
        f.write('\n'.join(summary_lines))
        f.write('\n')
    print(f'\nPixel stats summary -> {summary_path}')
    for line in summary_lines:
        print(f'  {line}')

    print(f'\nAll figures saved to {out_dir}/')
    print('Next: inspect figures, then run gan/baselines/pca.py')


if __name__ == '__main__':
    main()
