"""
pca.py — Stage 0: PCA + logistic regression baseline.

This is the most important thing to run before touching a neural network.
It answers: "Can a simple linear method already separate sim from real?"

  - If cross-validated accuracy is ~99%: the signal is linear and concentrated
    in a few principal components.  Inspect those components to find the bug.
    We might not need a GAN at all.
  - If accuracy is ~50% (chance): the sim-vs-real difference is non-linear or
    only visible to a convolutional network.  Proceed to Stage 2.
  - Anywhere in between (say 70-90%): the signal is partly linear.  The GAN
    needs to beat this number to be useful.

The output accuracy is the BASELINE TO BEAT in all later stages.

Steps (from the plan, Section 3):
  1. Normalize all sim and real images with the fixed arcsinh stretch.
  2. Flatten each 4-band image to a vector of length 4 x H x W.
  3. Stack into one matrix of shape (N_sim + N_real, 4HW).
  4. Fit PCA(n_components=50) on the combined matrix.
  5. Project all images into the 50-dim PCA space.
  6. Train LogisticRegression (label: 0=sim, 1=real).
  7. Report 5-fold cross-validated accuracy.
  8. Save the top PCA components reshaped to (4, H, W) as images — that's
     literally a picture of what separates sim from real.

Usage:
    python -m gan.baselines.pca
    python -m gan.baselines.pca --n-sim 5000 --n-real 5000
    python -m gan.baselines.pca --n-components 50 --out-dir output/gan/baselines
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sim-dir',      default='output/v3',
                   help='Sim images directory (default: output/v3)')
    p.add_argument('--real-dir',     default='output/gan/real_cutouts',
                   help='Real cutouts directory (default: output/gan/real_cutouts)')
    p.add_argument('--out-dir',      default='output/gan/baselines',
                   help='Output directory for results and figures')
    p.add_argument('--n-sim',        type=int, default=10000,
                   help='Max sim images to use (default: 10000)')
    p.add_argument('--n-real',       type=int, default=10000,
                   help='Max real images to use (default: 10000)')
    p.add_argument('--n-components', type=int, default=50,
                   help='PCA components (default: 50, as in the plan)')
    p.add_argument('--cv-folds',     type=int, default=5,
                   help='Cross-validation folds (default: 5)')
    p.add_argument('--seed',         type=int, default=42)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_normalize(sim_dir: Path, real_dir: Path,
                       n_sim: int, n_real: int,
                       stats: dict) -> tuple:
    """
    Load sim and real images, apply arcsinh normalization, flatten to vectors.

    Returns
    -------
    X : np.ndarray  (N_sim + N_real, 4 * H * W)  float32
    y : np.ndarray  (N_sim + N_real,)  int  (0=sim, 1=real)
    """
    from gan.data.normalize import normalize

    print('Loading and normalizing sim images...')
    sim_bands = []
    for band in BANDS:
        path = sim_dir / f'images_{band}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'Missing {path}. Run simulate_v3.py first.')
        arr = np.load(str(path), mmap_mode='r')
        arr = normalize(np.array(arr[:n_sim], dtype=np.float32), band, stats)
        sim_bands.append(arr)   # (N, H, W)

    print('Loading and normalizing real images...')
    real_bands = []
    for band in BANDS:
        path = real_dir / f'cutouts_{band}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'Missing {path}. Run prep_real_targets.py first.')
        arr = np.load(str(path), mmap_mode='r')
        arr = normalize(np.array(arr[:n_real], dtype=np.float32), band, stats)
        real_bands.append(arr)   # (M, H, W)

    n_s = sim_bands[0].shape[0]
    n_r = real_bands[0].shape[0]
    H, W = sim_bands[0].shape[1], sim_bands[0].shape[2]
    flat_len = 4 * H * W

    print(f'  sim: {n_s} images  real: {n_r} images  flat_len: {flat_len}')

    # Stack bands along channel axis then flatten
    sim_stack  = np.stack(sim_bands,  axis=1).reshape(n_s, flat_len)   # (N_s, 4HW)
    real_stack = np.stack(real_bands, axis=1).reshape(n_r, flat_len)   # (N_r, 4HW)

    X = np.concatenate([sim_stack, real_stack], axis=0)
    y = np.concatenate([np.zeros(n_s, dtype=int), np.ones(n_r, dtype=int)])
    return X, y, (H, W)


# ── PCA + logistic regression ─────────────────────────────────────────────────

def run_pca_baseline(X: np.ndarray, y: np.ndarray,
                     n_components: int, cv_folds: int, seed: int) -> dict:
    """
    Fit PCA → LogisticRegression, report cross-validated accuracy.

    Returns a results dict suitable for JSON serialisation.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline

    print(f'\nFitting PCA({n_components}) on {X.shape[0]} images x {X.shape[1]} features...')
    t0 = time.time()

    # NOTE: sklearn's PCA centres the data (subtracts column means), which
    # is correct here — we want to find variance axes, not mean offsets.
    pipe = Pipeline([
        ('pca', PCA(n_components=n_components, random_state=seed, svd_solver='randomized')),
        ('lr',  LogisticRegression(max_iter=1000, C=1.0, random_state=seed, solver='lbfgs')),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=['accuracy', 'roc_auc'],
                            return_train_score=True,
                            n_jobs=1)   # keep n_jobs=1 to avoid memory issues on HPC

    results = {
        'n_sim':               int((y == 0).sum()),
        'n_real':              int((y == 1).sum()),
        'n_components':        n_components,
        'cv_folds':            cv_folds,
        'cv_accuracy_mean':    float(scores['test_accuracy'].mean()),
        'cv_accuracy_std':     float(scores['test_accuracy'].std()),
        'cv_roc_auc_mean':     float(scores['test_roc_auc'].mean()),
        'cv_roc_auc_std':      float(scores['test_roc_auc'].std()),
        'train_accuracy_mean': float(scores['train_accuracy'].mean()),
        'elapsed_s':           round(time.time() - t0, 1),
    }

    print(f'  Elapsed: {results["elapsed_s"]:.1f}s')
    print(f'  CV accuracy: {results["cv_accuracy_mean"]:.4f} ± {results["cv_accuracy_std"]:.4f}')
    print(f'  CV ROC-AUC:  {results["cv_roc_auc_mean"]:.4f} ± {results["cv_roc_auc_std"]:.4f}')
    return results, pipe


# ── PCA component visualisation ───────────────────────────────────────────────

def save_top_components_figure(X: np.ndarray, y: np.ndarray,
                               n_components: int, H: int, W: int,
                               seed: int, out_path: Path) -> None:
    """
    Fit PCA on ALL data, reshape top components to (4, H, W), display per band.

    These images are literally the axes of maximum variance between sim and real.
    If the first component shows a bright blob in the center → that's the
    centered-lens shortcut.  If it shows a texture pattern → that's PSF/noise.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    print('\nFitting PCA for component visualisation...')
    pca = PCA(n_components=n_components, random_state=seed, svd_solver='randomized')
    pca.fit(X)

    n_show = min(4, n_components)
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for ci in range(n_show):
        comp = pca.components_[ci].reshape(4, H, W)   # (4, H, W)
        for bi, band in enumerate(BANDS):
            ax = axes[ci, bi]
            img = comp[bi]
            vmax = np.abs(img).max()
            ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
            ax.set_title(f'PC{ci+1} — {band}\n'
                         f'({pca.explained_variance_ratio_[ci]*100:.1f}% var)',
                         fontsize=8)
            ax.axis('off')

    plt.suptitle(
        'Top PCA components  (red=positive, blue=negative)\n'
        'Each panel: 4 bands of one principal component.\n'
        'These are the directions of maximum sim-vs-real variance.',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f'  Saved -> {out_path}')


def save_pca_projection_figure(X: np.ndarray, y: np.ndarray,
                               n_components: int, seed: int,
                               out_path: Path) -> None:
    """
    Project all images into PCA space, plot PC1 vs PC2 coloured by sim/real.
    If sim and real are cleanly separated in the first two components → the
    difference is obviously linear.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed, svd_solver='randomized')
    Z   = pca.fit_transform(X)   # (N, n_components)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=2, alpha=0.3,
               color='steelblue', label='sim', rasterized=True)
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=2, alpha=0.3,
               color='tomato', label='real', rasterized=True)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.legend(markerscale=4, fontsize=10)
    ax.set_title('PCA projection: sim vs real\n'
                 'Clean separation → difference is linear (inspect PC components above)',
                 fontsize=10)
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

    print('baseline_pca.py')
    print(f'  sim_dir      : {sim_dir}')
    print(f'  real_dir     : {real_dir}')
    print(f'  n_sim        : {args.n_sim}')
    print(f'  n_real       : {args.n_real}')
    print(f'  n_components : {args.n_components}')
    print(f'  cv_folds     : {args.cv_folds}')

    # Load normalization stats
    from gan.data.normalize import load_stats
    stats_path = real_dir / 'normalization.json'
    if not stats_path.exists():
        raise FileNotFoundError(
            f'normalization.json not found at {stats_path}. '
            f'Run prep_real_targets.py first.')
    stats = load_stats(str(stats_path))

    # Load and normalize data
    X, y, (H, W) = load_and_normalize(
        sim_dir, real_dir, args.n_sim, args.n_real, stats)

    # PCA + logistic regression
    results, pipe = run_pca_baseline(
        X, y, args.n_components, args.cv_folds, args.seed)

    # Save results JSON
    results_path = out_dir / 'pca_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved -> {results_path}')

    # Visualise top components (what does the PCA "see"?)
    save_top_components_figure(
        X, y, args.n_components, H, W, args.seed,
        out_dir / 'pca_top_components.png')

    # PCA projection scatter plot
    save_pca_projection_figure(
        X, y, args.n_components, args.seed,
        out_dir / 'pca_projection.png')

    # ── Interpretation guide ──────────────────────────────────────────────────

    acc = results['cv_accuracy_mean']
    print('\n' + '=' * 60)
    print(f'BASELINE ACCURACY: {acc*100:.1f}%')
    print('=' * 60)

    if acc > 0.98:
        print('>> RESULT: Very high accuracy (>98%).')
        print('   The sim-vs-real difference is STRONG AND LINEAR.')
        print('   Look at pca_top_components.png — the visible pattern IS the bug.')
        print('   You may not need a GAN. Show these components to your colleague.')
    elif acc > 0.80:
        print('>> RESULT: High accuracy (80-98%).')
        print('   The difference is partly linear. The GAN must beat this number.')
        print('   Inspect pca_top_components.png for the linear part of the signal.')
        print('   Proceed to Stage 2 (discriminator-only training).')
    elif acc > 0.60:
        print('>> RESULT: Moderate accuracy (60-80%).')
        print('   The difference is mostly non-linear. A CNN discriminator should')
        print('   do much better. Proceed to Stage 2.')
    else:
        print('>> RESULT: Near-chance accuracy (~50-60%).')
        print('   The linear signal is weak. Possible causes:')
        print('   (a) the sim is already very good (unlikely given the prof\'s observation)')
        print('   (b) a normalization bug is hiding the signal — check pixel_histograms.png')
        print('   (c) the difference is purely structural/CNN-detectable.')
        print('   Proceed to Stage 2 cautiously, and re-read Section 8.6 of the plan.')

    print(f'\nSave this number: {acc*100:.1f}% ± {results["cv_accuracy_std"]*100:.1f}%')
    print('This is the BASELINE. Every later stage must beat it.')
    print(f'Results: {results_path}')


if __name__ == '__main__':
    main()
