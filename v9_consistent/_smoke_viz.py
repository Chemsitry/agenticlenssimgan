"""Smoke-test visualization for full_5000_partial.
Per bin: figure of 20 image RGBs + figure of 20 matching arcs RGBs.
Plus one figure of 20 non-lensed image RGBs.
Uses the same RGB recipe as _show_lensed.py (F277W/F150W/F115W → R/G/B)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

ROOT  = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')
DATA  = ROOT / 'output' / 'full_5000_partial'
OUT   = ROOT / 'output' / 'smoke_full_5000'
OUT.mkdir(exist_ok=True)
N_PER = 20
RNG   = np.random.default_rng(7)
BANDS_RGB = ['F277W', 'F150W', 'F115W']  # R, G, B

mask = np.load(DATA / 'completed_mask.npy').astype(bool)
te   = np.load(DATA / 'theta_Es.npy')
lbl  = np.load(DATA / 'lensed.npy')
imgs = {b: np.load(DATA / f'images_{b}.npy', mmap_mode='r') for b in BANDS_RGB}
arcs = {b: np.load(DATA / f'arcs_{b}.npy',   mmap_mode='r') for b in BANDS_RGB}

print(f'Loaded: mask sum {mask.sum()}, image shape {imgs["F277W"].shape}')

bin_edges = np.linspace(0.5, 1.8, 11)


def make_rgb(r, g, b, smooth=1.0, gamma=0.50):
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
        ch = np.clip((ch - vlo) / max(vhi - vlo, 1e-9), 0, 1) ** gamma
        out[..., i] = ch
    return out


def grid_rgb(indices, source_dict, title, out_path, show_theta=True):
    nrows, ncols = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    for k in range(nrows * ncols):
        ax = axes.flat[k]
        if k < len(indices):
            idx = int(indices[k])
            rgb = make_rgb(source_dict['F277W'][idx],
                           source_dict['F150W'][idx],
                           source_dict['F115W'][idx])
            ax.imshow(rgb, origin='lower')
            t = f'idx {idx}'
            if show_theta and te[idx] > 0:
                t += f'  θ={te[idx]:.2f}"'
            ax.set_title(t, fontsize=8)
        ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path.name}')


# Per-bin figures
for bin_i in range(10):
    lo, hi = bin_edges[bin_i], bin_edges[bin_i + 1]
    sel = np.where(mask & (lbl > 0) & (te >= lo) & (te < hi))[0]
    if len(sel) == 0:
        print(f'bin {bin_i} [{lo:.2f},{hi:.2f}"]: empty, skipping')
        continue
    n_pick = min(N_PER, len(sel))
    picks = RNG.choice(sel, size=n_pick, replace=False)
    print(f'bin {bin_i} [{lo:.2f},{hi:.2f}"]: {len(sel)} available, plotting {n_pick}')
    grid_rgb(picks, imgs,
             f'Bin {bin_i}  θ_E ∈ [{lo:.2f}, {hi:.2f}"]  —  RGB (F277W/F150W/F115W)',
             OUT / f'bin{bin_i}_images.png')
    grid_rgb(picks, arcs,
             f'Bin {bin_i}  θ_E ∈ [{lo:.2f}, {hi:.2f}"]  —  arcs only',
             OUT / f'bin{bin_i}_arcs.png')

# Non-lensed
sel_nl = np.where(mask & (lbl == 0))[0]
picks_nl = RNG.choice(sel_nl, size=min(N_PER, len(sel_nl)), replace=False)
print(f'non-lensed: {len(sel_nl)} available, plotting {len(picks_nl)}')
grid_rgb(picks_nl, imgs,
         f'Non-lensed  —  RGB (F277W/F150W/F115W)',
         OUT / 'nonlensed_images.png',
         show_theta=False)

print(f'\nDone — figures in {OUT}/')
