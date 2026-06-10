"""Render all lensed images from a run as a single PNG grid."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.ndimage import gaussian_filter

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else 'output/smoke_test')
BANDS = ['F115W', 'F150W', 'F277W']

lab = np.load(OUT_DIR / 'lensed.npy')
te  = np.load(OUT_DIR / 'theta_Es.npy')
zl  = np.load(OUT_DIR / 'z_lens.npy')
imgs = {b: np.load(OUT_DIR / f'images_{b}.npy') for b in BANDS}

lensed_idx = np.where(lab == 1)[0]
n = len(lensed_idx)
ncols = 5
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=120)
axes = np.atleast_2d(axes).reshape(nrows, ncols)


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


for k, idx in enumerate(lensed_idx):
    r = k // ncols
    c = k % ncols
    ax = axes[r, c]
    rgb = make_rgb(imgs['F277W'][idx], imgs['F150W'][idx], imgs['F115W'][idx])
    ax.imshow(rgb, origin='lower')
    ax.set_title(f'θ_E={te[idx]:.2f}″  z_l={zl[idx]:.2f}', fontsize=9)
    ax.axis('off')

for k in range(n, nrows * ncols):
    axes[k // ncols, k % ncols].axis('off')

fig.suptitle(f'{n} lensed systems  (output/{OUT_DIR.name}/)', fontsize=14, y=1.00)
fig.tight_layout()
out = OUT_DIR / 'all_lensed.png'
fig.savefig(str(out), dpi=120, bbox_inches='tight')
print(f'wrote {out}')
