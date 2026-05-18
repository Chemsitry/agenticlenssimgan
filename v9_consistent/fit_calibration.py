"""
fit_calibration.py — fit per-band Faber-Jackson intercepts on the 5-galaxy
DESI x JWST sample, with the slope fixed at the classical L∝σ⁴ value.

Output: fj_params.json containing one intercept per band plus per-galaxy
residuals, plus a small QA plot fj_calibration.png.

Faber-Jackson (slope-fixed):
    M_abs(band) = -10 * log10(sigma_v) + b_intercept(band)
                = M_abs                            (per galaxy, per band)
    b_intercept = M_abs + 10 * log10(sigma_v)      (one number per galaxy)
We take the median b_intercept across our 5 galaxies as the calibrated
zero-point. Median is robust at N=5; std across galaxies → uncertainty.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18
import astropy.units as u

# Prefer the extended sample (catalog mags + mosaic-photometry additions) if present
_EXT = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent/sample_final_extended.parquet')
_ORIG = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache/sample_final.parquet')
SAMPLE_PARQUET = _EXT if _EXT.exists() else _ORIG
OUT_JSON = Path(__file__).parent / 'fj_params.json'
OUT_PLOT = Path(__file__).parent / 'fj_calibration.png'

BANDS = ['F115W', 'F150W', 'F277W']
MAG_COLS = {'F115W': 'm115', 'F150W': 'm150', 'F277W': 'm277'}
FJ_SLOPE = -10.0   # L ∝ σ⁴ in (M_abs vs log σ_v) space


def distance_modulus(z):
    """Planck18 distance modulus (mag) at redshift z."""
    dl = Planck18.luminosity_distance(z).to(u.pc).value
    return 5.0 * np.log10(dl / 10.0)


def fit_intercepts(df: pd.DataFrame) -> dict:
    """Per-band median intercept + scatter. Returns dict ready for JSON."""
    log_sig = np.log10(df['VDISP'].to_numpy())          # log km/s
    mu = np.array([distance_modulus(z) for z in df['Z'].to_numpy()])

    out = {'slope_fj': FJ_SLOPE,
           'cosmology': 'Planck18 (K-correction ignored)',
           'sample_parquet': str(SAMPLE_PARQUET),
           'n_galaxies': int(len(df)),
           'bands': {}}

    for band in BANDS:
        m_app = df[MAG_COLS[band]].to_numpy()
        M_abs = m_app - mu
        # b_intercept_i = M_abs_i + 10 * log_sig_i  (because M_abs = -10 log σ + b)
        b_i = M_abs - FJ_SLOPE * log_sig
        med = float(np.median(b_i))
        sca = float(np.std(b_i, ddof=1)) if len(b_i) > 1 else float('nan')
        out['bands'][band] = {
            'intercept_median': med,
            'intercept_scatter_dex_in_mag': sca,
            'per_galaxy_intercept': b_i.tolist(),
            'per_galaxy_M_abs': M_abs.tolist(),
            'per_galaxy_log_sigma': log_sig.tolist(),
        }
    return out


def plot_calibration(df, params, out_path):
    log_sig = np.log10(df['VDISP'].to_numpy())
    mu = np.array([distance_modulus(z) for z in df['Z'].to_numpy()])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    xx = np.linspace(1.7, 2.6, 50)
    for ax, band in zip(axes, BANDS):
        m_app = df[MAG_COLS[band]].to_numpy()
        M_abs = m_app - mu
        b = params['bands'][band]['intercept_median']
        ax.scatter(log_sig, M_abs, c='C0', s=70, edgecolor='k', zorder=3)
        for x, y, z in zip(log_sig, M_abs, df['Z'].to_numpy()):
            ax.annotate(f'z={z:.2f}', (x, y), textcoords='offset points',
                        xytext=(7, 5), fontsize=8, color='#444')
        ax.plot(xx, FJ_SLOPE * xx + b, 'k--', lw=1.5,
                label=f'FJ slope=-10 (fit b={b:+.2f})')
        ax.set_xlabel(r'$\log_{10}\,\sigma_v$  [km/s]')
        ax.set_title(f'{band}  N={len(df)}')
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc='lower left')
    axes[0].set_ylabel('Absolute AB mag  (apparent − μ(z), Planck18)')
    fig.suptitle('Faber-Jackson calibration on 5 DESI×JWST DEV galaxies (slope fixed at -10)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    df = pd.read_parquet(SAMPLE_PARQUET)
    n_total = len(df)
    df = df.dropna(subset=['m115', 'm150', 'm277', 'VDISP', 'Z']).reset_index(drop=True)
    if len(df) < n_total:
        print(f'  dropped {n_total - len(df)} rows with NaN mags/VDISP/Z')
    print(f'loaded {len(df)} galaxies from {SAMPLE_PARQUET.name}')
    print(df[['Z','VDISP','m115','m150','m277','field']].to_string(index=False))
    params = fit_intercepts(df)
    OUT_JSON.write_text(json.dumps(params, indent=2))
    print(f'\nwrote {OUT_JSON}')
    for band in BANDS:
        b = params['bands'][band]
        print(f'  {band}: intercept = {b["intercept_median"]:+.3f}  '
              f'(scatter {b["intercept_scatter_dex_in_mag"]:.3f} mag)')
    plot_calibration(df, params, OUT_PLOT)
    print(f'wrote {OUT_PLOT}')


if __name__ == '__main__':
    main()
