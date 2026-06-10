"""
test_calibration.py — sanity-test the σ_v calibration round-trip on the
5 DESI×JWST galaxies. For each galaxy, predict σ_v from its mags+z and
compare to the measured DESI σ_v.

Run after fit_calibration.py.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import calibration as cal
from fit_calibration import SAMPLE_PARQUET, BANDS, MAG_COLS


def main():
    df = pd.read_parquet(SAMPLE_PARQUET)
    rows = []
    for _, r in df.iterrows():
        mags = {b: r[MAG_COLS[b]] for b in BANDS}
        sigma_pred, per_band = cal.predict_sigma_v_multiband(mags, r['Z'])
        score = cal.consistency_score(mags, r['Z'], r['VDISP'])
        rows.append({
            'field':       r['field'],
            'z':           r['Z'],
            'sigma_obs':   r['VDISP'],
            'sigma_pred':  sigma_pred,
            'ratio':       sigma_pred / r['VDISP'],
            'pred_F115W':  per_band.get('F115W'),
            'pred_F150W':  per_band.get('F150W'),
            'pred_F277W':  per_band.get('F277W'),
            'resid_F115W': score.get('F115W'),
            'resid_F150W': score.get('F150W'),
            'resid_F277W': score.get('F277W'),
        })
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: f'{x:8.3f}')
    print('\nRound-trip: predict σ_v from each galaxy\'s mags+z, compare to measured σ_v')
    print(out.to_string(index=False))
    print()
    print(f'σ_pred / σ_obs:  median={out["ratio"].median():.3f}  '
          f'std={out["ratio"].std():.3f}')
    print('Per-band residuals (obs - predicted-by-FJ, mag) should be ~0 by '
          'construction since intercept is median fit on these same galaxies.')


if __name__ == '__main__':
    main()
