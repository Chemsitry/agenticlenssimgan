"""
calibration.py — use the fitted Faber-Jackson per-band intercepts to:

  1. predict_sigma_v(apparent_mag, z, band)    → σ_v in km/s
  2. predict_sigma_v_multiband(mags_dict, z)   → consensus σ_v across bands
  3. consistency_score(mags_dict, z, sigma_v)  → measure of agreement (mag)

Requires fj_params.json next to this file (run fit_calibration.py first).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.cosmology import Planck18
import astropy.units as u

PARAMS_PATH = Path(__file__).parent / 'fj_params.json'
BANDS = ('F115W', 'F150W', 'F277W')

_params = None
def _load():
    global _params
    if _params is None:
        if not PARAMS_PATH.exists():
            raise FileNotFoundError(
                f'{PARAMS_PATH} missing — run fit_calibration.py first.')
        _params = json.loads(PARAMS_PATH.read_text())
    return _params


def distance_modulus(z):
    dl = Planck18.luminosity_distance(z).to(u.pc).value
    return 5.0 * np.log10(dl / 10.0)


def predict_sigma_v(apparent_mag, z, band) -> float:
    """
    Invert M_abs = slope * log σ + b for σ_v.
        log σ_v = (M_abs - b) / slope
    """
    p = _load()
    slope = p['slope_fj']
    b = p['bands'][band]['intercept_median']
    M_abs = float(apparent_mag) - distance_modulus(z)
    log_sigma = (M_abs - b) / slope
    return 10.0 ** log_sigma


def predict_sigma_v_multiband(mags_dict, z, bands: Iterable[str] = BANDS,
                              reducer: str = 'median'):
    """
    Combine per-band predictions into one σ_v. Skips NaN/inf mags.

    Returns
    -------
    sigma_v : float (km/s)
    per_band : dict of {band: σ_v} for the bands that contributed
    """
    per_band = {}
    for b in bands:
        m = mags_dict.get(b)
        if m is None or not np.isfinite(m):
            continue
        per_band[b] = predict_sigma_v(m, z, b)
    if not per_band:
        return float('nan'), per_band
    arr = np.array(list(per_band.values()))
    consensus = float(np.median(arr)) if reducer == 'median' else float(np.mean(arr))
    return consensus, per_band


def consistency_score(mags_dict, z, sigma_v) -> dict:
    """
    Given a candidate σ_v and observed mags, compute the per-band residual
    Δm = m_obs - m_predicted_by_FJ(σ_v, z).
    Positive Δm means the cutout is fainter than FJ would predict for this σ_v.

    Returns dict with per-band residuals + an aggregate.
    """
    p = _load()
    slope = p['slope_fj']
    mu = distance_modulus(z)
    res = {}
    for b in BANDS:
        m_obs = mags_dict.get(b)
        if m_obs is None or not np.isfinite(m_obs):
            continue
        b_int = p['bands'][b]['intercept_median']
        M_pred = slope * np.log10(sigma_v) + b_int
        m_pred = M_pred + mu
        res[b] = float(m_obs - m_pred)
    if res:
        res['max_abs'] = float(np.max(np.abs(list(res.values()))))
        res['rms']     = float(np.sqrt(np.mean(np.square(list(res.values())[:3]))))
    return res


def is_consistent(mags_dict, z, sigma_v, max_dev_mag: float = 1.5) -> bool:
    """
    Quick gate: returns True if every band's residual is within max_dev_mag.
    Default 1.5 mag is generous (intrinsic FJ scatter is ~0.5 mag; cutout
    photometry adds noise; we mainly want to flag truly inconsistent objects).
    """
    s = consistency_score(mags_dict, z, sigma_v)
    if not s:
        return False
    return s['max_abs'] <= max_dev_mag


# Convenience: handle a numpy-array input for vectorized callers
def predict_sigma_v_array(mags_dict_arrays, z_array, bands: Iterable[str] = BANDS):
    """
    Vectorized predict_sigma_v_multiband.
    mags_dict_arrays: {band: 1-D array of apparent mags}
    z_array: 1-D array of redshifts (same length)
    """
    p = _load()
    slope = p['slope_fj']
    z = np.asarray(z_array)
    mu = np.array([distance_modulus(zi) for zi in z])
    stack = []
    for b in bands:
        if b not in mags_dict_arrays:
            continue
        m = np.asarray(mags_dict_arrays[b], dtype=float)
        if m.shape != z.shape:
            raise ValueError(f'shape mismatch: {b} {m.shape} vs z {z.shape}')
        M_abs = m - mu
        b_int = p['bands'][b]['intercept_median']
        log_sigma = (M_abs - b_int) / slope
        stack.append(10.0 ** log_sigma)
    if not stack:
        return np.full(len(z), np.nan)
    return np.median(np.vstack(stack), axis=0)
