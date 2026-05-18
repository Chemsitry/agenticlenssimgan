# data prep v9 — physically consistent lens (σ_v ↔ lens light)

**Status:** experimental. Sits *next to* `~/Desktop/data prep/`, never modifies it.

## What v8 does today
For each simulated lens system:
- **Lens light** = a real COSMOS-Web JWST cutout (`prepped_mosaic_630/lenses_v8/`)
- **Lens mass** = SIE with `σ_v` drawn from `TruncNorm(180, 50, [80, 350])` — **independent** of the cutout
- **Source light** = VELA stamp (or analytic Sérsic with `--sersic`)

The inconsistency: the *appearance* of the lens galaxy (brightness, color) has no relationship to the *bending strength* (σ_v) we plug into the SIE. A faint dwarf cutout could be paired with σ=320 km/s, or a giant cD with σ=90.

## What v9 fixes
For each cutout, read off its F115W/F150W/F277W apparent magnitude, then compute the σ_v consistent with the empirical Faber–Jackson relation (anchored on literature + fine-tuned with our 5-galaxy DESI×JWST sample at `~/Desktop/desi_jwst_dev/cache/sample_final.parquet`). Use that σ_v in the SIE.

Bonus: reject cutouts whose brightness is wildly inconsistent with any plausible σ_v (a sanity gate).

## Layout
```
data prep v9_consistent/
  README.md
  cutout_photometry.py     # sim-units cutout → AB mag per band
  fit_calibration.py       # fits FJ on our 5 galaxies, anchored to literature slope (L∝σ⁴)
  calibration.py           # predict_sigma_v(mags, z), consistency_score(...)
  fj_params.json           # output of fit_calibration.py
  test_calibration.py      # sanity-checks on the 5-galaxy sample
```

## Calibration choice (locked in)
- **Slope**: fixed to the classical Faber-Jackson value `L ∝ σ⁴` → in (M_abs vs log σ_v) space slope = -10. Robust at small N.
- **Intercept**: fit on our 5 DESI×JWST galaxies, per band (F115W, F150W, F277W).
- **Distance modulus**: Planck18 cosmology.
- **K-correction**: ignored at first pass (small for NIR at z<0.5).

## Plan of attack
1. Build calibration + photometry modules (done in this commit).
2. Smoke-test against the 5-galaxy sample.
3. Later: build a `simulate_v9_consistent.py` (copy of v8 simulate) that uses the calibration to set σ_v per cutout.

`simulate_v8.py` in `~/Desktop/data prep/` is **never** modified.
