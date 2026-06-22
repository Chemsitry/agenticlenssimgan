# v13_consistent — simulation outputs (catalog snapshot)

This folder holds the **catalog arrays** from Nate's v13 lens simulation, copied
into the repo so the catalog-vs-catalog checks (see `../catalogCheck.md`) can run
anywhere with zero new image generation.

## What "v13" means

There is no separate "v13" code. **v13 = the `../v9_consistent/` simulation
engine (`simulate_v9_consistent.py`) fed with v13 "unified" source scenes**
(JADES DR5 ellipticals across both GOODS fields + DESI calibration galaxies,
built by `../v9_consistent/prep_scenes_v12.py`). The output `metadata.json` still
carries the stale string `"version": "v9_consistent"`, but `params.scenes_source`
correctly reads `"v13 unified ..."`. **Trust the params, not the version string.**

## Where the data lives

The full run lives on CFS (too large for git):

    /global/cfs/projectdirs/deepsrch/natekv/v13_consistent/

| Kind | Files | Size each | In git? |
|------|-------|-----------|---------|
| Image cubes (N, 630, 630) | `images_F{band}.npy`, `galaxies_F{band}.npy`, `arcs_F{band}.npy`, plus `*_zoom` | ~4.6 GB | No — reference by CFS path |
| Catalog arrays (N,) | everything in `catalog/` below | ~23 KB | **Yes** |

- `images_*` = lens scene + simulated arc (the composite).
- `galaxies_*` = real JWST lens scene only (no arc).
- `arcs_*` = simulated lensed source only.
- `*_zoom` = same systems re-rendered on a tighter field of view. For every
  scalar catalog quantity the `_zoom` array is **identical** to the non-zoom one,
  so only the non-zoom copies are kept here.

## Catalog arrays in `catalog/` (N = 2911 systems)

All are 1-D float64 arrays of length 2911 (bool for the mask), aligned by index.

| File | Meaning | Units | min / median / max |
|------|---------|-------|--------------------|
| `theta_Es.npy` | Einstein radius of the SIE model | arcsec | 0.00 / 0.996 / **1.687** |
| `sigma_v.npy` | σ_v actually used in the SIE (clipped to [80,350]) | km/s | 80 / 256 / 350 |
| `sigma_v_F115W.npy` | photometry-derived σ_v from F115W (pre-clip) | km/s | 50.6 / 242.8 / 461.5 |
| `sigma_v_F150W.npy` | photometry-derived σ_v from F150W (pre-clip) | km/s | 50.3 / 237.8 / 440.4 |
| `sigma_v_F277W.npy` | photometry-derived σ_v from F277W (pre-clip) | km/s | 41.9 / 256.0 / 350.8 |
| `einstein_mass_msun.npy` | projected mass inside θ_E (Planck18) | M_⊙ | 0 / 3.19e11 / 1.13e12 |
| `photom_F115W.npy` | **lens-galaxy** apparent AB mag (60 px / 1.8″ aperture) | AB mag | 17.2 / 20.1 / 27.4 |
| `photom_F150W.npy` | lens-galaxy apparent AB mag | AB mag | 16.9 / 19.7 / 25.0 |
| `photom_F277W.npy` | lens-galaxy apparent AB mag | AB mag | 17.1 / 19.1 / 23.5 |
| `z_lens.npy` | lens redshift | — | 0.089 / 0.65 / 2.5 |
| `z_source.npy` | source redshift | — | 1.0 / 2.45 / 4.0 |
| `lensed.npy` | 1 = arc added, 0 = lens-only control | flag | 0 / 1 / 1 |
| `completed_mask.npy` | True = system finished rendering | bool | all True |

### Two facts that matter for the comparison

1. **`photom_F*` is LENS light only**, measured on the real scene cutout
   (`cutout_ab_mag`, 60 px aperture, 70–95 px sky annulus) — it is *not* the
   total system magnitude and does *not* include the arc. There is no
   `photom_F444W` (F444W is only the brightness anchor, not a calibration band).

2. **The arc/lens flux ratio is hard-coded, not lensing-derived.** The simulator
   anchors total arc flux to `target_ratio = 0.25 ×` the lens F444W flux, then
   applies per-band source colors. So any "implied arc/lens ratio" you read out
   of v13 is a *design assumption to be tested* against COWLS, not a physical
   prediction of the magnification.

## Loading

```python
import numpy as np
from pathlib import Path

cat = Path("v13_consistent/catalog")
theta_E = np.load(cat / "theta_Es.npy")        # arcsec
sigma_v = np.load(cat / "sigma_v.npy")          # km/s
m_lens  = {b: np.load(cat / f"photom_{b}.npy")  # lens AB mag
           for b in ("F115W", "F150W", "F277W")}
```

A tidy one-row-per-system table is also exported to `v13_catalog.csv` by
`build_catalog_csv.py` (run it to regenerate).
