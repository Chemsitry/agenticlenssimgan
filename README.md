# JWST Gravitational Lens Simulation — v13 + diagnostic GAN

This repo has two jobs:

1. **Hold the current lens simulation (v13)** and the engine that produces it.
2. **Check how well v13 matches reality**, so Nate can see where his simulator is
   off — first via fast *catalog-vs-catalog* checks (`catalogCheck.md`), later via
   a *diagnostic GAN* (`gan_plan.md`).

Strong gravitational lensing is when a massive foreground galaxy (the **lens**)
bends light from a more distant background galaxy (the **source**) into arcs or
rings. v13 simulates multi-band JWST/NIRCam images of such systems (F115W, F150W,
F277W, F444W) at 630×630 px, 0.03″/px (18.9″ field of view).

## What "v13" is

There is no standalone "v13" codebase. **v13 = the `v9_consistent/` engine
(`simulate_v9_consistent.py`) run on v13 "unified" source scenes** — JADES DR5
ellipticals (both GOODS fields) + DESI calibration galaxies, prepared by
`v9_consistent/prep_scenes_v12.py`. The σ_v that sets each lens's bending
strength is derived from the lens galaxy's own photometry through a
Faber–Jackson relation (so lens brightness and lens mass are physically tied).

The 2,911-system **output** lives on CFS and is summarized in this repo:

| Location | Contents |
|----------|----------|
| `/global/cfs/projectdirs/deepsrch/natekv/v13_consistent/` | Full run: 4.6 GB image cubes + catalog arrays |
| `v13_consistent/` (in-repo) | Small catalog arrays (~23 KB each) + `v13_catalog.csv` + docs |

See `v13_consistent/README.md` for the array dictionary and important caveats
(e.g. `photom_*` is lens-only AB mag; the arc/lens flux ratio is hard-coded to
0.25, not lensing-derived).

## Repo layout

| Path | Purpose |
|------|---------|
| `v13_consistent/` | v13 catalog snapshot + `build_catalog_csv.py` |
| `v9_consistent/` | The simulation engine + Faber–Jackson calibration that makes v13 |
| `catalogCheck.md` | **Plan: quantitative catalog-vs-real comparison (do today, zero new images)** |
| `catalog_checks.py` | **Runnable implementation of the 3 catalog checks** (numpy core; optional scipy/matplotlib) |
| `referenceData.md` | Where each comparison number comes from (COWLS columns, published σ_v samples, which COWLS2 figure) |
| `gan_plan.md` | Plan: diagnostic GAN that learns where sim ≠ real JWST |
| `gan/` | Diagnostic GAN code (data prep, baselines, slurm) |
| `gan_architecture_proposals.html` | v13-era proposals for the diagnostic comparison |
| `cowls_catalogue.csv` | Real comparison catalog: 385 COWLS lenses w/ θ_E + 4-band lens & source mags + magnification |
| `COWLS2.pdf` | COWLS II paper (Mahler et al. 2025) |
| `COWLS2im.jpeg` | COWLS II **Figure 1** — the 17-spectacular-lenses montage (visual reference only) |
| `slides/` | Group-meeting slides |

## The engine (`v9_consistent/`)

| File | Purpose |
|------|---------|
| `simulate_v9_consistent.py` | Main pipeline: real scene + simulated arc; σ_v from Faber–Jackson |
| `prep_scenes_v12.py` | Builds the v13 unified source-scene pool |
| `cutout_photometry.py` | Lens-cutout sim-units → AB mag per band |
| `fit_calibration.py`, `calibration.py`, `fj_params.json` | Faber–Jackson fit + σ_v(mags, z) |
| `export_csvs.py` | Human-readable galaxy catalogs (calibration + lens scenes) |
| `resume_simulation.py` | Restart a partial run |

## Validating v13 against reality

The source galaxies (VELA/JADES) and the comparison lenses (COWLS) come from
**different datasets**, so we compare **distributions and scaling relations**,
not image-to-image. Two stages:

- **Now — catalog vs catalog (`catalogCheck.md`).** Compare v13's Einstein radii,
  arc/lens flux ratios, and σ_v / Einstein-mass against COWLS and published lens
  samples (SLACS/BELLS/SL2S). Zero new images required.
- **Later — diagnostic GAN (`gan_plan.md`).** Train a discriminator to find the
  image features where v13 still disagrees with real JWST cutouts.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy astropy lenstronomy matplotlib pandas
```

`v13_consistent/build_catalog_csv.py` needs only numpy + the standard library and
runs in the bare environment.

## References

- Mahler et al. 2025, MNRAS 544, L8 — COWLS II: 17 spectacular lenses
- Nightingale et al. 2025, MNRAS 543, 203 — COWLS I: automated lens search
- Casey et al. 2023, ApJ 954, 31 — COSMOS-Web survey design
- Shuntov et al. 2025, A&A 695, A20 — COSMOS-Web photometric catalogue
