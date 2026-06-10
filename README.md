# JWST Gravitational Lens Simulation Pipeline

Multi-band JWST strong gravitational lens simulation pipeline for training ML classifiers. Uses real COSMOS-Web DR0.5 survey data (backgrounds, PSFs, noise) combined with physics-based lensing models calibrated against the COWLS II lens catalogue (Mahler et al. 2025).

## Overview

Strong gravitational lensing occurs when a massive foreground galaxy (the "lens") bends light from a more distant background galaxy (the "source"), producing arcs, rings, or multiple images. This pipeline simulates realistic multi-band images of these systems for training automated lens-finding classifiers.

Each simulated image contains:
- A **lens galaxy** (elliptical Sersic profile) with redshift-dependent SED colors
- A **lensed source** (star-forming Sersic profile) distorted by an SIE+shear mass model
- **Real sky background** extracted from COSMOS-Web DR0.5 deep mosaics
- **Empirical PSF** convolution and **Poisson noise** calibrated per band
- Four NIRCam bands: **F115W** (1.15 um), **F150W** (1.50 um), **F277W** (2.77 um), **F444W** (4.44 um)

Output images are 630x630 pixels at 0.03"/pix (18.9" field of view), matching the panel size in COWLS II Figure 1.

## Pipeline Versions

| Version | Description | Image Size | Key Changes |
|---------|-------------|-----------|-------------|
| v1 | Single-band F115W prototype | 125x125 | Gaussian PSF, SIS lens, jw01810 data |
| v2 | Multi-band with SLACS params | 125x125 | Empirical PSF, SIE+shear, COSMOS-Web i2d files |
| v3 | COWLS-calibrated distributions | 125x125 | DR0.5 deep mosaics, Lyman-break SEDs |
| v4 | Validated against real COWLS II data | 630x630 | Peak-matched amplitudes, recalibrated SEDs, 18.9" FoV |
| v5 | SimGAN-refined images | 630x630 | GAN refinement for realistic galaxy morphology |
| v6 | Real galaxy stamps | 630x630 | INTERPOL real-galaxy stamps replace Sersic profiles |
| v7 | VELA sources | 630x630 | 20,421 VELA hydrodynamic-simulation stamps as source light |
| **v8 (current)** | **Single-scene architecture** | **630x630** | **One real JWST cutout per sample; only the arcs are simulated** |
| v9_consistent (newest, experimental) | Physically consistent lens mass | 630x630 | sigma_v derived from lens photometry via Faber-Jackson; DESI sigma_v where available; stratified theta_E; lives in `v9_consistent/` |

## What's New in v6-v9 (merged 2026-06-09)

- **v8 architectural rethink** (`simulate_v8.py`): stop stitching separate lens + background cutouts. Each sample is ONE real 630x630 JWST cutout ("scene"); the central lens galaxy is untouched real pixels and only the lensed source is simulated and added. Eliminates prior artifacts (force fields, chunky halos, noise doubling, visible boxes).
- **v9_consistent** (`v9_consistent/`): in v8 the lens cutout's appearance had no relation to the sigma_v used in the SIE mass model. v9 derives sigma_v from the cutout's F115W/F150W/F277W photometry via a Faber-Jackson relation calibrated on DESI x JWST galaxies, with an inter-band consistency rejection gate and stratified theta_E sampling.
- **Output naming (v8+):** `sources_{band}.npy` was renamed `arcs_{band}.npy`. Per-band outputs: `images_{band}.npy` (composite), `galaxies_{band}.npy` (real scene only), `arcs_{band}.npy` (simulated arcs only).

### Bug fixes (v6-v8)

| Fix | Bug |
|---|---|
| v6 stamp augmentation | Rotation/flip was drawn per band, so one galaxy had different orientations in each of the 4 bands (non-physical). Now decided once per system. |
| v8 source SED | `starforming_color_ratios()` was computed but never applied; arcs had identical flux in all bands. Arcs now show Lyman-break dropout and Balmer break. |
| v8 scene pool | Stars (JWST diffraction spikes) passed the concentration filter and appeared as central "lens galaxies". Removed via a compactness cut (flux(r<3)/flux(r<15) in F444W, threshold 0.35). |
| v8 duplicates | Non-lensed images sampled scenes with replacement, producing pixel-identical duplicates. Scenes are now assigned without replacement. |

## What Was New in v4

- **SED color ratios recalibrated** against PyAutoLens-decomposed photometry from the COWLS II catalogue (440 real lens systems). Our `elliptical_color_ratios()` and `starforming_color_ratios()` functions now match real data within the observed scatter.
- **Peak-matched amplitude calibration** validated against a real COWLS II lens (Lens E), replacing the unconstrained stellar mass formula from v3.
- **630x630 px images** (18.9" FoV) matching the COWLS II paper's panel size.
- **Fixed arc/lens ratio** (0.25) ensuring visible arcs across all parameter combinations.
- **Source SED with Balmer break** — rest-frame optical break boost produces realistic red colors at z > 1, matching observed source photometry.

## Setup

### Prerequisites

- Python 3.10+ (tested with 3.14)
- macOS (Apple Silicon M3 tested) or Linux
- ~170 GB disk for raw COSMOS-Web mosaics (or use pre-extracted data)

### Installation

```bash
# Clone and enter repo
git clone <repo-url>
cd "data prep"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy astropy lenstronomy matplotlib

# For GAN training (v5), also install:
pip install torch torchvision pandas
```

### Data

Large data files are excluded from git (see `.gitignore`). You need:

| Directory | Contents | Size | How to Get |
|-----------|----------|------|------------|
| `raw_data/1727_mosaic/` | COSMOS-Web DR0.5 full mosaics (4 bands) | ~170 GB | MAST archive, program 1727 |
| `prepped_mosaic_630/` | Extracted backgrounds (2000/band) + PSFs | ~13 GB | Run `prep_mosaic.py` |
| `prepped_mosaic_224/` | Smaller backgrounds + real galaxy stamps | ~4 GB | Run `prep_mosaic.py --size 224` |
| `output/` | Generated simulation outputs | varies | Run `simulate_v4.py` |

## Usage

### Step 1: Prep Data

Extract backgrounds and PSFs from the DR0.5 mosaics:

```bash
.venv/bin/python3 prep_mosaic.py
```

This produces `prepped_mosaic_630/` with per-band subdirectories containing:
- `backgrounds.npy` — 2000 random 630x630 sky patches (float32)
- `psf_median.npy` — Median-stacked empirical PSF (63x63)
- `psf_stars.npy` — 200 individual PSF star cutouts (63x63)

### Step 2: Generate Simulations

```bash
# Generate 10 test images (quick check)
.venv/bin/python3 simulate_v4.py

# Generate full training dataset
.venv/bin/python3 simulate_v4.py --n 2000

# Custom: 5000 images with specific seed
.venv/bin/python3 simulate_v4.py --n 5000 --seed 123
```

Output in `output/v4/`:

| File | Shape | Description |
|------|-------|-------------|
| `images_F115W.npy` | (N, 630, 630) | Simulated images per band |
| `images_F150W.npy` | (N, 630, 630) | |
| `images_F277W.npy` | (N, 630, 630) | |
| `images_F444W.npy` | (N, 630, 630) | |
| `sources_F115W.npy` | (N, 630, 630) | Source-only (ground truth) per band |
| `lensed.npy` | (N,) | Binary labels: 1=lensed, 0=non-lensed |
| `theta_Es.npy` | (N,) | Einstein radius in arcsec |
| `z_lens.npy` | (N,) | Lens redshift |
| `z_source.npy` | (N,) | Source redshift |
| `masses.npy` | (N,) | log10(halo mass / M_sun) |
| `metadata.json` | — | Full configuration and parameter distributions |
| `preview_*.png` | — | RGB composite grid for visual inspection |

Half the images are lensed (lens + arcs + background), half are non-lensed (lens galaxy + background only).

### Step 3: Validate (Optional)

Compare simulated SEDs against real COWLS II photometry:

```bash
# SED comparison plot: real COWLS II vs our model
.venv/bin/python3 plot_real_seds.py

# Single-band F115W validation against a real COWLS II lens
.venv/bin/python3 validate_f115w.py
```

### Step 4: GAN Refinement (v5, Experimental)

Refine simulated images using a SimGAN trained on real galaxy morphology:

```bash
# Prepare training data (center crops of sims + galaxy-centered crops of real backgrounds)
.venv/bin/python3 prep_gan_data.py

# Train SimGAN refiner (200 epochs, ~90 min on M3)
.venv/bin/python3 train_simgan.py --epochs 200 --batch 8 --bands all --out_dir output/v5

# Apply trained refiner to create v5 dataset
.venv/bin/python3 refine_dataset.py --input output/v4 --output output/v5
```

## Key Files

### Simulation Pipeline

| File | Purpose |
|------|---------|
| `simulate_v8.py` | **Main simulation pipeline** (v8) — real scene + simulated arcs |
| `v9_consistent/simulate_v9_consistent.py` | Newest pipeline (v9) — photometry-consistent lens mass |
| `prep_scenes_v8.py` / `prep_lenses_v8.py` | Build the v8 real-scene pool (with star/compactness filtering) |
| `prep_vela_v7.py` | Extract VELA hydrodynamic-simulation source stamps |
| `simulate_v7.py` / `simulate_v6.py` / `simulate_v4.py` / `simulate_v3.py` | Previous pipelines (preserved) |
| `prep_mosaic.py` | Extract backgrounds/PSFs from DR0.5 mosaics |
| `validate_f115w.py` | Single-band validation against real COWLS II Lens E |

### Analysis & Visualization

| File | Purpose |
|------|---------|
| `plot_seds.py` | Plot simulated SED color ratios (lens + source populations) |
| `plot_real_seds.py` | Compare real COWLS II SEDs vs simulated model |
| `cowls_catalogue.csv` | COWLS II catalogue: 440 lens candidates with 4-band photometry |
| `COWLS2.pdf` | Reference paper (Mahler et al. 2025) |

### GAN Refinement (v5)

| File | Purpose |
|------|---------|
| `prep_gan_data.py` | Prepare training data for SimGAN |
| `train_simgan.py` | SimGAN training: refiner + PatchGAN discriminator |
| `refine_dataset.py` | Apply trained refiner to create refined dataset |

### Diagnostic GAN (sim-vs-real feedback)

| File | Purpose |
|------|---------|
| `gan_plan.md` | Staged plan: train a discriminator to find where the simulator disagrees with real JWST data |
| `gan/` | Diagnostic GAN code (data prep, audits, PCA baseline, slurm scripts) |

### ML Training

| File | Purpose |
|------|---------|
| `train_cvae.py` | Conditional VAE for lens image generation (PyTorch) |
| `eval_generative.ipynb` | cVAE evaluation: power spectra, reconstructions, t-SNE |

## Physics Parameters (v4)

All parameter distributions are calibrated against the COWLS survey (Nightingale et al. 2025) and COWLS II (Mahler et al. 2025):

| Parameter | Distribution | Range | Description |
|-----------|-------------|-------|-------------|
| z_lens | TruncNorm(0.7, 0.4) | [0.05, 2.5] | Lens redshift |
| z_source | TruncNorm(2.5, 1.5) | [0.5, 7.0] | Source redshift |
| sigma_v | TruncNorm(180, 50) km/s | [80, 350] | Velocity dispersion |
| theta_E | Derived from sigma_v | [0.5, 1.5] arcsec | Einstein radius (post-filter) |
| log10(M_halo) | Uniform | [11.0, 13.0] | Halo mass |
| UV slope | Normal(-0.5, 1.0) | [-2.5, 1.5] | Source UV spectral slope |

### SED Color Ratios

Lens galaxies (elliptical, recalibrated against COWLS II):
```
F150W/F115W = 1.68 + 0.09 * z_lens
F277W/F115W = 8.79 + 8.40 * z_lens
F444W/F115W = 0.42 + 18.11 * z_lens
```

Source galaxies include Lyman-break dropout suppression and Balmer/4000A break boost for rest-frame optical wavelengths.

## References

- Mahler et al. 2025, MNRAS 544, L8 — COWLS II: 17 spectacular lenses
- Nightingale et al. 2025, MNRAS 543, 203 — COWLS I: automated lens search
- Casey et al. 2023, ApJ 954, 31 — COSMOS-Web survey design
- Shuntov et al. 2025, A&A 695, A20 — COSMOS-Web photometric catalogue
