# referenceData.md — what to compare the v13 catalog checks against

This is the companion to `catalogCheck.md`. It answers one question: **for each
check, where does the *real* data come from, exactly?** (file, column, paper) —
and it clears up the COWLS2 image question.

## TL;DR — comparison data per check

| Check | v13 side | Real side (what to compare to) | Where it lives |
|-------|----------|--------------------------------|----------------|
| **1. Einstein radius** | `v13_consistent/catalog/theta_Es.npy` | COWLS `einstein_radius` column | `cowls_catalogue.csv` (in repo) |
| **2. arc/lens flux ratio** | hard-coded 0.25 @ F444W (+ optional cube photometry) | COWLS `F{band}_lens_magnitude_ab`, `F{band}_source_magnitude_ab`, `F{band}_magnification` | `cowls_catalogue.csv` (in repo) |
| **3. σ_v & Einstein mass** | `sigma_v.npy`, `einstein_mass_msun.npy` | **published spectroscopic lens samples** (SLACS / BELLS / SL2S) — *not* in COWLS | external; see below |

The single script `catalog_checks.py` already wires up Checks 1 and 2 from the
in-repo CSV. Check 3's σ_v comparison currently uses literature *ranges*
hard-coded in the script; to make it a real number-vs-number test you need to
download one of the published tables described in §3.

---

## 1. The COWLS catalogue (`cowls_catalogue.csv`) — Checks 1 & 2

**What it is.** The official public catalogue from the COSMOS-Web Lens Survey.
It is a verbatim copy of `catalogue.csv` from the COWLS data release:

- **GitHub:** https://github.com/Jammy2211/COWLS_COSMOS_Web_Lens_Survey
- **Papers:** COWLS I — Nightingale et al. 2025 (arXiv:2503.08777, the search +
  lens models); COWLS II — Mahler et al. 2025 (arXiv:2503.08782, the 17
  spectacular lenses); COWLS III — Hogg et al. 2025 (abundance vs predictions).
- **Contents:** 439 visually-inspected candidate lenses in the 0.54 deg² field;
  our file has 440 rows (one is the index header) and **385 with a finite
  `einstein_radius`** (those with a successful PyAutoLens model).

**How the numbers were produced (so you know what you're comparing to).** Each
candidate was modelled with **PyAutoLens**: a Multi-Gaussian-Expansion lens
light model, a **Singular Isothermal Ellipsoid (SIE) + external shear** mass
model (the *same* mass family v13 uses), and an adaptive-Voronoi source
reconstruction. A "primary" band (where the source is clearest) sets the mass
model, which is then applied to the other bands.

**Columns we use:**

| Column | Meaning | Used by |
|--------|---------|---------|
| `einstein_radius` | θ_E (″) from the SIE fit | Check 1 |
| `F{band}_lens_magnitude_ab` | lens-galaxy AB mag (4 bands) | Check 2 |
| `F{band}_source_magnitude_ab` | **delensed (intrinsic)** source AB mag | Check 2 |
| `F{band}_magnification` | total magnification μ | Check 2 |
| `lens_spec_z`, `lens_cw_photo_z_med` | lens redshift | context |
| `lens_cw_stmass_med` | lens log₁₀ stellar mass | context |

**Important — "delensed" resolves the Check 2 ambiguity.** The COWLS docs state
`*_source_magnitude_ab` are *delensed* (intrinsic) source magnitudes. So the
**observed arc** brightness must put the magnification back:
`m_arc = m_source − 2.5·log₁₀(μ)`. This is interpretation **(A)** in
`catalogCheck.md`, and it is the one `catalog_checks.py` uses by default. (Run
with `--observed` only to see how wrong the alternative reading would be.)

**What COWLS does NOT contain:** there is **no velocity dispersion (σ_v)** and
**no source redshift** column. That has two consequences:
- Check 3's σ_v comparison cannot use COWLS — it needs external samples (§3).
- We cannot compute a COWLS *Einstein mass* directly, because M_E needs both the
  lens and source redshifts (distance ratio D_d D_s / D_ds), and the source z is
  absent here. (The 17 lenses in COWLS II Table 1 do list some source z's, but
  the full catalogue.csv does not.)

---

## 2. The COWLS2 image — which figure is it?

You asked which image the COWLS2 PDF refers to. Answer:

- **`COWLS2im.jpeg` in this repo is Figure 1 of COWLS II (Mahler et al. 2025).**
  The paper says verbatim: *"We present in Figure 1 the 17 most spectacular
  systems..."*. It is the montage of **17 colour cutouts** labelled with COSJ
  names (credit: Gozaliasl & Virolainen). The lenses in it (e.g.
  `COSJ100121+022740`, `COSJ100047+015023`, `COSJ100028+021928`) are rows in
  `cowls_catalogue.csv`.
- COWLS II **Figure 2** is a *different* montage of the same 17 lenses labelled
  with ID letters (A–Q); **Table 1** lists their θ_E and redshifts.

**Use it correctly:** Figure 1 is a **visual reference, not the data for the
catalog checks.** The catalog checks compare *numbers* from
`cowls_catalogue.csv`. Figure 1 is useful for (a) sanity-checking that real
arcs/rings look like what v13 produces (colours, arc placement), and (b) the
*later* diagnostic-GAN stage (`gan_plan.md`), which compares images. The 17
"spectacular" lenses are also a **biased, best-case subset** (large rings, clean
deblending), so never treat them as a representative sample for distributions —
use the full 385-row catalogue for that.

---

## 3. Published lens samples — Check 3 (σ_v & Einstein mass)

Because COWLS has no σ_v, the velocity-dispersion comparison must come from
spectroscopic strong-lens samples. These are the standard ones; each publishes a
table with σ_v (from spectroscopy), lens/source redshift, and θ_E — exactly what
Check 3 needs.

| Sample | Reference | What to grab | Where |
|--------|-----------|--------------|-------|
| **SLACS** *(used)* | Bolton et al. 2008, ApJ 682, 964 | σ_v (SDSS), z_lens, z_source, θ_E | VizieR `J/ApJ/682/964` (table4 ⋈ table5) — **auto-fetched** by `fetch_published_sigma.py` |
| **BELLS** | Brownstein et al. 2012, ApJ 744, 41 | σ_v, z_lens, z_source, θ_E | VizieR / paper Table |
| **SL2S** | Sonnenfeld et al. 2013, ApJ 777, 98 (+2015) | σ_v, z_lens, z_source, θ_E | VizieR / paper Table |

**Caveat on what σ_v means.** Published σ_v are *spectroscopic stellar* velocity
dispersions (aperture-corrected). v13's σ_v is an SIE *model* dispersion derived
from lens photometry via Faber–Jackson. For ellipticals these agree to ~5–10%,
but they are not identical definitions — don't over-read a small median offset.

### The Check 3 data file (`published_sigma.csv`) — already built
Run once on a login node (needs internet):
```bash
python3 fetch_published_sigma.py      # downloads VizieR J/ApJ/682/964 -> published_sigma.csv
```
This writes **62 grade-A SLACS lenses** (σ median 245 km/s [160–396], θ_E median
1.19″, z_lens 0.06–0.36) with columns:
```
sample,name,sigma_v_kms,sigma_v_err,z_lens,z_source,theta_E_arcsec,grade
```
`catalog_checks.py` Check 3 picks this file up automatically and then:
- runs a real two-sample KS test, v13 vs SLACS — current result **D=0.206,
  p≈0.01** (v13 median 265 vs SLACS 245: v13 is shifted high and truncated at
  the 350 clip while SLACS reaches 396);
- overlays the SLACS Einstein masses (computed from z_lens, z_source, θ_E via
  `astropy.cosmology.Planck18`) on the M_E–σ_v panel. SLACS sits slightly below
  v13 at fixed σ_v — partly real, partly geometry (SLACS lenses are low-z).

**To extend:** append BELLS/SL2S rows with the same columns (or add fetchers) and
Check 3 will include them automatically; with `z_source` present every sample
also gets an Einstein mass for the M_E comparison.

**σ_v definition caveat (again):** SLACS σ_v here is the raw SDSS-fibre value
(uncorrected for aperture); v13's is an SIE/Faber–Jackson model dispersion. Close
for ellipticals, not identical — read the KS result as "distributions differ
modestly," not "v13 is wrong by D."

---

## 4. Quick reference — file locations

| Data | Path |
|------|------|
| v13 catalog arrays | `v13_consistent/catalog/*.npy` (+ `v13_consistent/v13_catalog.csv`) |
| v13 full image cubes (4.6 GB each) | `/global/cfs/projectdirs/deepsrch/natekv/v13_consistent/{images,galaxies,arcs}_F{band}.npy` |
| COWLS catalogue (numbers) | `cowls_catalogue.csv` |
| COWLS II Figure 1 (17-lens montage) | `COWLS2im.jpeg` |
| COWLS II paper | `COWLS2.pdf` |
| Published σ_v sample (SLACS) | `published_sigma.csv` (built by `fetch_published_sigma.py` from VizieR `J/ApJ/682/964`) |
