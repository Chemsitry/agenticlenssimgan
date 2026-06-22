# catalogCheck.md — quantitative catalog-vs-catalog validation of v13

**Goal:** in one sitting, with **zero new image generation**, produce a small,
defensible, quantitative answer to the question *"where is v13 off from real
lenses?"* — and hand Nate concrete numbers + one or two plots per check.

**Audience note (read me first):** this plan is written for someone newer to ML
/ stats. Every check has a plain-English "what & why", then the exact math, then
runnable code, then "how to read the result". A glossary is at the bottom — terms
in **bold-italic** like ***CDF*** are defined there.

---

## 0. The core idea: compare *distributions*, not images

Our simulated sources (VELA / JADES) and our real comparison lenses (**COWLS**)
come from **different datasets and different selection rules**. So we must *not*
compare image-to-image or expect a 1-to-1 match. Instead we compare:

- **Marginal distributions** — e.g. "what does the spread of Einstein radii look
  like in v13 vs in COWLS?"
- **Scaling relations** — physical laws like θ_E ∝ σ_v² that should hold
  regardless of which survey you drew the galaxies from.

This is what "**survey-independent**" means here: a scaling relation (a slope and
normalization between two physical quantities) is a property of physics, so it
should match across surveys even when the raw populations differ. A *raw count*
("how many lenses with θ_E > 1.5″") is survey-dependent and must be quoted
carefully (see the selection-function caveat in each check).

### Inputs

> **Where the real comparison data comes from** (every file, column, and paper)
> is documented in `referenceData.md`. The runnable implementation of all three
> checks is `catalog_checks.py`.

**v13 (simulation), in `v13_consistent/catalog/`** — 1-D arrays, length 2911,
aligned by index. Filter to `lensed == 1` (2500 systems) for lens comparisons;
the other 411 are lens-only controls with `theta_E = 0`.

| Array | Symbol | Note |
|-------|--------|------|
| `theta_Es.npy` | θ_E (″) | SIE Einstein radius |
| `sigma_v.npy` | σ_v (km/s) | value used in SIE, **clipped to [80, 350]** |
| `sigma_v_F{115,150,277}.npy` | σ_v,phot | Faber–Jackson prediction per band, **un-clipped** |
| `einstein_mass_msun.npy` | M_E (M_⊙) | projected mass inside θ_E |
| `photom_F{115,150,277}.npy` | m_lens (AB) | **lens-galaxy** apparent mag (no arc, no F444W) |
| `z_lens.npy`, `z_source.npy` | z_d, z_s | redshifts |

**Real (COWLS), in `cowls_catalogue.csv`** — 385 systems, one row each.

| Column | Meaning |
|--------|---------|
| `einstein_radius` | θ_E (″) from PyAutoLens fit |
| `F{band}_lens_magnitude_ab` | lens-galaxy AB mag (4 bands) |
| `F{band}_source_magnitude_ab` | source AB mag (see Check 2 — intrinsic vs observed!) |
| `F{band}_magnification` | total magnification μ (4 bands) |
| `lens_spec_z`, `lens_cw_photo_z_med` | lens redshift |
| `lens_cw_stmass_med` | lens log10 stellar mass (M_⊙) |

### Deliverables (what "done" looks like)

1. One script, `catalog_checks.py` (repo root), that loads both catalogs and runs
   all three checks.
2. Three figures in `output/catalog_checks/`: `theta_E.png`, `flux_ratio.png`,
   `sigma_mass.png`.
3. A 1-page results table (printed + saved as `catalog_checks_summary.csv`) with,
   per check: the headline number, a ***KS*** statistic + p-value, and a
   one-line interpretation.

### Shared setup (loader skeleton)

```python
import numpy as np, csv
from pathlib import Path

CAT = Path("v13_consistent/catalog")
def load(name): return np.load(CAT / f"{name}.npy")

lensed = load("lensed") > 0.5            # boolean mask, 2500 True
thetaE_sim  = load("theta_Es")[lensed]   # arcsec
sigma_sim   = load("sigma_v")[lensed]    # km/s (clipped)
mass_sim    = load("einstein_mass_msun")[lensed]
mlens_sim   = {b: load(f"photom_{b}")[lensed] for b in ("F115W","F150W","F277W")}

rows = list(csv.DictReader(open("cowls_catalogue.csv")))
def col(name):
    out = []
    for r in rows:
        try: out.append(float(r[name]))
        except (ValueError, KeyError): out.append(np.nan)
    return np.array(out)
```

> Stats need `scipy` (`ks_2samp`). The bare login env has Python 3.6 + numpy
> only; load a conda module or `.venv` with `scipy`, `pandas`, `matplotlib`
> before running the full script. The numbers quoted below were computed with
> numpy alone, so they reproduce even without scipy.

---

## 1. Einstein radius: θ_E(v13) vs COWLS `einstein_radius`

### What & why (plain English)
The **Einstein radius** θ_E is the angular size of the ring/arc — it is the single
best one-number summary of "how strong" a lens is (bigger θ_E = more mass inside
the ring). If v13's θ_E distribution is shifted or truncated relative to real
lenses, the simulator is making systematically wrong-strength lenses. You already
noticed v13 "tops out at ~1.69″"; this check **quantifies how much of the real
high-θ_E tail that ceiling throws away**, and also checks the *shape* of the whole
distribution, not just the tail.

### Method
1. Build `thetaE_sim` (lensed only, drop θ_E = 0) and `thetaE_cowls =
   col("einstein_radius")` (drop NaN and ≤ 0).
2. **Fairness step — match the ranges.** v13 only simulates θ_E ∈ [0.5, 2.5] by
   design (small lenses make no visible arc). COWLS contains many θ_E < 0.5
   (median 0.505 over all 385). So make the headline comparison on the
   **overlapping subsample θ_E ≥ 0.5** (COWLS n = 195), and *separately* report
   the full-sample tail fractions for context.
3. Compare three things:
   - **Tail fraction** beyond v13's ceiling: `(thetaE_cowls > 1.687).mean()`.
   - **Whole-shape**: overlay the two ***CDFs*** and run a two-sample ***KS test***
     (`scipy.stats.ks_2samp`) on the θ_E ≥ 0.5 subsamples.
   - **Quantiles**: median, p90, max of each.

### Math
The SIE Einstein radius depends only on σ_v and a distance ratio:
```
θ_E = 4π (σ_v / c)²  (D_ds / D_s)          [radians → ×206265 for arcsec]
```
So a ceiling on σ_v (v13 clips at 350 km/s) directly imposes a ceiling on θ_E.
This is the mechanical reason the tail is missing — see Check 3.

### Code
```python
te_sim = thetaE_sim[thetaE_sim > 0]
te_cow = col("einstein_radius"); te_cow = te_cow[np.isfinite(te_cow) & (te_cow > 0)]
sub = te_cow[te_cow >= 0.5]                       # overlapping range
ceiling = 1.687
print("missing tail  : %.1f%% of COWLS(>0.5) exceed v13 ceiling" % (100*(sub>ceiling).mean()))
print("median  sim/real(>0.5): %.3f / %.3f" % (np.median(te_sim), np.median(sub)))
# from scipy.stats import ks_2samp; D,p = ks_2samp(te_sim, sub)
```
Plot: two CDFs on one axis (step plot), a vertical dashed line at 1.687″, shade
the COWLS area to its right (that shaded area = "the tail v13 can't make").

### Expected result (precomputed, numpy-only)
| Quantity | v13 (lensed) | COWLS (θ_E ≥ 0.5) |
|---|---|---|
| n | 2500 | 195 |
| median θ_E | **1.10″** | **0.73″** |
| p90 θ_E | 1.58″ | — |
| max θ_E | **1.687″** | 2.25″ |
| fraction > 1.687″ | 0% (by construction) | **4.6% (9 systems)** |
| fraction > 1.5″ | small | 8.2% (16) |

### How to read it (two findings, both actionable)
- **Truncation (the one you flagged):** ~**4.6%** of comparable real lenses lie
  beyond v13's ceiling — small in count but it is exactly the **massive
  group/cluster-scale regime** that is scientifically most interesting. The fix
  is in Check 3 (raise the σ_v clip).
- **Shape shift (a second, separate problem):** v13's median θ_E (1.10″) is
  *higher* than real (0.73″). v13 over-produces *large* arcs and under-produces
  *small* ones. Expect the KS test to reject "same distribution" strongly. This
  points at the σ_v / stratified-θ_E sampling, not just the clip.

### Caveat
COWLS is a visually-graded JWST-selected sample with its own (unmodelled)
selection toward findable arcs; v13 imposes θ_E ∈ [0.5, 2.5]. The tail-fraction
number is therefore a *lower bound* on what a volume-complete survey would show.
Quote it as "of COWLS lenses with θ_E ≥ 0.5″", never as an absolute sky rate.

---

## 2. Implied arc/lens flux ratio vs COWLS lens + source mags

### What & why (plain English)
"How bright is the arc compared to the lens galaxy it sits on?" is the
***flux ratio*** = (arc flux) / (lens flux). It controls whether arcs are visible
and how a classifier learns to find them. **In v13 this ratio is not a physical
prediction — it is hard-coded.** The simulator sets total arc flux to
`target_ratio = 0.25 ×` the lens's F444W flux, then recolors the arc per band
(`simulate_v9_consistent.py`). This check tests whether that single assumption
(0.25, anchored in the reddest band) matches what real lenses actually show.

### The subtlety you must resolve first: intrinsic vs observed source mag
COWLS gives a `source_magnitude` **and** a `magnification` per band. Two readings:
- **(A) intrinsic source** (de-magnified). Then observed arc flux needs the
  magnification put back: `m_arc = m_source − 2.5·log10(μ)`.
- **(B) already-observed source**. Then `m_arc = m_source` and μ is informational.

**Resolved:** the COWLS public release documents `*_source_magnitude_ab` as
**delensed (intrinsic)** source magnitudes, so interpretation **(A)** is correct
— `catalog_checks.py` uses it by default (run `--observed` to see how wrong (B)
would be). See `referenceData.md` for the provenance of every comparison column.

### Math
Flux ratio in band b, from AB magnitudes (smaller mag = brighter; a 1-mag
difference = 10^0.4 ≈ 2.51× in flux):
```
(A) r_b = 10^[ -0.4 ( (m_source,b − 2.5 log10 μ_b) − m_lens,b ) ]
(B) r_b = 10^[ -0.4 ( m_source,b − m_lens,b ) ]
```

### Two ways to get v13's ratio (do the fast one today)
- **(a) Design value — pure catalog, today.** State what the code imposes: arc
  total flux ≈ 0.25 × lens F444W flux, modulated per band by the source color
  ratios in `starforming_color_ratios()`. Compare this *single assumed F444W
  anchor (0.25)* to the COWLS F444W ratio distribution. This already answers "is
  0.25 reasonable?"
- **(b) Measured ratio — optional follow-up, needs the cubes (not new images).**
  Memory-map the existing arrays on CFS and do matched-aperture photometry:
  ```python
  arcs = np.load("/global/cfs/.../v13_consistent/arcs_F150W.npy", mmap_mode="r")
  gal  = np.load("/global/cfs/.../v13_consistent/galaxies_F150W.npy", mmap_mode="r")
  # sum a central aperture on each, ratio = arc_sum / gal_sum, per system
  ```
  This recovers the *realized* per-band ratio and its scatter (the design anchor
  alone can't, because the per-band ratio also depends on each lens's own color).

### Code (COWLS side)
```python
def cowls_ratio(b, intrinsic=True):
    ml = col(f"{b}_lens_magnitude_ab"); ms = col(f"{b}_source_magnitude_ab")
    mu = col(f"{b}_magnification")
    ok = np.isfinite(ml)&np.isfinite(ms)&np.isfinite(mu)&(mu>0)
    m_arc = ms[ok] - (2.5*np.log10(mu[ok]) if intrinsic else 0.0)
    return 10**(-0.4*(m_arc - ml[ok]))
for b in ("F115W","F150W","F277W","F444W"):
    r = cowls_ratio(b); print(b, "median arc/lens =", round(np.median(r),3))
```
Plot: per-band box/violin of the COWLS ratio distribution, with a horizontal line
at v13's 0.25 F444W anchor (and, if you did (b), the v13 measured distribution
overlaid).

### Expected result (precomputed, interpretation A)
| Band | COWLS median arc/lens (IQR) | v13 assumption |
|---|---|---|
| F115W | **0.20** (0.06–0.49) | recolored from F444W anchor |
| F150W | **0.20** (0.06–0.47) | recolored from F444W anchor |
| F277W | **0.075** (0.03–0.18) | recolored from F444W anchor |
| F444W | **0.075** (0.03–0.18) | **0.25 (hard-coded)** |

### How to read it (the headline)
- Real arcs are **strongly band-dependent**: bright (≈0.20) in the blue bands,
  faint (≈0.075) in the red bands. That is physical — lenses are red ellipticals
  (bright in F277W/F444W), arcs are blue star-forming galaxies (bright in
  F115W/F150W).
- v13 **anchors the ratio in F444W at 0.25**, i.e. the *reddest* band where real
  arcs are *faintest*. So at F444W v13 arcs look ~**3× too bright** relative to
  the lens, and the *band trend* may be inverted vs reality.
- **Recommendation for Nate:** move the flux anchor to a blue band (F115W/F150W)
  and set the per-band ratios from this COWLS table, so the arc SED carries the
  observed blue-bright/red-faint trend instead of a flat 0.25 in F444W.

### Caveat
The intrinsic-vs-observed choice (A vs B) must be locked from the paper first;
COWLS source photometry also has large scatter (wide IQRs), so compare
*distributions*, not single medians.

---

## 3. σ_v and Einstein mass vs published lens samples

### What & why (plain English)
The velocity dispersion **σ_v** measures how fast stars move in the lens galaxy —
it is the physical knob that sets lens strength (θ_E ∝ σ_v²) and is the most
widely published lens quantity, so it is the cleanest **survey-independent**
cross-check. The **Einstein mass M_E** is the total mass inside the ring. We
check that v13's σ_v distribution, and its M_E–σ_v scaling, match well-studied
real lens samples (SLACS, BELLS, SL2S), and we connect σ_v back to the θ_E
ceiling from Check 1.

### Reference values (published)
| Sample | lens z | σ_v range / median (km/s) | source |
|---|---|---|---|
| SLACS | 0.06–0.51 | ~160–360 / **~243** | Auger+ 2009 (ApJ 705, 1099) |
| BELLS | 0.4–0.7 | ~180–400 | Brownstein+ 2012 |
| SL2S | 0.2–0.8 | ~150–350 | Sonnenfeld+ 2013 |

(Pull the actual σ_v tables from these papers into a small `published_sigma.csv`
if you want a real KS test rather than a range comparison.)

### What to compute
1. **σ_v marginal distribution.** Histogram/CDF of v13 `sigma_v` (lensed) vs the
   SLACS range; report median and the fraction sitting on the clip boundaries
   (`==80`, `==350`).
2. **The clip vs the true prediction.** Overlay `sigma_v` (clipped) and
   `sigma_v_F150W` (un-clipped FJ prediction). The gap above 350 is the
   population v13 is throwing away — and (via θ_E ∝ σ_v²) it is exactly the
   missing θ_E tail from Check 1.
3. **M_E–σ_v scaling relation (survey-independent).** Plot log10 M_E vs log10 σ_v
   for v13; the SIE forces M_E ∝ σ_v⁴ × (distance factors). Overlay published
   lens points / the SLACS relation and compare **slope + normalization**, not
   raw counts.

### Math
```
θ_E   = 4π (σ_v/c)² (D_ds/D_s)
M_E   = (c²/4G) (D_d D_s / D_ds) θ_E²      ⇒   M_E ∝ σ_v⁴ × (D_d D_s / D_ds)(D_ds/D_s)²
```
So in log-log, M_E vs σ_v should have slope ≈ 4 (with redshift-driven scatter
from the distance factors). Checking that slope is a pure physics test that does
not care which survey the galaxies came from.

### Code
```python
sig = sigma_sim                          # clipped, lensed only
print("sigma_v median %.0f | frac@80 %.3f | frac@350 %.3f"
      % (np.median(sig), (sig<=80.001).mean(), (sig>=349.999).mean()))
m = mass_sim > 0
b, a = np.polyfit(np.log10(sig[m]), np.log10(mass_sim[m]), 1)  # slope b ~ expect ~4
print("M_E-sigma_v log-log slope = %.2f" % b)
```
Plots: (i) σ_v CDF v13 vs SLACS band; (ii) clipped vs un-clipped σ_v histogram
with the 350 line; (iii) M_E–σ_v scatter with the fitted slope + SLACS overlay.

### Expected result (precomputed, numpy-only)
| Quantity | v13 (lensed) | Published |
|---|---|---|
| median σ_v | **265 km/s** | SLACS ~243 |
| frac at floor (80) | 0% | — |
| frac at ceiling (350) | **0.5%** | — |
| un-clipped FJ σ_v max | **461 km/s** | — |
| median log10 M_E | **11.59** (≈3.9e11 M_⊙) | SLACS ~10.8–11.7 |

### How to read it (and the key fix)
- v13's σ_v median (265) is a touch **higher** than SLACS (243) — consistent with
  the θ_E shape-shift in Check 1 (v13 lenses run a bit strong).
- Only **0.5%** of systems are pinned at the 350 ceiling, *yet* the FJ relation
  itself predicts σ_v up to **461 km/s**. The clip is discarding a real,
  physically-motivated high-σ_v tail.
- **Connect the dots:** σ_v = 350 → θ_E ≈ 1.69″ (the ceiling). COWLS's largest is
  θ_E = 2.25″, which needs σ_v ≈ 350·√(2.25/1.69) ≈ **405 km/s** — well inside
  the FJ prediction. **Raising the σ_v clip from 350 to ~410–460 km/s would
  recover essentially all of the missing θ_E tail from Check 1**, using numbers
  the simulator is *already computing* but currently throwing away.

### Caveat
Published σ_v are spectroscopic *stellar* dispersions (aperture-corrected); v13's
σ_v is an SIE *model* dispersion derived from photometry via Faber–Jackson. For
ellipticals these agree to ~5–10%, but they are not identical definitions — note
this when comparing, and don't over-interpret a 5% median offset.

---

## 4. Suggested execution order (one sitting)

1. Write `catalog_checks.py` with the loader skeleton (§0) + the three check
   functions. ~1–2 hrs.
2. Run Check 1 (θ_E) → `theta_E.png` + tail/median/KS numbers. **Fastest, most
   visual win.**
3. Run Check 3 (σ_v / M_E) → it explains *why* Check 1's tail is missing and
   yields the concrete "raise the clip to ~410–460" recommendation.
4. Run Check 2 (flux ratio, interpretation A) → the band-dependence finding;
   defer the cube-photometry version (2b) to a follow-up.
5. Fill `catalog_checks_summary.csv` and drop the three figures into a slide.

**One-paragraph summary you can already write today:** *v13 reproduces the bulk
σ_v / Einstein-mass scale of real lenses (median σ_v 265 vs SLACS 243; median
log M_E 11.6), but (1) its Einstein-radius distribution is both shifted high
(median 1.10″ vs 0.73″) and truncated at 1.69″, missing the ~5% of real lenses
above it; (2) that truncation is an artificial σ_v clip at 350 km/s — the
underlying Faber–Jackson values reach 461, so lifting the clip to ~410–460
recovers the tail; and (3) the arc/lens flux ratio is hard-coded to 0.25 in
F444W, whereas real arcs are blue-bright / red-faint (~0.20 in F115W/F150W,
~0.075 in F277W/F444W), so v13 arcs are ~3× too bright in the reddest band.*

---

## Glossary

- ***AB magnitude*** — a brightness scale where smaller = brighter; a difference
  of 1 mag = a factor 10^0.4 ≈ 2.51 in flux. Flux ratio from mags:
  `10^(-0.4·Δm)`.
- ***CDF (cumulative distribution function)*** — for a value x, the fraction of
  the sample ≤ x. Overlaying two CDFs is the clearest way to see how two
  distributions differ across the whole range at once.
- ***KS test (Kolmogorov–Smirnov)*** — a statistic D = the biggest vertical gap
  between two CDFs, with a p-value for "could these two samples come from the same
  distribution?" Small p (< 0.05) ⇒ they differ significantly. `scipy.stats.ks_2samp`.
- ***Einstein radius θ_E*** — angular radius of the ring/arc; the headline
  measure of lens strength. θ_E ∝ σ_v².
- ***Velocity dispersion σ_v*** — spread of stellar speeds in the lens; sets lens
  strength. Here derived from lens photometry via Faber–Jackson.
- ***Einstein mass M_E*** — total projected mass inside θ_E.
- ***Magnification μ*** — how many times brighter lensing makes the source;
  observed arc flux = intrinsic source flux × μ.
- ***SIE*** — Singular Isothermal Ellipsoid, the mass model v13 uses for the lens.
- ***Faber–Jackson relation*** — empirical link between an elliptical galaxy's
  luminosity and its σ_v (L ∝ σ_v⁴); v13 uses it to set σ_v from brightness.
- ***Selection function*** — the (often unmodelled) rules deciding which systems
  end up in a catalog; why raw counts between two surveys can't be compared
  directly.
