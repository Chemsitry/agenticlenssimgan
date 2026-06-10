# GAN Plan: Diagnosing `simulate_v3`

A staged technical plan for building a GAN whose **purpose is to find what's wrong with the simulator**, not to generate pretty images. The discriminator is the deliverable.

Author note: [gan_architecture_proposals.html](gan_architecture_proposals.html) was **rewritten on 2026-06-09** for the v8/v9 era — it now decides *what* to build and *which real dataset to compare against* (the five data options A–E); this plan covers *how* to execute. The original HTML (an image-refinement survey for v3) is preserved in git history. Our goal (set 2026-05-26) is diagnostic: the lead professor can see a sim-vs-real mismatch we can't; a second colleague's PCA found a signal; we want a tool that tells us **where** the simulator disagrees with real JWST data so we can fix it. Section 0 below critiques the *original* HTML and is kept for the record.

---

## UPDATE 2026-06-09 — colleague's v4–v9 work merged in

We merged the colleague's branches from [agentic-cosmic-webb-sim](https://github.com/kvinneslandn-ML-AI/agentic-cosmic-webb-sim) (`v8` and `newest_version_physical_accuracy`). The merge was clean — no conflicts — because all of his work is in new files. The simulator has moved well past v3. **Read this section before acting on the rest of the plan**, because some stages are now easier, some are moot, and the colleague's own bug fixes give us free validation data.

### What the colleague built since v3

| Version | File(s) | What changed |
|---|---|---|
| v4 | `simulate_v4.py`, `validate_f115w.py` | 630×630 images (18.9" FoV). SED color ratios recalibrated against real COWLS II photometry (440 lenses). Peak-matched amplitude calibration validated against a real lens. |
| v5 | `prep_gan_data.py`, `train_simgan.py`, `refine_dataset.py` | **The colleague already wrote a SimGAN** (refiner + PatchGAN discriminator) for image refinement. Different goal than ours (his refines; ours diagnoses) but the PyTorch code is directly reusable reference, alongside `train_cvae.py`. |
| v6 | `simulate_v6.py`, `prep_stamps_v6.py` | Real galaxy stamps (INTERPOL) replace Sersic profiles for lens + source light — same idea as v3's stamps, rebuilt. |
| v7 | `simulate_v7.py`, `prep_vela_v7.py` | Source light replaced by 20,421 stamps from VELA hydrodynamic simulations. |
| v8 | `simulate_v8.py`, `prep_scenes_v8.py`, `prep_lenses_v8.py` | **Architectural rethink.** No more stitching of separate lens + background cutouts: each sample is ONE real 630×630 JWST cutout ("scene"); the lens galaxy at its center is untouched real pixels. Only the lensed arcs are simulated and added on top. The commit message says this "eliminates all prior artifacts: force fields, chunky halos, noise doubling, visible boxes." |
| v9 | `v9_consistent/simulate_v9_consistent.py` + helpers | **Physical consistency fix.** In v8, lens *appearance* (a real cutout) and lens *mass* (σ_v drawn independently from a TruncNorm) were unrelated — a faint dwarf could get σ_v = 320 km/s. v9 derives σ_v from the cutout's own F115W/F150W/F277W photometry via a Faber–Jackson relation (calibrated on DESI×JWST galaxies, using measured DESI σ_v where available), with a 0.4-dex inter-band consistency rejection and stratified θ_E sampling. |

### Bugs the colleague fixed (and what they mean for us)

These are exactly the kinds of "tells" our diagnostic discriminator is designed to find. He found four of them by eye/inspection; our job is to find the ones that are left.

| Fix (commit) | The bug | Why a discriminator would have caught it |
|---|---|---|
| v6 augmentation (`a05385b`) | Stamp rotation/flip was drawn *per band*, so the same galaxy had a different orientation in each of the 4 bands. Real galaxies have one orientation at all wavelengths. | A 4-channel D learns cross-band morphology cues immediately — this was a trivial giveaway. |
| v8 source SED (`b6eb344`) | `starforming_color_ratios()` was computed per system but **never applied** — arcs had identical flux in every band (flat, colorless SED). After the fix, arcs show the Lyman-break dropout and Balmer break, i.e. realistic orange/red lens-arc colors. | Color (band-ratio) cues are the first thing a multi-band D keys on. |
| v8 star contamination (`59173a8`) | Stars (with JWST 6-point diffraction spikes) passed the brightness/concentration filter and ended up as "lens galaxies" at scene centers. Fixed with a compactness cut (flux(r<3)/flux(r<15) in F444W, threshold 0.35; scene pool 78 → 46). | A star posing as a lens with arcs around it is unphysical; D would flag those scenes. |
| v8 duplicate non-lensed images (`9d71afa`) | Non-lensed samples drew scenes *with replacement* — 8 of 25 non-lensed images were pixel-identical duplicates. Fixed by sampling scene indices without replacement. | Duplicates leak between train/test splits and inflate any classifier metric — this validates our Section 9.3 rule: **split by scene/stamp ID, never randomly**. |

### How this changes the plan

1. **Pick the diagnostic target deliberately.** This plan was written against `simulate_v3`. The colleague's current best is v8/v9. To give him useful feedback, run the pipeline against **v8 or v9 output** (the thing he'd act on). The v3 analysis remains the worked example in the text below; the architecture transfers unchanged except where noted.
2. **Stage 1 (jitter + shortcut elimination) is moot for v8+.** In v8 the background and lens are real pixels — there is no "centered synthetic lens" shortcut and no synthetic background for D to key on (Risk 8.2 disappears). The only simulated pixels are the arcs, so D's attention (Stage 4 heatmaps) should concentrate on arcs and their seams with the real scene. That is a sharper diagnostic, not a weaker one.
3. **File renames (v8+):** the arc-only ground truth is `arcs_{band}.npy`, not `sources_{band}.npy` (`c6aebcf`). Per-band outputs are `images_{band}.npy` (composite), `galaxies_{band}.npy` (real scene only), `arcs_{band}.npy` (simulated arcs only). Our `build_arc_mask()` and dataset code must read `arcs_*` when targeting v8/v9.
4. **Image size is 630×630 in v8/v9** (vs 125/224 in v3). The PatchDiscriminator in Section 3 needs ~2 extra stride-2 blocks (or train on random 224 crops, which also augments). Update the Section 9 open question accordingly.
5. **Free validation of our pipeline (recommended first experiment):** the colleague's fixed bugs give us before/after datasets. Generate a small pre-fix dataset (e.g. `git checkout cc6ce01 -- simulate_v8.py`, run, restore) and a post-fix one; our Stage 2 D should separate pre-fix sims from real *easily* (flat-SED arcs) and post-fix sims less easily. If D can't catch a bug we *know* was there, the pipeline isn't ready to hunt unknown bugs.
6. **New Stage 4 hypothesis from v9:** v8's mass–light inconsistency is visible in images as "arc radius (θ_E) uncorrelated with lens-galaxy brightness." Add `D_score vs (θ_E, lens scene brightness)` to the Stage 5.3 parameter-correlation list; comparing D on v8 vs v9 output tests whether the Faber–Jackson fix closed a real gap.
7. **Feedback deliverable for the colleague** now has a concrete shape: (a) confirmation that the four fixed bugs are no longer detectable by D; (b) ranked list of *remaining* detectable differences (heatmaps + parameter correlations); (c) for v9 specifically, whether the σ_v↔photometry consistency improved D-resistance relative to v8.
8. **Companion documents (2026-06-09):** [simulate_v8_explainer.html](simulate_v8_explainer.html) explains the v8 data inputs/outputs and lists the hard-coded quirks; the rewritten [gan_architecture_proposals.html](gan_architecture_proposals.html) works through the central question of *which real dataset to compare the sims against* (recommended: real COWLS lens cutouts vs lensed sims, with a real-vs-real null control) and the v8-era architecture recommendation. Read both before Stage 0.
9. **Dataset-size warning:** `simulate_v8.py` caps at `--n 92` (46 scenes; non-lensed samples must be unique). The 30k-image targets in the stages below were written for v3 and are unreachable in v8 until the scene pool grows — plan accordingly and prefer patch-level training.

---

## 0. Why this plan looks different from the HTML proposal

The HTML's top recommendation is *Architecture A — Parameter-Calibrated Simulator + Residual SimGAN*. It proposes a small neural network (an MLP) to learn good values for `lens_center_x`, `lens_center_y`, `source_offset`, crop offset, etc. That is overkill for our situation:

| HTML proposal | Our reality |
|---|---|
| Learn lens-center distribution with an MLP | The lens center is hardcoded to `(0, 0)` in [simulate_v3.py:554](simulate_v3.py#L554) and [simulate_v3.py:562](simulate_v3.py#L562). Replace those two lines with `rng.uniform(...)`. This is a 5-line simulator patch, not a neural network. |
| Refiner produces the final deliverable | Refined images are not the deliverable. The discriminator's **explanation** of the gap is the deliverable. |
| Real targets are "lens candidates or matched cutouts" | We're using *random* COSMOS-Web 4-band cutouts from the same DR0.5 mosaics already used by [prep_mosaic.py](prep_mosaic.py). |
| Six architectures, ranked | We commit to one architecture early (small refiner + PatchGAN discriminator + interpretability head) and iterate. |

Implication: we move "fix simulator shortcuts" out of the GAN and into the simulator. The GAN's job is only to find shortcuts we don't already know about.

---

## 1. Background for an ML beginner

### 1.1 What a GAN is, in plain language

A **GAN** (Generative Adversarial Network) is two neural networks playing a game:

- **Generator (G):** takes some input and tries to produce something that looks "real."
- **Discriminator (D):** looks at images and tries to decide if each one is real or came from G.

They train together. D gets better at spotting fakes; G gets better at fooling D. At equilibrium, G's outputs should be indistinguishable from real data.

### 1.2 Why a GAN is the right tool for *this* problem

Most uses of GANs care about G (the generator). For us, **D is the scientist**. If D can reliably tell sim from real, then by definition there is a learnable difference. By looking at *where* and *why* D made its decisions (gradients, patch scores, saliency), we get clues about what the simulator is doing wrong.

So our GAN is not really "G fights D until equilibrium." It is "we train D as hard as we can, and we read D's mind."

### 1.3 PyTorch concepts you'll meet

You said you've never used PyTorch. The plan uses these terms; here's a one-line cheat-sheet:

- `torch.Tensor` — like a numpy array, but can live on a GPU and remembers how it was computed (for gradients).
- `nn.Module` — base class for neural networks. You subclass it, define layers in `__init__`, and write a `forward()` method that says how data flows through them.
- `nn.Conv2d(in_channels, out_channels, kernel, stride, padding)` — a 2D convolution. Think "a learnable image filter."
- `optimizer = torch.optim.Adam(model.parameters(), lr=...)` — Adam is a popular gradient-descent variant. It updates the model's weights using the gradients PyTorch computed.
- Training step pattern:
  ```python
  optimizer.zero_grad()       # clear old gradients
  pred = model(input)          # forward pass
  loss = loss_fn(pred, target) # compute loss
  loss.backward()              # backprop: fill in .grad for every parameter
  optimizer.step()             # update parameters using .grad
  ```
- `DataLoader` — iterable that batches your data. You give it a `Dataset` object and it yields mini-batches.
- `.detach()` — break the computation graph at this tensor. We'll need this so D's training doesn't accidentally update G.

There's existing PyTorch code you can read for reference: [train_cvae.py](train_cvae.py). It's a *conditional VAE* (a different generative model), but the PyTorch idioms (model class, optimizer, training loop, validation loop, saving weights) are the same as what we'll write for the GAN.

---

## 2. The data we have and need

### 2.1 What `simulate_v3` already produces

Per the audit in [simulate_v3.py](simulate_v3.py) and [Summary.txt](Summary.txt):

- `output/v3/images_{F115W,F150W,F277W,F444W}.npy`, shape `(N, 125, 125)` (or `(N, 224, 224)` with `--size 224`). Float32. Units: simulator-internal flux (`sim_units`). Includes lens light + lensed arcs + Poisson noise + real background patch.
- `output/v3/sources_{band}.npy` — same shape, **arc-only** (no lens light, no background). This is our physics ground-truth mask: we know exactly which pixels carry the lensing signal. We will weight loss heavily on these pixels so the GAN cannot "fix" the arcs out of existence. (In v8+ this file is named `arcs_{band}.npy` — see the merge update section above.)
- `output/v3/lensed.npy`, `theta_Es.npy`, `z_lens.npy`, `z_source.npy`, `masses.npy` — scalar metadata per image.
- `output/v3/metadata.json` — parameter distributions used.

What is **not** saved currently (must be added — see Stage 1):
- lens mass center (`center_x`, `center_y`) — currently always 0
- source offset and angle (`src_offset`, `src_angle`)
- mass-model ellipticities (`e1`, `e2`)
- external shear (`gamma1`, `gamma2`)
- arc/lens flux ratio
- UV slope of source SED
- source stamp index, lens stamp index, background patch index

### 2.2 The real data we will assemble

Real targets are random 4-band cutouts from the COSMOS-Web DR0.5 mosaics under `raw_data/1727_mosaic/`. The simulator already reads these mosaics — see [prep_mosaic.py](prep_mosaic.py) — so the data is on disk. We will write a sibling script, `prep_real_targets.py`, that:

1. Opens all four band mosaics.
2. Samples random `(x, y)` patch centers.
3. Extracts an aligned 4-band `(125, 125)` cutout at each center.
4. Rejects cutouts that touch mosaic edges, contain NaNs, or have any pixel with zero weight (per the WHT map).
5. Stores results in `prepped_mosaic/real_targets/cutouts_{band}.npy`, shape `(M, 125, 125)`, plus a `cutout_info.json` with the sky coordinates of each patch.

We aim for `M ≈ 30,000`. That is more than enough to avoid D overfitting.

Crucially, these cutouts are **not** the same as the `backgrounds.npy` already extracted by `prep_mosaic.py`. Backgrounds are intentionally *empty-sky* patches; we want the *full sky distribution* including bright galaxies. Otherwise D learns "real images have galaxies in them; sims have nothing in the background — fake!" and we've learned nothing.

### 2.3 Normalization (the part the old GAN got wrong)

The old GAN at `/global/homes/f/forrestc/small-lens-forecast-sims/gan/data_loader.py` does per-image min-max normalization. **This is the single biggest bug to avoid.** It destroys all physical flux, sky RMS, and color information.

Our normalization will be **fixed** across the whole dataset, per band:

```python
# For each band, compute these ONCE from the real cutouts and save:
sky_med[band]   = sigma_clipped_median(real_cutouts[band].ravel(), sigma=3)
sky_sigma[band] = sigma_clipped_std(real_cutouts[band].ravel(), sigma=3)
k = 3.0  # softening factor

# Then for any image (sim or real) in any band:
def normalize(image, band):
    return np.arcsinh((image - sky_med[band]) / (k * sky_sigma[band]))
```

`arcsinh` (asinh) is approximately linear near zero (so faint pixels are preserved) and approximately logarithmic far from zero (so bright pixels don't dominate). It's the standard astronomical stretch. Importantly, it is **invertible** — we can always recover physical units for display or flux checks.

Save `sky_med`, `sky_sigma`, and `k` in `prepped_mosaic/real_targets/normalization.json`.

---

## 3. The staged plan

Each stage has: **deliverable**, **success criterion**, **estimated effort**. Each stage builds on the previous one. Do not skip ahead.

### Stage 0 — Classical baselines (no neural networks yet)

**Why:** Before training anything, we need to know if a simple linear method can already separate sim from real. If PCA can do it, the GAN must do *at least as well*. Also, this stage forces us to fix any data-loading bugs before they bite us in training.

**Deliverables:**
- `prep_real_targets.py` — extracts the ~30k real cutouts (Section 2.2).
- `audit_distributions.py` — produces 4-panel figures of: per-band pixel histograms (sim vs real, log-scale y-axis), radial profile of brightest galaxy in stamp, distribution of `argmax` pixel position (the centered-lens shortcut test), and 2D power spectrum per band.
- `baseline_pca.py` — reproduces the colleague's PCA discrepancy test:
  1. Normalize all sim images and all real images (per Section 2.3).
  2. Flatten each image to a vector of length `4 × 125 × 125 = 62500`.
  3. Stack into one big matrix of shape `(N_sim + N_real, 62500)`.
  4. Fit `sklearn.decomposition.PCA(n_components=50)` on the combined matrix.
  5. Project all images into the 50-dim PCA space.
  6. Train `sklearn.linear_model.LogisticRegression` on (PCA features → "is sim or real").
  7. Report cross-validated accuracy.

**Success criterion:** Three things must be true to advance to Stage 1:
1. Real cutouts load without NaNs and pass visual smell-test (`matplotlib.imshow` an RGB of 16 random ones).
2. Sim and real pixel histograms overlap in their bulk but disagree somewhere (otherwise there is nothing for a GAN to find).
3. PCA + logistic regression reports CV accuracy that is recorded as the **baseline number to beat**. If it's already 99%, congratulations: the answer is in the first 50 principal components and we don't need a GAN at all — we can already inspect those components to find the bug. If it's near 50% (chance), the difference is non-linear and we need the GAN.

**Effort:** ~2 days.

### Stage 1 — Simulator parameter audit and shortcut elimination

**Why:** Eliminate sim-vs-real differences we already know about, so the GAN can focus on differences we don't. Two specific changes to [simulate_v3.py](simulate_v3.py):

1. **Save more metadata.** Add to the `out` dict in `_simulate_worker` and to the corresponding `np.save(...)` block in `main`: `lens_center_x`, `lens_center_y`, `src_offset`, `src_angle`, `e1`, `e2`, `gamma1`, `gamma2`, `arc_lens_ratio`, `uv_slope`, `src_stamp_idx`, `lens_stamp_idx`, `bg_idx`. Without these we cannot do parameter-correlation analysis in Stage 4.

2. **Randomize lens center** (with a flag, so the colleague can compare). Replace the hardcoded `center_x=0., center_y=0.` in [simulate_v3.py:554](simulate_v3.py#L554), [simulate_v3.py:562](simulate_v3.py#L562), and the corresponding lens-mass `kwargs_lens` line. Sample once per image:
   ```python
   if args.jitter_centers:
       lens_cx = rng.uniform(-pixel_scale * pixels * 0.2, pixel_scale * pixels * 0.2)
       lens_cy = rng.uniform(-pixel_scale * pixels * 0.2, pixel_scale * pixels * 0.2)
   else:
       lens_cx, lens_cy = 0.0, 0.0
   ```
   Use `lens_cx, lens_cy` everywhere `0.` appears for lens position. The jitter range (`0.2 × FoV`) is a starting guess; refine after seeing Stage 0 audit of where real galaxies sit in random cutouts.

**Deliverables:**
- Modified `simulate_v3.py` with `--jitter-centers` flag and full metadata dump.
- Re-run on NERSC: `simulate_v3.py --n 30000 --jitter-centers --size 125` → a new `output/v3_jitter/` dataset comparable in size to the real-cutout set.
- Re-run Stage 0's PCA test on the new sim set. **The baseline accuracy should drop** — we've removed at least one shortcut. The new accuracy is the *new* number to beat.

**Success criterion:** PCA baseline strictly decreases after the simulator change. If it doesn't, we did something wrong (or the prof's mismatch isn't position-related — also useful information).

**Effort:** ~2 days.

### Stage 2 — Discriminator-only first pass

**Why:** Train just the discriminator, no generator. This isn't yet a GAN. It's a binary classifier: "sim or real?". This step is the single most informative thing we can do, and the riskiest part of the plan, because it's where bugs in data loading or normalization will show up as suspiciously perfect accuracy.

**Architecture (PyTorch sketch — explanations below):**

```python
import torch.nn as nn
import torch.nn.utils.spectral_norm as SN

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN-style discriminator.
    Outputs a HxW map of logits, one per ~70x70 receptive field patch.
    """
    def __init__(self, in_channels=4, base=64):
        super().__init__()
        def block(ic, oc, stride):
            return nn.Sequential(
                SN(nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            block(in_channels, base,   stride=2),  # 125 -> 62
            block(base,        base*2, stride=2),  # 62 -> 31
            block(base*2,      base*4, stride=2),  # 31 -> 15
            block(base*4,      base*8, stride=1),  # 15 -> 14
            SN(nn.Conv2d(base*8, 1, kernel_size=4, stride=1, padding=1)),  # -> 13x13 logit map
        )
    def forward(self, x):
        return self.net(x)  # shape (B, 1, 13, 13)
```

Why this shape:
- **4 input channels:** the four bands, stacked along the channel axis. The discriminator can learn that the *ratio* of F115W to F444W (a color) differs between sim and real, not just per-band marginals.
- **PatchGAN (a small output map, not a scalar):** each cell in the 13×13 output map is a logit for one local patch of the input. This forces D to focus on local texture and prevents it from winning with one global cue (like "sim is centered, real isn't"). Background reading: [pix2pix paper](https://arxiv.org/abs/1611.07004), Section 4.2.
- **Spectral normalization (SN):** caps the Lipschitz constant of each conv layer. This is the simplest and most reliable trick to stop D from becoming so confident that gradients vanish. Background: Miyato+ 2018. PyTorch has this built in via `torch.nn.utils.spectral_norm`.
- **No sigmoid at the output.** We use a *hinge loss* (below), which works with raw logits.

**Loss (hinge GAN — easier to train than BCE for beginners):**
```python
def hinge_d_loss(d_real, d_fake):
    # d_real, d_fake are tensors of logits (any shape; we mean-reduce)
    loss_real = torch.relu(1.0 - d_real).mean()
    loss_fake = torch.relu(1.0 + d_fake).mean()
    return 0.5 * (loss_real + loss_fake)
```

Hinge loss only pushes D away from "real ≥ +1" and "fake ≤ −1." Once it's there, gradient is zero — no vanishing-gradient pathology like BCE+sigmoid.

**Training:** Stage 2 has no G. Just a classifier:
```python
for sim_batch, real_batch in zip(sim_loader, real_loader):
    sim_batch  = sim_batch.to(device)   # (B, 4, 125, 125), normalized
    real_batch = real_batch.to(device)
    d_real = D(real_batch)
    d_fake = D(sim_batch)
    loss = hinge_d_loss(d_real, d_fake)
    opt.zero_grad(); loss.backward(); opt.step()
```

**Deliverables:**
- `models/discriminator.py` — the PatchDiscriminator class.
- `train_d_only.py` — the classifier training script.
- `eval_d_only.py` — for the trained D, report: (a) overall classification accuracy on a held-out split (compare against the Stage 1 PCA baseline), (b) per-band ablation: zero out each band one at a time, retrain, see which band carries the most discriminative signal.

**Success criterion:** D's held-out accuracy beats the Stage 1 PCA baseline. If it does not, the difference is either purely linear (PCA suffices — skip to interpretability) or our D architecture is too small (try `base=128`).

**Effort:** ~3 days including NERSC scheduling.

### Stage 3 — Add a refiner; this is the actual GAN

(Continued in Section 4 below — architecture details and pseudocode.)

### Stage 4 — Interpretability layer

(Continued in Section 5 below.)

### Stage 5 — Simulator–GAN co-iteration

(Continued in Section 6 below.)

---

## 4. Stage 3 in detail — the actual GAN

### 4.1 Why we add a refiner at all

After Stage 2, we already have a discriminator that can spot sim-vs-real differences. Why introduce a generator?

Two reasons:
1. **Sharper diagnosis.** A discriminator alone learns *some* difference well enough to win. But it might lock onto one easy difference and ignore everything else. Pairing it with a refiner that tries to erase the gap forces D to keep finding new differences. Each round of training reveals a new aspect of the simulator gap.
2. **Sanity-check the diagnosis.** If the refiner can erase D's chosen feature in just a few thousand iterations, that feature was probably "shallow" (e.g., a sky-level offset). If D keeps finding new features no matter how hard the refiner tries, those features are deeper bugs in the simulator.

### 4.2 Refiner architecture (residual image-to-image)

The refiner takes a 4-channel sim image and outputs a *small additive correction* to that same image. Starting from the identity map (i.e., output = input) is essential — otherwise the refiner trashes the arcs on iteration 1.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.GroupNorm(8, channels),
        )
    def forward(self, x):
        return x + self.block(x)

class Refiner(nn.Module):
    """Input: (B, 5, 125, 125) = 4 sim bands + 1 arc-mask channel.
    Output: (B, 4, 125, 125)   = 4 bands of additive correction in physical units."""
    def __init__(self, n_blocks=4, channels=64):
        super().__init__()
        self.entry = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(5, channels, kernel_size=7),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.exit = nn.Conv2d(channels, 4, kernel_size=1)  # NO activation; raw residual
    def forward(self, x_and_mask):
        h = self.entry(x_and_mask)
        h = self.blocks(h)
        return self.exit(h)
```

Design notes:
- **ReflectionPad2d** instead of zero-padding. Zero-pad creates dark edges that propagate inward and look like artifacts. Reflection-pad gives a smoother boundary condition.
- **GroupNorm**, not BatchNorm. BatchNorm couples examples in a batch, which is bad for GAN stability and a known mode-collapse risk.
- **5 input channels:** the 4 bands plus the binary arc mask `M_arc` (constructed from `sources_*.npy`, see below). Giving the refiner the arc location explicitly is a form of physics conditioning — it knows where it must not damage.
- **4 output channels, no activation:** the output is a raw residual in normalized units. We do *not* squash it through tanh/sigmoid — we let the loss control its magnitude (see "alpha" below).
- **Small — only `4 × 64 = ~250k parameters`.** A larger refiner has the capacity to invent fake structure. We don't want that.

### 4.3 How the refiner is applied

```python
# Input batch of normalized sim images
x_sim = ...                          # (B, 4, 125, 125)
m_arc = build_arc_mask(sources)      # (B, 1, 125, 125), float {0, 1}

delta = R(torch.cat([x_sim, m_arc], dim=1))   # (B, 4, 125, 125)
alpha = 0.10                                   # max residual magnitude (start small)
x_refined = x_sim + alpha * torch.tanh(delta)
```

The `tanh(delta)` bounds each per-pixel residual to `[-1, +1]` in normalized units; `alpha=0.10` then bounds the actual change to `±0.10 × asinh-units`. This is the single most important hyperparameter for arc preservation. Start at `alpha=0.05`, then loosen.

The arc mask comes from existing simulator output:
```python
def build_arc_mask(sources):
    # sources: (B, 4, 125, 125) — arc-only ground truth from sources_*.npy
    arc_signal = sources.abs().sum(dim=1, keepdim=True)         # (B, 1, H, W)
    threshold  = arc_signal.flatten(1).quantile(0.95, dim=1)    # per-image
    return (arc_signal > threshold[:, None, None, None]).float()
```

### 4.4 Losses (refiner side)

The refiner's loss has three terms:

```python
def refiner_loss(D, x_sim, x_refined, m_arc, lambda_pres=20.0, arc_weight=10.0):
    # 1. Adversarial — fool D (hinge form)
    L_adv = -D(x_refined).mean()

    # 2. Pixel preservation — penalize big changes ANYWHERE
    diff = (x_refined - x_sim)
    L_pres = diff.abs().mean()

    # 3. Arc preservation — penalize changes on arc pixels much harder
    arc_diff = diff.abs() * m_arc                  # only arc pixels
    L_arc = arc_diff.sum() / (m_arc.sum() + 1e-6)

    return L_adv + lambda_pres * L_pres + arc_weight * lambda_pres * L_arc
```

Why split into "everywhere" + "arc-weighted"? Because the simulator has two kinds of pixels: arc pixels (physics signal — must be preserved) and non-arc pixels (lens light, background, noise — refiner is allowed to adjust these to match real). Without `L_arc`, a global L1 will allow the refiner to wash out the arcs to lower D's score.

Starting hyperparameters: `alpha=0.05`, `lambda_pres=20`, `arc_weight=10`. Document every change. Don't tune more than one at a time.

### 4.5 The training loop

```python
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_R = torch.optim.Adam(R.parameters(), lr=1e-4, betas=(0.5, 0.999))

for step, (sim_batch, sources_batch, real_batch) in enumerate(loader):
    sim_batch     = sim_batch.to(device)
    sources_batch = sources_batch.to(device)
    real_batch    = real_batch.to(device)
    m_arc = build_arc_mask(sources_batch)

    # ── Update D ────────────────────────────────────────────────────
    with torch.no_grad():
        delta = R(torch.cat([sim_batch, m_arc], dim=1))
        x_refined = sim_batch + alpha * torch.tanh(delta)
    d_real = D(real_batch)
    d_fake = D(x_refined)               # x_refined came through no_grad, so safe
    loss_d = hinge_d_loss(d_real, d_fake)
    opt_D.zero_grad(); loss_d.backward(); opt_D.step()

    # ── Update R every k steps (k=2 typically; D should train faster)
    if step % 2 == 0:
        delta = R(torch.cat([sim_batch, m_arc], dim=1))
        x_refined = sim_batch + alpha * torch.tanh(delta)
        loss_r = refiner_loss(D, sim_batch, x_refined, m_arc,
                              lambda_pres=20.0, arc_weight=10.0)
        opt_R.zero_grad(); loss_r.backward(); opt_R.step()

    # ── Log every 50 steps
    if step % 50 == 0:
        log({'loss_d': loss_d.item(), 'loss_r': loss_r.item(),
             'd_real_mean': d_real.mean().item(),
             'd_fake_mean': d_fake.mean().item()})
```

Beginner-trap notes:
- The `with torch.no_grad():` block when updating D is *critical*. Without it, the gradient from D's loss would flow back into R, partially training R toward making D's job easier. (Bug-find: try removing it once to see how subtle this can be.)
- The `betas=(0.5, 0.999)` Adam setting is folklore from the original DCGAN paper. It is empirically better for GANs than the default `(0.9, 0.999)`.
- We update D twice as often as R. This is just one common choice; if D becomes too strong (its loss approaches 0 and stays there) we'd flip it.

### 4.6 Diagnostics during Stage 3 training

Every N steps, dump a panel:
- `sim_batch[0]` per band
- `x_refined[0]` per band
- `delta[0]` per band (the raw learned correction)
- D's logit-map on a sim image and a refined image, displayed as a heatmap (this is where you start *seeing* what D cares about)

If a single image dump shows the refiner producing huge corrections inside the arc mask, halt and increase `arc_weight`. If D's logit map is bright everywhere uniformly, D found a global cue — probably normalization is broken.

**Stage 3 success criterion:** D's accuracy on a held-out sim/real test drops compared to Stage 2 (because the refiner is genuinely closing the gap), but D never drops to 50% (chance). The interesting middle is where D's logit map *focuses* on specific regions — those regions are the simulator-vs-real differences.

**Effort:** ~1–2 weeks including hyperparameter exploration.

---

## 5. Stage 4 in detail — making D talk

This is where the diagnostic payoff lives. After Stages 2 and 3 we have a trained D. Now we extract from it *what* it learned. Four tools, in order of how much code they take:

### 5.1 Patch-score heatmaps (free — just plot D's output)

D's output is a `(B, 1, 13, 13)` map of logits. Each cell corresponds to a roughly 70×70-pixel patch of the input. Upsample the map back to 125×125 and overlay it on the input image. Bright cells = "D thinks this region looks fake."

```python
def patch_score_overlay(D, x, save_to):
    with torch.no_grad():
        logit_map = D(x).squeeze(1)            # (B, 13, 13)
    upsampled = F.interpolate(logit_map.unsqueeze(1),
                              size=(125, 125), mode='bilinear')
    # Plot: x[0,band], upsampled[0] heatmap, alpha-blended
    ...
```

Do this for: (a) random sim images, (b) random real images, (c) refined sim images. The places D consistently lights up on sim but not on real are the **structural** disagreements. The places it lights up on both at random are noise/uncertainty.

### 5.2 Per-band ablation

For a held-out test set, zero out one band at a time and re-run D. Record accuracy drop. The band whose absence hurts D most is the band carrying the most discriminative signal — that's a clue about the simulator's color/SED modeling.

Already a deliverable in Stage 2's `eval_d_only.py`; re-run after Stage 3 with the full pipeline.

### 5.3 Parameter-conditional analysis (the Stage 1 metadata pays off here)

For every sim image we now have a full parameter vector (lens center, theta_E, z_lens, z_source, ellipticity, shear, arc/lens flux ratio, UV slope, source/lens stamp index, background index). For each test sim image, record D's per-image score (mean of the logit map).

Then for each parameter, plot `D_score vs parameter`. Strong correlations are the diagnostic gold:
- Strong negative slope of D_score vs theta_E → "D thinks bigger-Einstein-radius systems look more fake" → maybe the simulator gets large arcs wrong.
- Strong correlation with `src_stamp_idx` or `lens_stamp_idx` → "particular source/lens stamps look unrealistic when lensed" → maybe the INTERPOL convolution treatment is off (see [Summary.txt](Summary.txt) Step 13 — known double-PSF-convolution issue).
- Strong correlation with `z_source` → "high-z sources look wrong" → maybe the Lyman-break SED model is off.
- Correlation with `bg_idx` → "specific background patches look wrong when combined with our noise model" → maybe Poisson is the wrong noise model for the background.

This is the analysis that turns "the GAN works" into "the simulator has problem X."

Deliverable: `interpret_parameter_correlations.py`.

### 5.4 Saliency / integrated gradients (slightly more code)

For a single sim image, compute the gradient of D's output with respect to each input pixel. The magnitude tells us *which pixels* drive D's decision. Integrated gradients (Sundararajan+ 2017) averages gradients along a path from a baseline (e.g., zero image) to the input, which is more reliable than raw gradients. PyTorch package `captum` provides this in two lines:

```python
from captum.attr import IntegratedGradients
ig = IntegratedGradients(D)
attribution = ig.attribute(x_sim, target=0)  # (B, 4, 125, 125)
```

Visualize per-band. Combined with the patch-score heatmaps and the parameter analysis, this is what you show the lead prof at the end: "Here are the pixels and regions where D is making its decision, and here are the simulator parameters those decisions correlate with."

### 5.5 Comparison against the PCA baseline (loop back to Stage 0)

Project the trained D's penultimate-layer activations into a 2D plot (UMAP or PCA). Color sim vs real. Compare against the Stage 0 PCA projection of raw pixels. If the GAN learned something new, the two projections will *differ* — D's space will separate sim and real along axes that pixel PCA does not.

**Stage 4 success criterion:** At least one parameter correlation in 5.3 has |r| > 0.3 with D's score, OR the patch-score heatmap consistently lights up an interpretable region (e.g., arc edges). Either gives us something to write up.

**Effort:** ~1 week of plotting + writeup.

---

## 6. Stage 5 in detail — simulator–GAN co-iteration

Once Stage 4 has produced one or more concrete hypotheses about the simulator gap (e.g., "the arc edges look too sharp" or "the F444W background noise is the wrong color"), the loop is:

1. **Propose a simulator fix** based on the Stage 4 diagnostic. Examples:
   - If D loves arc edges: PSF deconvolve the source stamps before INTERPOL (Upgrade 3 in [Summary.txt](Summary.txt)).
   - If D loves the F444W noise: revisit the Poisson noise model in [simulate_v3.py:301-317](simulate_v3.py#L301-L317) — maybe read-noise needs to be added explicitly.
   - If D loves the center pixel: the centered-lens shortcut isn't fully fixed — widen the jitter range.
2. **Patch the simulator.** Generate a fresh dataset.
3. **Re-run Stage 3 training from scratch.** Do not warm-start D from the previous checkpoint — we want a clean experiment.
4. **Compare:** new D-accuracy on held-out test set, vs. old. If accuracy dropped, the fix worked. If unchanged, the fix was orthogonal to what D is using.
5. **Look at new D's heatmap and parameter correlations.** What is D using *now*?
6. **Goto 1.**

Each iteration takes ~1 week. Realistically, three iterations gets you to "we've explained most of what was visually obvious to the prof," and that's the natural stopping point — write it up.

**Stage 5 success criterion:** Two or more simulator fixes have been proposed, implemented, and shown via D-accuracy regression to have closed part of the sim-real gap. This is the deliverable for the colleague.

**Effort:** open-ended; budget 3–6 weeks.

---

## 7. Repository layout

All new code lives in `agenticlenssim/agenticlenssimgan/`. Recommended structure:

```
agenticlenssimgan/
├── simulate_v3.py            # existing, modified in Stage 1
├── prep_mosaic.py            # existing
├── prep_sources.py           # existing
├── prep_lenses.py            # existing
│
├── gan/                      # NEW: all GAN code lives here
│   ├── __init__.py
│   ├── data/
│   │   ├── prep_real_targets.py     # Stage 0: extracts random COSMOS-Web cutouts
│   │   ├── audit_distributions.py   # Stage 0: sim vs real histograms / radial profiles
│   │   ├── normalize.py             # Stage 0: fixed asinh normalization
│   │   └── datasets.py              # PyTorch Dataset for sim+sources, real
│   ├── baselines/
│   │   └── pca.py                   # Stage 0: PCA + logistic regression baseline
│   ├── models/
│   │   ├── discriminator.py         # Stage 2: PatchDiscriminator
│   │   └── refiner.py               # Stage 3: Refiner with arc-mask input
│   ├── losses.py                    # hinge GAN loss, refiner_loss
│   ├── train_d_only.py              # Stage 2: D-only classifier
│   ├── train_gan.py                 # Stage 3: full GAN training
│   ├── eval/
│   │   ├── eval_d_only.py           # Stage 2: held-out classification accuracy
│   │   ├── patch_heatmaps.py        # Stage 4: per-patch D scores
│   │   ├── parameter_correlations.py# Stage 4: D-score vs sim parameter
│   │   └── saliency.py              # Stage 4: integrated gradients
│   └── README.md                    # how to run each stage
│
├── gan_plan.md               # THIS FILE
└── gan_architecture_proposals.html  # the older proposal we partially overrode
```

Outputs go to `output/gan/` (a sibling of `output/v3/`):
```
output/gan/
├── real_cutouts/             # Stage 0 output of prep_real_targets.py
│   ├── cutouts_{band}.npy
│   ├── cutout_info.json
│   └── normalization.json
├── baselines/
│   └── pca_results.json      # baseline accuracy to beat
├── stage2_d_only/
│   ├── d_weights.pt
│   └── eval_report.json
└── stage3_gan_run_NN/        # one directory per training run
    ├── d_weights.pt
    ├── r_weights.pt
    ├── train_log.npy
    ├── config.json           # all hyperparams used
    └── diagnostics/          # heatmaps, saliency, correlation plots
```

Save config files (with full hyperparameters and code git-hash) at the start of every training run. Otherwise, after 20 runs, you will not remember which one used `alpha=0.05` vs `alpha=0.10`.

---

## 8. Risks and decision points

Listed roughly in order of likely impact.

### 8.1 The PCA baseline crushes the GAN

**Symptom:** PCA + logistic regression hits 99% accuracy in Stage 0.

**Meaning:** The sim-vs-real signal is linear and concentrated in a few principal components. A GAN is overkill and a distraction.

**Action:** Skip directly to inspecting the top PCA components (reshape each back to `(4, 125, 125)` and visualize). The components that contribute most to the logistic regression's decision function are the simulator-vs-real "directions" — literally a picture of what's wrong. Write this up; the GAN may not be needed at all.

### 8.2 Real cutouts have galaxies; sims have a centered lens — D wins trivially

**Symptom:** Stage 2 D hits 100% accuracy in the first epoch.

**Meaning:** D learned the centered-galaxy shortcut from sim, not anything subtle.

**Action:** This is the central scenario the HTML proposal worried about. The mitigation is Stage 1's `--jitter-centers` flag. If 100% accuracy persists *after* jittering, increase jitter range, or pre-filter real cutouts to ones with a bright central object (concentration filter, similar to [prep_lenses.py](prep_lenses.py) detection logic).

### 8.3 Mode collapse / refiner trashes arcs

**Symptom:** `loss_arc` is climbing; `delta` is nearly identical across input images.

**Meaning:** Refiner found one global correction that lowers D's score and applied it everywhere.

**Action:** Increase `arc_weight` (try ×4). Lower `alpha` (try ÷2). Increase `lambda_pres`. If still bad, reduce refiner capacity (fewer blocks).

### 8.4 D wins forever; refiner gradient saturates

**Symptom:** D loss approaches 0; refiner loss stops decreasing.

**Meaning:** D is too strong. Hinge loss helps here vs BCE — but spectral norm and ratio of D:R updates are the levers.

**Action:** Flip the update ratio (R every step, D every 2 steps). Add gradient penalty (WGAN-GP style). Reduce D learning rate.

### 8.5 We can't reproduce the prof's "obvious" observation

**Symptom:** After Stage 5, two simulator fixes and three iterations, D's accuracy on held-out is essentially the same as Stage 2.

**Meaning:** The prof's observation is something neither PCA nor a CNN-PatchGAN can capture. Examples: redshift distribution of lensed systems (not visible per-image), spatial clustering of lenses across the sky (not visible per-image), or a science-level inconsistency (e.g., implied source-counts contradict known luminosity function).

**Action:** Stop and ask the prof. The plan has run its course; the answer was outside the per-image-pixel-statistics frame the GAN can address. This is itself a finding worth writing up.

### 8.6 Real cutouts are too similar to sims to give D anything to learn

**Symptom:** Stage 2 D hits ~50% (chance) and stays there. Stage 0 PCA was also ~50%.

**Meaning:** The simulator is already very good at the per-cutout marginals, and the prof's observation operates at a higher level.

**Action:** Same as 8.5 — escalate to the prof.

---

## 9. Open questions to resolve before starting

These are decisions that don't change the structure of the plan, but you'll need to answer Stage 0:

1. **Image size.** [simulate_v3.py](simulate_v3.py) supports both 125 and 224 with `--size`. Larger is more diagnostic (more context per cutout). Pick one and stick with it. **Recommendation: 224**, because [Summary.txt](Summary.txt) Upgrade 2 already flagged 125 as too tight for the larger lens systems. **Post-merge note:** v8/v9 output is 630×630; if targeting those, either add ~2 stride-2 blocks to D or train on random 224-pixel crops of the 630 images (the crop doubles as augmentation).
2. **How many sim images.** Stage 1 says ~30k to match real. With multiprocessing on Perlmutter that's ~1 hour. Cheap.
3. **Train/val/test split.** Split *by background patch ID* and *by source/lens stamp ID*, not by random image. Otherwise the same lens light or background appears in both train and test, leaking.
4. **Whether to include non-lensed sims (`lensed=0`) in training.** Probably yes — D should learn that "non-lensed sim" looks more like "random real" than "lensed sim" does. Useful sanity check.
5. **Wandb or local logging.** Local-only for v0. NERSC compute nodes often don't have outbound network. Save JSON + numpy logs to `output/gan/stage3_gan_run_NN/train_log.npy`.

---

## 10. Glossary

(Alphabetical, brief.)

- **Adversarial loss** — The part of the loss that pits G against D. Each tries to make the other's loss worse.
- **Arc** — The lensed image of a background source galaxy. The thing the simulator is ultimately trying to produce.
- **arcsinh / asinh stretch** — `arcsinh(x)`. Behaves like `x` near zero and like `log(2x)` for large `x`. Standard astronomical display stretch.
- **Batch norm vs group norm** — Normalization layers inside a network. BatchNorm computes statistics across the batch axis; GroupNorm across feature groups. GAN folklore prefers GroupNorm for stability.
- **Discriminator (D)** — The "is it real or fake?" network. In our case, the deliverable.
- **DCGAN** — Original convolutional GAN paper (Radford+ 2015). Source of many heuristics still in use (Adam betas, LeakyReLU, no fully-connected layers).
- **Generator (G) / Refiner (R)** — The "make it look real" network. We use "refiner" because our G doesn't start from noise — it starts from a simulator image and produces a small correction.
- **Hinge loss** — GAN loss form used in modern image GANs (Lim+Ye 2017, Tran+ 2017). More stable than BCE+sigmoid for image-resolution discriminators.
- **INTERPOL** — Lenstronomy light-model name for "use this pixelized image as the light profile." Used in [simulate_v3.py](simulate_v3.py) for both source and lens with real DR0.5 stamps.
- **Lipschitz constant** — How fast a function can change. Bounded Lipschitz constant on D = bounded gradients = stable GAN training. Achieved via spectral normalization or gradient penalty.
- **MLP** — Multi-Layer Perceptron. A neural network of fully-connected layers, no convolutions. Used in the HTML proposal for the parameter generator we decided not to build.
- **Mode collapse** — Failure mode where G produces a small set of similar outputs that fool D, ignoring most of the input variety. For us: refiner producing identical corrections regardless of input image.
- **PatchGAN** — A fully-convolutional discriminator that outputs a grid of scores (one per receptive-field patch) instead of a single scalar. Source: pix2pix (Isola+ 2017).
- **PCA** — Principal Component Analysis. Linear dimensionality reduction. Our Stage 0 baseline.
- **PSF** — Point Spread Function. The image of a point source through the telescope. Convolution with the PSF blurs the simulated lens.
- **Residual block** — `output = input + f(input)`. Lets the network learn corrections to its input rather than rewriting from scratch. Critical for refiners.
- **Sigma-clipped statistics** — Compute mean/std, drop points more than N sigma away, recompute. Iterate. Robust to outliers (e.g., bright stars in a "sky" measurement).
- **SimGAN** — Apple 2017 paper (Shrivastava+) introducing simulator + refiner + local discriminator + history buffer. The architectural ancestor of what we're building.
- **Spectral normalization** — Divide each conv layer's weights by their largest singular value during the forward pass. Bounds the Lipschitz constant. (Miyato+ 2018.)
- **Tanh-residual output** — `output = input + alpha * tanh(delta)`. Bounds the correction's magnitude. Our refiner's output form.
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty (Gulrajani+ 2017). The "industrial-strength" stable GAN loss. We *don't* start here; we try hinge first.

---

## 11. What success looks like

A short pitch you could give the colleague at the end:

> "We built a 4-band PatchGAN discriminator that can tell `simulate_v3` outputs from random COSMOS-Web cutouts with X% accuracy on a held-out test set, beating a 50-component PCA baseline by Y points. The discriminator's per-patch attention maps consistently highlight [arc edges / F444W background / lens-galaxy edges / …]; its per-image score correlates most strongly with [theta_E / z_source / lens stamp ID / …]. We proposed and tested three simulator changes targeting those features. After all three, GAN accuracy fell from X% to Z%, indicating the simulator now matches real JWST data in the dimensions the GAN could measure. Remaining gap: [a clear statement of what the GAN couldn't capture, sent back to the prof]."

That paragraph is the goal of the whole plan. Every stage exists to make some clause of that paragraph factually true.

---

## 12. Sequencing summary

| Stage | What | Effort | Blocking for |
|---|---|---|---|
| 0 | Data prep + PCA baseline | 2 days | everything |
| 1 | Simulator metadata + jitter | 2 days | Stage 2 onwards |
| 2 | D-only classifier | 3 days | Stage 3 |
| 3 | Full GAN | 1–2 wk | Stage 4 |
| 4 | Interpretability | 1 wk | Stage 5 |
| 5 | Sim–GAN co-iteration | 3–6 wk | (final deliverable) |

Total realistic timeline: **6–10 weeks** of focused work, with the colleague reviewing simulator changes between Stages 1 and 5.

End of plan.

