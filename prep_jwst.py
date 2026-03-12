"""
prep_jwst.py

Prepares real JWST NIRCam F115W data for use as realistic backgrounds
in the gravitational lensing simulation notebook.

Outputs (saved to ./prepped/):
  - real_backgrounds.npy     : (N, 125, 125) array of sky patch cutouts
  - background_rms.npy       : scalar, sigma-clipped RMS of the sky background
  - background_mean.npy      : scalar, sigma-clipped mean background level
  - psf_stars.npy            : (n_stars, 63, 63) array of empirical PSF stamps
                               (used to build a PSF model if desired)
"""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# ── Config ──────────────────────────────────────────────────────────────────
FITS_PATH = "MAST_2026-02-26T2313/JWST/jw01810-o002_t002_nircam_clear-f115w_i2d.fits"
OUT_DIR   = "prepped"
PATCH_SIZE     = 125       # pixels — must match simulation notebook
N_PATCHES      = 5000      # number of background cutouts to extract
PSF_STAMP_SIZE = 63        # pixels for PSF star stamps (odd number)
MASK_SIGMA     = 3.0       # sigma threshold for source masking
VALID_FRAC     = 0.90      # fraction of patch that must be unmasked to keep it
RANDOM_SEED    = 42
# ────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load the science image ────────────────────────────────────────────────
print("Loading F115W mosaic (SCI extension)...")
with fits.open(FITS_PATH, memmap=True) as hdul:
    sci  = hdul["SCI"].data.astype(np.float32)
    wht  = hdul["WHT"].data.astype(np.float32)
    header = hdul["SCI"].header

ny, nx = sci.shape
print(f"  Image shape: {ny} x {nx}")

# ── 2. Build a validity mask (finite, non-zero weight) ──────────────────────
print("Building validity mask...")
valid = np.isfinite(sci) & (wht > 0)

# ── 3. Measure sky background via sigma-clipping ────────────────────────────
print("Measuring background (sigma-clipped stats on 1-in-100 pixels)...")
# Subsample to keep memory manageable
subsample = sci[valid][::100]
bg_mean, bg_median, bg_rms = sigma_clipped_stats(subsample, sigma=3.0, maxiters=5)
print(f"  Background mean : {bg_mean:.6f}")
print(f"  Background median: {bg_median:.6f}")
print(f"  Background RMS  : {bg_rms:.6f}")

np.save(os.path.join(OUT_DIR, "background_rms.npy"),  np.float32(bg_rms))
np.save(os.path.join(OUT_DIR, "background_mean.npy"), np.float32(bg_mean))

# ── 4. Build a source mask ───────────────────────────────────────────────────
print("Building source mask (sigma-clip threshold)...")
# Flag pixels more than MASK_SIGMA above the background as sources
source_mask = (sci - bg_median) > MASK_SIGMA * bg_rms
# Combined bad-pixel mask: invalid OR is a bright source
bad = ~valid | source_mask

# ── 5. Extract 125×125 background cutouts ───────────────────────────────────
print(f"Extracting up to {N_PATCHES} background patches ({PATCH_SIZE}x{PATCH_SIZE} px)...")
half   = PATCH_SIZE // 2
margin = half + 1

patches = []
attempts = 0
max_attempts = N_PATCHES * 20

while len(patches) < N_PATCHES and attempts < max_attempts:
    attempts += 1
    # Random center strictly inside the image
    cy = rng.integers(margin, ny - margin)
    cx = rng.integers(margin, nx - margin)

    patch_bad = bad[cy - half : cy + half + 1, cx - half : cx + half + 1]

    # Skip if patch shape is wrong (edge case)
    if patch_bad.shape != (PATCH_SIZE, PATCH_SIZE):
        continue

    # Keep only patches that are mostly unmasked (real sky, not source-dominated)
    if np.mean(~patch_bad) < VALID_FRAC:
        continue

    patch = sci[cy - half : cy + half + 1, cx - half : cx + half + 1].copy()
    # Subtract local background so patches have zero mean
    patch -= bg_median
    patches.append(patch)

    if len(patches) % 500 == 0:
        print(f"  {len(patches)} / {N_PATCHES} patches collected ({attempts} attempts)")

patches = np.array(patches, dtype=np.float32)
print(f"  Done: {len(patches)} patches collected in {attempts} attempts.")
np.save(os.path.join(OUT_DIR, "real_backgrounds.npy"), patches)
print(f"  Saved → {OUT_DIR}/real_backgrounds.npy  shape={patches.shape}")

# ── 6. Extract PSF star stamps ───────────────────────────────────────────────
print("Extracting PSF star stamps...")
shalf = PSF_STAMP_SIZE // 2
margin_psf = shalf + 5

# Find candidate point sources:
#   - bright (> 20-sigma above background)
#   - isolated (no other bright source within PSF_STAMP_SIZE)
bright_thresh  = bg_median + 20 * bg_rms
compact_thresh = bg_median + 5  * bg_rms  # used later to check compactness

# Label connected bright regions quickly using simple threshold
from scipy.ndimage import label as nd_label, find_objects as nd_find_objects

bright_mask = (sci > bright_thresh) & valid
labeled, n_features = nd_label(bright_mask)
print(f"  Found {n_features} bright regions.")

psf_stamps = []
if n_features > 0:  # noqa: keep block even when psf_stamps pre-initialized
    slices = nd_find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        sy = sl[0]; sx = sl[1]
        # Keep only compact (point-source-like) objects: bounding box ≤ 10 px
        if (sy.stop - sy.start) > 10 or (sx.stop - sx.start) > 10:
            continue
        # Centroid
        cy = (sy.start + sy.stop) // 2
        cx = (sx.start + sx.stop) // 2
        if cy < margin_psf or cy > ny - margin_psf or cx < margin_psf or cx > nx - margin_psf:
            continue
        stamp = sci[cy - shalf : cy + shalf + 1, cx - shalf : cx + shalf + 1].copy()
        if stamp.shape != (PSF_STAMP_SIZE, PSF_STAMP_SIZE):
            continue
        # Normalise
        stamp -= bg_median
        total = np.sum(stamp)
        if total <= 0:
            continue
        stamp /= total
        psf_stamps.append(stamp.astype(np.float32))
        if len(psf_stamps) >= 200:
            break

if psf_stamps:
    psf_stamps = np.array(psf_stamps, dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "psf_stars.npy"), psf_stamps)
    print(f"  Saved → {OUT_DIR}/psf_stars.npy  shape={psf_stamps.shape}")
else:
    print("  No suitable PSF stars found — skipping psf_stars.npy")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("\n=== Prep complete ===")
print(f"  Output directory : {OUT_DIR}/")
print(f"  real_backgrounds : {patches.shape}  dtype={patches.dtype}")
print(f"  background_rms   : {bg_rms:.6f}  (in native FITS flux units)")
print(f"  background_mean  : {bg_mean:.6f}")
if len(psf_stamps):
    print(f"  psf_stars        : {psf_stamps.shape}")
print()
print("NOTE: The FITS image is in MJy/sr. The simulation notebook uses a")
print("  `sum_to_flux` factor (6.5 nJy/pixel-sum) calibrated to a different")
print("  dataset. You will need to re-derive this factor or re-scale the")
print("  background patches when integrating them into the sim notebook.")
