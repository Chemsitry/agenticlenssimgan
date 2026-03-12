"""
prep_multifield.py

Multi-field, multi-band background prep pipeline for JWST NIRCam data.
Extends prep_jwst.py to handle multiple FITS files and multiple bands.

Extracts aligned sky patches from SW and LW channel images using WCS
coordinate mapping, stores backgrounds in raw MJy/sr units with
per-band PIXAR_SR metadata in a companion JSON.

Output structure:
  prepped_v2/
    <field>/F115W/backgrounds.npy   (N, 125, 125) float32
    <field>/F277W/backgrounds.npy   (N, 63, 63)   float32
    <field>/multiband_backgrounds.npz
    combined/F115W_backgrounds.npy  (N_total, 125, 125)
    pixar_sr.json                   per-band PIXAR_SR values

Usage:
    python prep_multifield.py --field jw01810 --bands F115W F277W
    python prep_multifield.py --field ceers --fits raw_data/1345/F115W/*.fits
    python prep_multifield.py --combine
"""

import argparse
import json
import os
import glob as glob_module
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

# ── Configuration ────────────────────────────────────────────────────────────

# SW bands: 0.031 "/pix — 125x125 px ≈ 3.875" FoV
SW_BANDS  = {"F090W", "F115W", "F150W", "F200W"}
SW_HALF   = 62   # half-size for 125x125 patch

# LW bands: 0.063 "/pix — 63x63 px ≈ 3.969" FoV (same angular scale as SW)
LW_BANDS  = {"F277W", "F356W", "F444W"}
LW_HALF   = 31   # half-size for 63x63 patch

N_PATCHES      = 5000
MASK_SIGMA     = 3.0
VALID_FRAC     = 0.90
RANDOM_SEED    = 42
OUT_DIR        = "prepped_v2"

# Map each JWST program ID to its FITS path glob pattern
FIELD_FITS_PATTERNS = {
    "jw01810": {
        "F115W": "MAST_2026-02-26T2313/JWST/jw01810-o002_t002_nircam_clear-f115w_i2d.fits",
        # Add other bands here as they become available:
        # "F277W": "MAST_2026-02-26T2313/JWST/jw01810-*-f277w_i2d.fits",
    },
    "ceers": {
        "F115W": "raw_data/1345/F115W/**/*i2d*.fits*",
        "F150W": "raw_data/1345/F150W/**/*i2d*.fits*",
        "F200W": "raw_data/1345/F200W/**/*i2d*.fits*",
        "F277W": "raw_data/1345/F277W/**/*i2d*.fits*",
        "F356W": "raw_data/1345/F356W/**/*i2d*.fits*",
        "F444W": "raw_data/1345/F444W/**/*i2d*.fits*",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_fits_band(fits_path: str):
    """Load SCI data, WHT, WCS, and PIXAR_SR from a JWST i2d FITS file."""
    with fits.open(fits_path, memmap=True) as hdul:
        sci    = hdul["SCI"].data.astype(np.float32)
        wht    = hdul["WHT"].data.astype(np.float32)
        header = hdul["SCI"].header
        wcs    = WCS(header, naxis=2)
        # PIXAR_SR is in the primary header for JWST pipeline products
        pixar_sr = hdul[0].header.get("PIXAR_SR", header.get("PIXAR_SR", None))
    return sci, wht, wcs, header, pixar_sr


def measure_background(sci, wht):
    """Sigma-clipped stats on a 1-in-100 subsample of valid pixels."""
    valid = np.isfinite(sci) & (wht > 0)
    subsample = sci[valid][::100]
    bg_mean, bg_median, bg_rms = sigma_clipped_stats(subsample, sigma=3.0, maxiters=5)
    return bg_mean, bg_median, bg_rms, valid


def extract_patches_single_band(sci, wht, half, n_patches, rng, bg_subtract=True):
    """Extract sky patches from a single-band image. Returns patches in raw MJy/sr."""
    bg_mean, bg_median, bg_rms, valid = measure_background(sci, wht)
    source_mask = (sci - bg_median) > MASK_SIGMA * bg_rms
    bad = ~valid | source_mask

    patch_size = 2 * half + 1
    ny, nx = sci.shape
    margin = half + 1

    patches = []
    centers = []   # pixel (cy, cx) for WCS mapping to other bands
    attempts = 0
    max_attempts = n_patches * 20

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        cy = int(rng.integers(margin, ny - margin))
        cx = int(rng.integers(margin, nx - margin))

        patch_bad = bad[cy - half : cy + half + 1, cx - half : cx + half + 1]
        if patch_bad.shape != (patch_size, patch_size):
            continue
        if np.mean(~patch_bad) < VALID_FRAC:
            continue

        patch = sci[cy - half : cy + half + 1, cx - half : cx + half + 1].copy()
        if bg_subtract:
            patch -= bg_median
        patches.append(patch)
        centers.append((cy, cx))

        if len(patches) % 500 == 0:
            print(f"    {len(patches)} / {n_patches} patches ({attempts} attempts)")

    patches = np.array(patches, dtype=np.float32)
    print(f"    Extracted {len(patches)} patches in {attempts} attempts.")
    return patches, centers, bg_median, bg_rms


def extract_aligned_patch(sci, wht, wcs_ref, wcs_band, center_ref, half_ref, half_band, bg_median):
    """
    Given a reference pixel center (cy, cx) in wcs_ref coordinates,
    map it to wcs_band and extract a patch.
    Returns patch or None if out of bounds / too masked.
    """
    from astropy.coordinates import SkyCoord
    cy_ref, cx_ref = center_ref
    sky = wcs_ref.pixel_to_world(cx_ref, cy_ref)   # returns SkyCoord (x,y -> ra,dec)
    px, py = wcs_band.world_to_pixel(sky)
    cx = int(round(float(px)))
    cy = int(round(float(py)))

    ny, nx = sci.shape
    margin = half_band + 1
    if cy < margin or cy >= ny - margin or cx < margin or cx >= nx - margin:
        return None

    valid = np.isfinite(sci) & (wht > 0)
    source_mask = (sci - bg_median) > MASK_SIGMA * 1e-4  # rough threshold
    bad = ~valid | source_mask

    patch = sci[cy - half_band : cy + half_band + 1, cx - half_band : cx + half_band + 1].copy()
    patch_bad = bad[cy - half_band : cy + half_band + 1, cx - half_band : cx + half_band + 1]

    patch_size = 2 * half_band + 1
    if patch.shape != (patch_size, patch_size):
        return None
    if np.mean(~patch_bad) < VALID_FRAC:
        return None

    patch -= bg_median
    return patch.astype(np.float32)


# ── Main per-field processing ─────────────────────────────────────────────────

def process_field(field_name: str, bands: list, fits_overrides: dict = None):
    """
    Process one JWST field for the requested bands.

    Parameters
    ----------
    field_name : str
        Key in FIELD_FITS_PATTERNS (e.g. 'jw01810', 'ceers')
    bands : list of str
        Bands to extract (e.g. ['F115W', 'F277W'])
    fits_overrides : dict, optional
        {band: fits_path} to override default pattern lookup
    """
    print(f"\n{'='*60}")
    print(f"Processing field: {field_name}  bands: {bands}")
    print(f"{'='*60}")

    field_out = Path(OUT_DIR) / field_name
    field_out.mkdir(parents=True, exist_ok=True)

    patterns = FIELD_FITS_PATTERNS.get(field_name, {})
    if fits_overrides:
        patterns = {**patterns, **fits_overrides}

    # ── Identify reference SW band (F115W preferred) ─────────────────────────
    sw_ref_band = next((b for b in ["F115W", "F090W", "F150W", "F200W"] if b in bands), None)
    if sw_ref_band is None:
        raise ValueError(f"No SW reference band found in {bands}. Include at least F115W.")

    ref_pattern = patterns.get(sw_ref_band)
    if ref_pattern is None:
        raise FileNotFoundError(f"No FITS path/pattern for {field_name}/{sw_ref_band}")

    ref_fits_files = sorted(glob_module.glob(ref_pattern, recursive=True))
    if not ref_fits_files:
        raise FileNotFoundError(f"No files matched pattern: {ref_pattern}")

    print(f"\n  Reference band: {sw_ref_band}  ({len(ref_fits_files)} file(s))")

    all_band_patches = {b: [] for b in bands}
    pixar_sr_map = {}
    rng = np.random.default_rng(RANDOM_SEED)

    for fits_path in ref_fits_files:
        print(f"\n  -- File: {os.path.basename(fits_path)}")
        sci_ref, wht_ref, wcs_ref, _, pixar_ref = load_fits_band(fits_path)

        if pixar_ref is not None:
            pixar_sr_map[sw_ref_band] = float(pixar_ref)

        half_ref = SW_HALF if sw_ref_band in SW_BANDS else LW_HALF
        print(f"  Extracting {sw_ref_band} patches ({2*half_ref+1}x{2*half_ref+1} px)...")
        patches_ref, centers, bg_med_ref, _ = extract_patches_single_band(
            sci_ref, wht_ref, half_ref, N_PATCHES, rng
        )
        all_band_patches[sw_ref_band].extend(patches_ref)

        # ── Extract other bands at aligned positions ───────────────────────
        other_bands = [b for b in bands if b != sw_ref_band]
        if other_bands:
            _, bg_med_ref_raw, _, _ = measure_background(sci_ref, wht_ref)

        for band in other_bands:
            band_pattern = patterns.get(band)
            if band_pattern is None:
                print(f"  Skipping {band}: no FITS path configured.")
                continue

            band_files = sorted(glob_module.glob(band_pattern, recursive=True))
            if not band_files:
                print(f"  Skipping {band}: no files matched {band_pattern}")
                continue

            print(f"  Extracting aligned {band} patches...")
            half_band = SW_HALF if band in SW_BANDS else LW_HALF

            sci_b, wht_b, wcs_b, _, pixar_b = load_fits_band(band_files[0])
            if pixar_b is not None:
                pixar_sr_map[band] = float(pixar_b)

            _, bg_med_b, _, _ = measure_background(sci_b, wht_b)
            aligned = []
            for ctr in centers:
                patch = extract_aligned_patch(
                    sci_b, wht_b, wcs_ref, wcs_b, ctr,
                    half_ref, half_band, bg_med_b
                )
                aligned.append(patch)  # may be None for edge patches

            # Replace None with zeros (invalid patches flagged separately)
            patch_size = 2 * half_band + 1
            for j, p in enumerate(aligned):
                if p is None:
                    aligned[j] = np.zeros((patch_size, patch_size), dtype=np.float32)

            all_band_patches[band].extend(aligned)
            print(f"    {len(aligned)} aligned patches extracted.")

    # ── Save per-band outputs ─────────────────────────────────────────────────
    npz_dict = {}
    for band in bands:
        arr = np.array(all_band_patches[band], dtype=np.float32)
        if len(arr) == 0:
            print(f"  WARNING: No patches for band {band}, skipping.")
            continue

        band_out = field_out / band
        band_out.mkdir(parents=True, exist_ok=True)
        out_path = band_out / "backgrounds.npy"
        np.save(str(out_path), arr)
        print(f"  Saved {band}: {arr.shape} -> {out_path}")
        npz_dict[band] = arr

    # ── Save multi-band NPZ ───────────────────────────────────────────────────
    if len(npz_dict) > 1:
        npz_path = field_out / "multiband_backgrounds.npz"
        np.savez_compressed(str(npz_path), **npz_dict)
        print(f"  Saved multiband NPZ -> {npz_path}")

    return all_band_patches, pixar_sr_map


def combine_fields(band: str = "F115W"):
    """Merge per-field backgrounds.npy arrays into a combined output."""
    print(f"\n=== Combining all fields for band {band} ===")
    combined_dir = Path(OUT_DIR) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for field_dir in sorted(Path(OUT_DIR).iterdir()):
        if not field_dir.is_dir() or field_dir.name == "combined":
            continue
        npy = field_dir / band / "backgrounds.npy"
        if npy.exists():
            arr = np.load(str(npy))
            parts.append(arr)
            print(f"  {field_dir.name}/{band}: {arr.shape}")

    if not parts:
        print(f"  No {band} background files found to combine.")
        return

    combined = np.concatenate(parts, axis=0)
    out_path = combined_dir / f"{band}_backgrounds.npy"
    np.save(str(out_path), combined)
    print(f"  Combined -> {out_path}  shape={combined.shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-field multi-band JWST background prep pipeline"
    )
    parser.add_argument("--field", default="jw01810",
                        help="Field name key (jw01810, ceers, etc.)")
    parser.add_argument("--bands", nargs="+", default=["F115W"],
                        help="Bands to extract (default: F115W)")
    parser.add_argument("--fits", nargs="*", default=None,
                        help="Override FITS file paths for the first band listed")
    parser.add_argument("--combine", action="store_true",
                        help="After processing, combine all fields into combined/")
    parser.add_argument("--combine-band", default="F115W",
                        help="Band to combine across fields (default: F115W)")
    args = parser.parse_args()

    fits_overrides = {}
    if args.fits:
        fits_overrides[args.bands[0]] = args.fits[0] if len(args.fits) == 1 else args.fits

    all_patches, pixar_map = process_field(args.field, args.bands, fits_overrides or None)

    # Save PIXAR_SR companion JSON
    if pixar_map:
        json_path = Path(OUT_DIR) / "pixar_sr.json"
        existing = {}
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
        existing.update(pixar_map)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  Saved PIXAR_SR metadata -> {json_path}")
        for band, val in pixar_map.items():
            print(f"    {band}: {val:.6e} sr/pix")

    if args.combine:
        combine_fields(args.combine_band)

    print("\n=== prep_multifield.py complete ===")
