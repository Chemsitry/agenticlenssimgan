"""
prep_cosmicwebb.py

Extracts background patches and PSF stars from COSMOS-Web (program 1727)
i2d files for F115W, F150W, F277W, F444W.

Output:
  prepped_cosmicwebb/
    F115W/  backgrounds.npy (N, 125, 125)  psf_median.npy (63, 63)
    F150W/  backgrounds.npy (N, 125, 125)  psf_median.npy (63, 63)
    F277W/  backgrounds.npy (N, 63, 63)    psf_median.npy (31, 31)
    F444W/  backgrounds.npy (N, 63, 63)    psf_median.npy (31, 31)
    band_info.json   — per-band PIXAR_SR, PHOTMJSR, XPOSURE, pixel_scale
"""

import json
import glob as glob_module
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# ── Config ─────────────────────────────────────────────────────────────────

BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
RAW_DIR = 'raw_data/1727'
OUT_DIR = 'prepped_cosmicwebb'

# SW: 0.031"/pix, LW: 0.063"/pix
SW_BANDS = {'F115W', 'F150W'}
LW_BANDS = {'F277W', 'F444W'}

# Patch sizes (match angular FoV: ~3.9" across)
SW_PATCH = 125   # 125 * 0.031 = 3.875"
LW_PATCH = 63    # 63 * 0.063 = 3.969"

# PSF stamp sizes
SW_PSF_STAMP = 63   # 63 * 0.031 = 1.95"
LW_PSF_STAMP = 31   # 31 * 0.063 = 1.95"

N_PATCHES = 2000    # per band (single detector ~2k x 2k)
N_PSF_TARGET = 50   # stars per band
SEED = 42


def get_fits_path(band):
    """Find the downloaded i2d FITS for this band."""
    files = sorted(glob_module.glob(f"{RAW_DIR}/{band}/**/*i2d*.fits", recursive=True))
    if not files:
        raise FileNotFoundError(f"No i2d FITS for {band} in {RAW_DIR}/{band}/")
    return files[0]


def load_band(fits_path):
    """Load SCI, WHT, and header info."""
    with fits.open(fits_path) as hdul:
        sci = hdul['SCI'].data.astype(np.float32)
        wht = hdul['WHT'].data.astype(np.float32)
        hdr = hdul['SCI'].header
        pri = hdul['PRIMARY'].header
        info = {
            'pixar_sr': float(hdr.get('PIXAR_SR', pri.get('PIXAR_SR', 0))),
            'photmjsr': float(hdr.get('PHOTMJSR', pri.get('PHOTMJSR', 0))),
            'xposure': float(hdr.get('XPOSURE', pri.get('XPOSURE', pri.get('EFFEXPTM', 0)))),
            'pixel_scale': abs(float(hdr.get('CD1_1', hdr.get('CDELT1', 0)))) * 3600,
        }
    return sci, wht, info


def extract_backgrounds(sci, wht, patch_half, n_patches, rng):
    """Extract sky patches, background-subtracted, in MJy/sr."""
    valid = np.isfinite(sci) & (wht > 0)
    subsample = sci[valid][::100]
    bg_mean, bg_median, bg_rms = sigma_clipped_stats(subsample, sigma=3.0, maxiters=5)

    patch_size = 2 * patch_half + 1
    ny, nx = sci.shape
    margin = patch_half + 1

    patches = []
    attempts = 0
    max_attempts = n_patches * 20

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        cy = int(rng.integers(margin, ny - margin))
        cx = int(rng.integers(margin, nx - margin))

        slc = sci[cy - patch_half:cy + patch_half + 1,
                   cx - patch_half:cx + patch_half + 1]
        wslc = wht[cy - patch_half:cy + patch_half + 1,
                    cx - patch_half:cx + patch_half + 1]

        if slc.shape != (patch_size, patch_size):
            continue
        # Require 95% valid pixels
        frac_valid = np.sum(np.isfinite(slc) & (wslc > 0)) / slc.size
        if frac_valid < 0.95:
            continue

        patch = slc.copy()
        patch -= bg_median
        patches.append(patch)

        if len(patches) % 500 == 0:
            print(f"    {len(patches)}/{n_patches} patches ({attempts} attempts)")

    patches = np.array(patches, dtype=np.float32)
    print(f"    Extracted {len(patches)} patches in {attempts} attempts "
          f"(bg_median={bg_median:.4f}, bg_rms={bg_rms:.6f} MJy/sr)")
    return patches, bg_median, bg_rms


def extract_psf_stars(sci, wht, stamp_half, n_target, bg_median, bg_rms):
    """Extract bright, isolated, compact point sources for PSF estimation."""
    valid = np.isfinite(sci) & (wht > 0)
    threshold = bg_median + 20 * bg_rms  # 20-sigma detection

    stamp_size = 2 * stamp_half + 1
    ny, nx = sci.shape
    margin = stamp_half + 2

    # Find bright pixels
    bright = (sci > threshold) & valid
    bright[:margin, :] = False
    bright[-margin:, :] = False
    bright[:, :margin] = False
    bright[:, -margin:] = False

    # Local maxima in 5x5
    from scipy.ndimage import maximum_filter
    local_max = (sci == maximum_filter(sci, size=5)) & bright

    ys, xs = np.where(local_max)
    # Sort by brightness (brightest first)
    order = np.argsort(sci[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]

    stamps = []
    used = np.zeros_like(sci, dtype=bool)

    for cy, cx in zip(ys, xs):
        if len(stamps) >= n_target:
            break
        # Isolation: no other bright pixel within stamp_half
        if used[cy - stamp_half:cy + stamp_half + 1,
                cx - stamp_half:cx + stamp_half + 1].any():
            continue

        stamp = sci[cy - stamp_half:cy + stamp_half + 1,
                    cx - stamp_half:cx + stamp_half + 1].copy()
        if stamp.shape != (stamp_size, stamp_size):
            continue
        if not np.all(np.isfinite(stamp)):
            continue

        # Compactness: ≥50% flux within r < 10 px
        stamp_sub = stamp - bg_median
        yy, xx = np.mgrid[:stamp_size, :stamp_size]
        r = np.sqrt((yy - stamp_half)**2 + (xx - stamp_half)**2)
        total = np.sum(stamp_sub)
        inner = np.sum(stamp_sub[r < min(10, stamp_half)])
        if total <= 0 or inner / total < 0.5:
            continue

        # Normalize
        stamp_sub /= np.sum(stamp_sub)
        stamps.append(stamp_sub.astype(np.float32))
        used[cy - stamp_half:cy + stamp_half + 1,
             cx - stamp_half:cx + stamp_half + 1] = True

    print(f"    Found {len(stamps)} PSF star stamps")
    return stamps


def build_median_psf(stamps):
    """Median-stack PSF stamps, filtering NaN stamps."""
    clean = [s for s in stamps if not np.any(np.isnan(s))]
    if len(clean) < 3:
        print(f"    WARNING: only {len(clean)} clean stamps")
        if not clean:
            return None
    stack = np.stack(clean, axis=0)
    median = np.median(stack, axis=0)
    median = np.clip(median, 0, None)
    median /= median.sum()
    print(f"    Median PSF from {len(clean)} stamps, peak at {np.unravel_index(np.argmax(median), median.shape)}")
    return median


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    band_info = {}

    for band in BANDS:
        print(f"\n{'='*60}")
        print(f"  Band: {band}")
        print(f"{'='*60}")

        fits_path = get_fits_path(band)
        print(f"  File: {os.path.basename(fits_path)}")
        sci, wht, info = load_band(fits_path)
        band_info[band] = info
        print(f"  Shape: {sci.shape}, pixel_scale={info['pixel_scale']:.4f}\"/pix")
        print(f"  PIXAR_SR={info['pixar_sr']:.6e}, PHOTMJSR={info['photmjsr']:.4f}, XPOSURE={info['xposure']:.1f}s")

        is_sw = band in SW_BANDS
        patch_half = SW_PATCH // 2 if is_sw else LW_PATCH // 2
        psf_half = SW_PSF_STAMP // 2 if is_sw else LW_PSF_STAMP // 2

        # Backgrounds
        print(f"\n  Extracting background patches ({2*patch_half+1}x{2*patch_half+1})...")
        patches, bg_med, bg_rms = extract_backgrounds(sci, wht, patch_half, N_PATCHES, rng)

        band_out = Path(OUT_DIR) / band
        band_out.mkdir(parents=True, exist_ok=True)
        np.save(str(band_out / 'backgrounds.npy'), patches)
        print(f"  Saved {band}/backgrounds.npy: {patches.shape}")

        # PSF
        print(f"\n  Extracting PSF stars ({2*psf_half+1}x{2*psf_half+1})...")
        stamps = extract_psf_stars(sci, wht, psf_half, N_PSF_TARGET, bg_med, bg_rms)
        if stamps:
            median_psf = build_median_psf(stamps)
            if median_psf is not None:
                np.save(str(band_out / 'psf_median.npy'), median_psf)
                print(f"  Saved {band}/psf_median.npy: {median_psf.shape}")

    # Save band info JSON
    json_path = Path(OUT_DIR) / 'band_info.json'
    with open(json_path, 'w') as f:
        json.dump(band_info, f, indent=2)
    print(f"\nSaved {json_path}")

    print("\n=== prep_cosmicwebb.py complete ===")


if __name__ == '__main__':
    main()
