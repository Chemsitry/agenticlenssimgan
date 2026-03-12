"""
build_psf.py

Builds a clean, lenstronomy-ready empirical PSF kernel from the star stamps
collected by prep_jwst.py.

Input:  prepped/psf_stars.npy  — (200, 63, 63) float32 stamps
Output: prepped/psf_median.npy — (63, 63) float32, normalized to sum=1.0

Usage:
    python build_psf.py [--stamps prepped/psf_stars.npy] [--out prepped/psf_median.npy]
"""

import argparse
import numpy as np
import os


def build_psf_kernel(stamps_path: str, out_path: str) -> None:
    print(f"Loading PSF stamps from {stamps_path} ...")
    stamps = np.load(stamps_path)                          # (N, 63, 63)
    print(f"  Loaded {stamps.shape[0]} stamps, shape {stamps.shape[1]}x{stamps.shape[2]}")

    # Drop stamps with any NaN pixels (mosaic edge artefacts)
    nan_mask = np.isnan(stamps).any(axis=(1, 2))
    n_bad = nan_mask.sum()
    print(f"  Dropping {n_bad} stamps containing NaNs -> {stamps.shape[0] - n_bad} clean stamps")
    clean = stamps[~nan_mask]

    if clean.shape[0] == 0:
        raise RuntimeError("No clean PSF stamps remain after NaN filtering.")

    # Median stack (robust against faint companions and cosmic rays)
    kernel = np.median(clean, axis=0)                      # (63, 63)

    # Normalize so that the kernel sums to exactly 1.0
    total = kernel.sum()
    if total <= 0:
        raise RuntimeError(f"Median kernel has non-positive sum ({total}). Check input stamps.")
    kernel = kernel / total

    # Sanity checks
    assert kernel.shape == (63, 63), f"Unexpected kernel shape: {kernel.shape}"
    assert not np.isnan(kernel).any(), "NaNs present in output kernel"
    assert not np.isinf(kernel).any(), "Infs present in output kernel"
    center_y, center_x = np.unravel_index(np.argmax(kernel), kernel.shape)
    print(f"  Peak pixel: ({center_y}, {center_x})  — expected near (31, 31)")
    print(f"  Kernel sum: {kernel.sum():.8f}  (should be ~1.0)")
    print(f"  Kernel min: {kernel.min():.6e}   max: {kernel.max():.6e}")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    np.save(out_path, kernel.astype(np.float32))
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build empirical PSF kernel from star stamps")
    parser.add_argument("--stamps", default="prepped/psf_stars.npy",
                        help="Path to psf_stars.npy (default: prepped/psf_stars.npy)")
    parser.add_argument("--out",    default="prepped/psf_median.npy",
                        help="Output path (default: prepped/psf_median.npy)")
    args = parser.parse_args()

    build_psf_kernel(args.stamps, args.out)
    print("\nDone.")
