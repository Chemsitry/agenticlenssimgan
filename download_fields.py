"""
download_fields.py

Downloads JWST NIRCam i2d mosaics from MAST for a given program.
Default: CEERS (proposal 1345), F115W only.

Usage:
    pip install astroquery
    python download_fields.py
    python download_fields.py --proposal 1345 --filter F115W
    python download_fields.py --proposal 1345 --filter F115W F150W F200W

Downloads to: raw_data/<proposal_id>/<filter>/

Notes:
  - CEERS tiles are named hlsp_ceers_jwst_nircam_nircam*_<filt>_v0.5_i2d.fits.gz
  - astropy opens .fits.gz directly with memmap=False
  - First run downloads ~2-4 GB per filter
  - calib_level=3 fetches the fully reduced, drizzled mosaics
"""

import argparse
import os
from pathlib import Path


def download_jwst_mosaics(proposal_id: str, filters: list[str]) -> None:
    try:
        from astroquery.mast import Observations
    except ImportError:
        raise SystemExit(
            "astroquery not installed. Run: pip install astroquery"
        )

    for filt in filters:
        out_dir = Path(f"raw_data/{proposal_id}/{filt}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Querying MAST: proposal={proposal_id}, filter={filt} ===")

        obs = Observations.query_criteria(
            obs_collection="JWST",
            proposal_id=proposal_id,
            dataproduct_type="image",
            calib_level=3,
            filters=filt,
        )

        if obs is None or len(obs) == 0:
            print(f"  No observations found for proposal {proposal_id}, filter {filt}")
            continue

        print(f"  Found {len(obs)} observation(s)")

        products = Observations.get_product_list(obs)
        print(f"  Total products: {len(products)}")

        # Keep only SCIENCE extension i2d files
        filtered = Observations.filter_products(
            products,
            productType="SCIENCE",
            extension="fits",
        )

        # Narrow to i2d products specifically
        i2d_mask = [
            "i2d" in str(row["productFilename"]) for row in filtered
        ]
        i2d = filtered[i2d_mask]
        print(f"  i2d SCIENCE products: {len(i2d)}")

        if len(i2d) == 0:
            print(f"  No i2d products found — skipping {filt}")
            continue

        print(f"  Downloading to {out_dir} ...")
        manifest = Observations.download_products(
            i2d,
            download_dir=str(out_dir),
        )
        print(f"  Download manifest:")
        for row in manifest:
            status = row.get("Status", "?")
            local  = row.get("Local Path", "?")
            print(f"    [{status}] {local}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download JWST NIRCam i2d mosaics from MAST"
    )
    parser.add_argument(
        "--proposal", default="1345",
        help="MAST proposal ID (default: 1345 = CEERS)"
    )
    parser.add_argument(
        "--filter", dest="filters", nargs="+",
        default=["F115W"],
        help="One or more NIRCam filter names (default: F115W)"
    )
    args = parser.parse_args()

    download_jwst_mosaics(args.proposal, args.filters)
