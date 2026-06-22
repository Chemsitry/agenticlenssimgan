"""
build_catalog_csv.py — flatten the v13 catalog arrays in catalog/ into one tidy
human-readable table: v13_catalog.csv (one row per simulated system).

Why: the per-quantity .npy files are convenient for code but hard to eyeball.
This makes a single CSV you can open in a spreadsheet or load with pandas.

Run:
    python3 v13_consistent/build_catalog_csv.py
(uses only numpy + the standard library, so it works in the bare environment.)
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CAT = HERE / "catalog"

# column name -> array file (all length-N, aligned by index)
COLUMNS = {
    "lensed": "lensed.npy",
    "completed": "completed_mask.npy",
    "z_lens": "z_lens.npy",
    "z_source": "z_source.npy",
    "theta_E_arcsec": "theta_Es.npy",
    "sigma_v_kms": "sigma_v.npy",
    "sigma_v_F115W_kms": "sigma_v_F115W.npy",
    "sigma_v_F150W_kms": "sigma_v_F150W.npy",
    "sigma_v_F277W_kms": "sigma_v_F277W.npy",
    "einstein_mass_msun": "einstein_mass_msun.npy",
    "lens_mag_F115W_ab": "photom_F115W.npy",
    "lens_mag_F150W_ab": "photom_F150W.npy",
    "lens_mag_F277W_ab": "photom_F277W.npy",
}


def main() -> None:
    arrays = {name: np.load(CAT / fname) for name, fname in COLUMNS.items()}
    n = len(next(iter(arrays.values())))
    assert all(len(a) == n for a in arrays.values()), "arrays are not the same length"

    out = HERE / "v13_catalog.csv"
    names = ["index", *COLUMNS.keys()]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            row = [i]
            for name in COLUMNS:
                v = arrays[name][i]
                if isinstance(v, np.bool_):
                    row.append(int(v))
                else:
                    row.append(f"{float(v):.6g}")
            w.writerow(row)

    print(f"wrote {out}  ({n} rows, {len(names)} columns)")


if __name__ == "__main__":
    main()
