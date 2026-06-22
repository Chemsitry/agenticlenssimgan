"""
fetch_published_sigma.py — build published_sigma.csv for Check 3 of
catalog_checks.py from a real, citable strong-lens sample.

Source: the Sloan Lens ACS Survey (SLACS) Paper V, Bolton et al. 2008
(ApJ 682, 964), via VizieR catalogue J/ApJ/682/964. We merge:
  - table4 "SLACS HST-ACS target observational data": sigma (SDSS velocity
    dispersion, km/s), zFG (lens redshift), zBG (source redshift), Lens grade
  - table5 "grade-A strong lens model parameters": bSIE (SIE Einstein radius, ")
on the system Name, keeping grade-A confirmed lenses with a measured sigma.

Requires only the Python standard library + outbound HTTPS (run on a NERSC login
node, which has internet). Writes published_sigma.csv to the repo root.

    python3 fetch_published_sigma.py

Note: SLACS sigma is the SDSS fibre value (uncorrected for aperture). It is a
spectroscopic *stellar* dispersion, conceptually close to v13's SIE/Faber-Jackson
sigma for ellipticals but not identical (see referenceData.md).
"""
import csv
from pathlib import Path
from urllib.request import urlopen

URL = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
       "-source=J/ApJ/682/964/%s&-out.max=unlimited&-out.all")
OUT = Path(__file__).resolve().parent / "published_sigma.csv"


def fetch_table(name):
    """Download one VizieR TSV table and return a list of dict rows."""
    text = urlopen(URL % name, timeout=90).read().decode("utf-8", "replace")
    cols = None
    in_body = False
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if cols is None:                      # first data line = column names
            cols = [c.strip() for c in line.split("\t")]
            continue
        if not in_body:                       # skip units line, then dashes line
            if set(line.replace("\t", "").strip()) <= set("-") and "-" in line:
                in_body = True
            continue
        vals = line.split("\t")
        vals += [""] * (len(cols) - len(vals))
        rows.append({c: v.strip() for c, v in zip(cols, vals)})
    return rows


def main():
    print("fetching SLACS Bolton+2008 (VizieR J/ApJ/682/964) ...")
    t4 = fetch_table("table4")
    t5 = fetch_table("table5")
    bsie = {r["Name"]: r.get("bSIE", "") for r in t5 if r.get("Name")}

    out = []
    for r in t4:
        if r.get("Lens", "") != "A":          # confirmed grade-A lenses only
            continue
        if not r.get("sigma", ""):            # require a measured dispersion
            continue
        name = r.get("Name", "")
        out.append({
            "sample": "SLACS (Bolton+2008)",
            "name": name,
            "sigma_v_kms": r.get("sigma", ""),
            "sigma_v_err": r.get("e_sigma", ""),
            "z_lens": r.get("zFG", ""),
            "z_source": r.get("zBG", ""),
            "theta_E_arcsec": bsie.get(name, ""),
            "grade": r.get("Lens", ""),
        })

    fields = ["sample", "name", "sigma_v_kms", "sigma_v_err",
              "z_lens", "z_source", "theta_E_arcsec", "grade"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print("wrote %s with %d grade-A SLACS lenses" % (OUT, len(out)))


if __name__ == "__main__":
    main()
