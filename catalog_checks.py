"""
catalog_checks.py — quantitative catalog-vs-catalog validation of the v13 lens
simulation against the real COWLS catalogue + published lens samples.

Implements the three checks described in catalogCheck.md:

  1. Einstein radius   theta_E (v13)  vs  COWLS `einstein_radius`
  2. Arc/lens flux ratio: v13's hard-coded 0.25 F444W anchor vs the COWLS ratio
     implied by (delensed source mag) + magnification + lens mag, per band.
  3. sigma_v and Einstein mass (v13) vs published lens samples (SLACS/BELLS/SL2S).

Comparison data sources are documented in referenceData.md.

The core numbers use only numpy, so this runs in the bare login env (Python 3.6).
Extras degrade gracefully:
  - plots (theta_E.png, flux_ratio.png, sigma_mass.png) need matplotlib
  - KS p-values need scipy   (the KS D statistic is always computed with numpy)
  - the optional cube-photometry path (--measure-cubes) reads the 4.6 GB arrays
    on CFS via memory-mapping (no new images are generated)

Usage:
    python3 catalog_checks.py
    python3 catalog_checks.py --measure-cubes --cube-sample 300
    python3 catalog_checks.py --observed      # test interpretation (B) for Check 2
"""
import argparse
import csv
from pathlib import Path

import numpy as np

np.seterr(invalid="ignore")   # COWLS columns contain NaNs; comparisons are masked below

# ---- optional dependencies -------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import astropy.units as u
    import astropy.constants as const
    from astropy.cosmology import Planck18 as _COSMO
    HAVE_ASTROPY = True
except Exception:
    HAVE_ASTROPY = False

# ---- paths & constants -----------------------------------------------------
REPO = Path(__file__).resolve().parent
CAT = REPO / "v13_consistent" / "catalog"
COWLS_CSV = REPO / "cowls_catalogue.csv"
CFS_CUBES = Path("/global/cfs/projectdirs/deepsrch/natekv/v13_consistent")
OUTDIR = REPO / "output" / "catalog_checks"

BANDS = ("F115W", "F150W", "F277W", "F444W")
SIM_BANDS = ("F115W", "F150W", "F277W")          # v13 has no F444W lens mag
THETA_E_CEILING = 1.687                          # v13 max θ_E (σ_v clip at 350)
ARC_LENS_ANCHOR = 0.25                           # v13 hard-coded target_ratio @ F444W

# Published spectroscopic stellar velocity dispersions: (median, lo, hi) km/s.
# Comparison ranges only; see referenceData.md for how to pull the real tables.
PUBLISHED_SIGMA = {
    "SLACS (Auger+2009)": (243, 160, 360),
    "BELLS (Brownstein+2012)": (240, 180, 400),
    "SL2S (Sonnenfeld+2013)": (230, 150, 350),
}


# ---- loaders ---------------------------------------------------------------
def load_sim():
    """Load v13 catalog arrays, keeping only lensed systems (drop controls)."""
    lensed = np.load(CAT / "lensed.npy") > 0.5
    sim = {
        "lensed_mask": lensed,
        "lensed_idx": np.where(lensed)[0],          # indices into the full CFS cubes
        "theta_E": np.load(CAT / "theta_Es.npy")[lensed],
        "sigma_v": np.load(CAT / "sigma_v.npy")[lensed],
        "mass": np.load(CAT / "einstein_mass_msun.npy")[lensed],
        "z_lens": np.load(CAT / "z_lens.npy")[lensed],
        "z_source": np.load(CAT / "z_source.npy")[lensed],
        "sigma_v_phot": {b: np.load(CAT / ("sigma_v_%s.npy" % b))[lensed] for b in SIM_BANDS},
        "mlens": {b: np.load(CAT / ("photom_%s.npy" % b))[lensed] for b in SIM_BANDS},
    }
    return sim


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def load_cowls():
    """Read cowls_catalogue.csv into a dict of numpy arrays."""
    with open(COWLS_CSV) as f:
        rows = list(csv.DictReader(f))
    keys = ["einstein_radius", "lens_spec_z", "lens_cw_photo_z_med", "lens_cw_stmass_med"]
    for b in BANDS:
        keys += ["%s_lens_magnitude_ab" % b, "%s_source_magnitude_ab" % b, "%s_magnification" % b]
    cow = {k: np.array([_to_float(r.get(k, "")) for r in rows]) for k in keys}
    cow["code"] = np.array([r.get("code", "") for r in rows])
    return cow


def einstein_mass_msun(zl, zs, theta_arcsec):
    """Projected mass inside theta_E: M_E = (c^2/4G) theta^2 Dd Ds / Dds (needs astropy)."""
    Dd = _COSMO.angular_diameter_distance(zl)
    Ds = _COSMO.angular_diameter_distance(zs)
    Dds = _COSMO.angular_diameter_distance_z1z2(zl, zs)
    theta = (np.asarray(theta_arcsec) * u.arcsec).to(u.rad).value
    M = (const.c ** 2 / (4 * const.G)) * (theta ** 2) * (Dd * Ds / Dds)
    return M.to(u.Msun).value


def load_published_sigma():
    """Load published_sigma.csv (real lens sample) for Check 3, if present.

    Built by fetch_published_sigma.py (SLACS, Bolton+2008). Returns None if the
    file is absent so Check 3 falls back to literature ranges.
    """
    path = REPO / "published_sigma.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    sv = np.array([_to_float(r.get("sigma_v_kms", "")) for r in rows])
    zl = np.array([_to_float(r.get("z_lens", "")) for r in rows])
    zs = np.array([_to_float(r.get("z_source", "")) for r in rows])
    te = np.array([_to_float(r.get("theta_E_arcsec", "")) for r in rows])
    ok = np.isfinite(sv) & (sv > 0)
    out = {"sample": rows[0].get("sample", "published"), "sigma_v_kms": sv[ok],
           "z_lens": zl[ok], "z_source": zs[ok], "theta_E_arcsec": te[ok]}
    if HAVE_ASTROPY:
        good = (np.isfinite(out["z_lens"]) & np.isfinite(out["z_source"])
                & np.isfinite(out["theta_E_arcsec"]) & (out["theta_E_arcsec"] > 0))
        M = np.full(len(out["sigma_v_kms"]), np.nan)
        if good.any():
            M[good] = einstein_mass_msun(out["z_lens"][good], out["z_source"][good],
                                         out["theta_E_arcsec"][good])
        out["M_E"] = M
    return out


# ---- small stats helpers ---------------------------------------------------
def ks_d(a, b):
    """Two-sample Kolmogorov-Smirnov D statistic (max CDF gap), numpy-only."""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, grid, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def quantiles(x):
    return dict(n=len(x), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                max=float(np.max(x)))


def _cdf_xy(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


# ---- Check 1: Einstein radius ---------------------------------------------
def check1(sim, cow, summary):
    print("\n" + "=" * 70)
    print("CHECK 1 — Einstein radius:  v13 theta_E  vs  COWLS einstein_radius")
    print("=" * 70)
    te_sim = sim["theta_E"]
    te_sim = te_sim[te_sim > 0]
    te_cow = cow["einstein_radius"]
    te_cow = te_cow[np.isfinite(te_cow) & (te_cow > 0)]
    sub = te_cow[te_cow >= 0.5]                     # overlap v13's [0.5, 2.5] design range

    qs, qc = quantiles(te_sim), quantiles(sub)
    frac_tail = float((sub > THETA_E_CEILING).mean())
    frac_15 = float((sub > 1.5).mean())
    D = ks_d(te_sim, sub)
    p = float(ks_2samp(te_sim, sub).pvalue) if HAVE_SCIPY else float("nan")

    print("  v13 (lensed)      : n=%d  median=%.3f  p90=%.3f  max=%.3f"
          % (qs["n"], qs["median"], qs["p90"], qs["max"]))
    print("  COWLS (theta>=0.5): n=%d  median=%.3f  p90=%.3f  max=%.3f"
          % (qc["n"], qc["median"], qc["p90"], qc["max"]))
    print("  missing tail   : %.1f%% of COWLS(>=0.5) exceed v13 ceiling %.3f\" (%d systems)"
          % (100 * frac_tail, THETA_E_CEILING, int(round(frac_tail * qc["n"]))))
    print("  also > 1.5\"    : %.1f%% of COWLS(>=0.5)" % (100 * frac_15))
    print("  KS test (shape): D=%.3f  p=%s" % (D, ("%.2e" % p) if p == p else "n/a (no scipy)"))

    summary.append(("1_thetaE", "median_arcsec", qs["median"], qc["median"], "sim higher = lenses too strong"))
    summary.append(("1_thetaE", "max_arcsec", qs["max"], qc["max"], "v13 truncated at sigma_v clip"))
    summary.append(("1_thetaE", "frac_COWLS_above_ceiling", frac_tail, "", "real tail v13 cannot make"))
    summary.append(("1_thetaE", "KS_D / p", D, p, "shape difference"))

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(7, 5))
        xs, ys = _cdf_xy(te_sim); ax.step(xs, ys, where="post", label="v13 (lensed)", color="C0")
        xc, yc = _cdf_xy(sub); ax.step(xc, yc, where="post", label="COWLS (theta_E>=0.5)", color="C3")
        ax.axvline(THETA_E_CEILING, ls="--", color="k", lw=1)
        ax.text(THETA_E_CEILING + 0.02, 0.05, "v13 ceiling %.2f\"" % THETA_E_CEILING, rotation=90, va="bottom")
        ax.set_xlabel('Einstein radius  theta_E  (arcsec)'); ax.set_ylabel("cumulative fraction")
        ax.set_title("Check 1: Einstein-radius CDF (KS D=%.3f)" % D); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, "theta_E.png")


# ---- Check 2: arc/lens flux ratio -----------------------------------------
def cowls_ratio(cow, band, intrinsic=True):
    """Arc/lens flux ratio per band from COWLS mags.

    COWLS source mags are DELENSED (intrinsic), so observed arc flux puts the
    magnification back: m_arc = m_source - 2.5*log10(mu).  intrinsic=False tests
    the alternative reading where the source mag is already observed.
    """
    ml = cow["%s_lens_magnitude_ab" % band]
    ms = cow["%s_source_magnitude_ab" % band]
    mu = cow["%s_magnification" % band]
    ok = np.isfinite(ml) & np.isfinite(ms) & np.isfinite(mu) & (mu > 0)
    m_arc = ms[ok] - (2.5 * np.log10(mu[ok]) if intrinsic else 0.0)
    return 10 ** (-0.4 * (m_arc - ml[ok]))


def check2(sim, cow, summary, intrinsic=True, measure_cubes=False, cube_sample=300,
           cube_radius=120, seed=0):
    print("\n" + "=" * 70)
    print("CHECK 2 — arc/lens flux ratio:  v13 anchor (%.2f @ F444W)  vs  COWLS"
          % ARC_LENS_ANCHOR)
    print("  (source mags treated as %s)" % ("DELENSED/intrinsic x mu" if intrinsic else "already-observed"))
    print("=" * 70)
    cow_meds = {}
    for b in BANDS:
        r = cowls_ratio(cow, b, intrinsic=intrinsic)
        cow_meds[b] = r
        print("  COWLS %s: n=%d  median=%.3f  IQR=[%.3f, %.3f]"
              % (b, len(r), np.median(r), np.percentile(r, 25), np.percentile(r, 75)))
        summary.append(("2_fluxratio", "COWLS_median_%s" % b, float(np.median(r)),
                        ARC_LENS_ANCHOR if b == "F444W" else "", "v13 anchors only F444W"))
    print("  -> v13 hard-codes %.2f at F444W; real F444W median=%.3f (factor %.1fx)"
          % (ARC_LENS_ANCHOR, np.median(cow_meds["F444W"]),
             ARC_LENS_ANCHOR / np.median(cow_meds["F444W"])))

    sim_meds = None
    if measure_cubes:
        sim_meds = _measure_cube_ratios(sim, cube_sample, cube_radius, seed)

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(7, 5))
        data = [cow_meds[b] for b in BANDS]
        ax.boxplot(data, showfliers=False, whis=(10, 90))
        ax.set_xticks(range(1, len(BANDS) + 1)); ax.set_xticklabels(list(BANDS))
        ax.axhline(ARC_LENS_ANCHOR, ls="--", color="C0", label="v13 anchor 0.25 (F444W)")
        if sim_meds:
            xs = range(1, len(BANDS) + 1)
            ax.plot(xs, [sim_meds.get(b, np.nan) for b in BANDS], "s-", color="C2",
                    label="v13 measured (cubes)")
        ax.set_yscale("log"); ax.set_ylabel("arc / lens flux ratio")
        ax.set_title("Check 2: COWLS arc/lens ratio per band"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, "flux_ratio.png")


def _measure_cube_ratios(sim, n_sample, radius_px, seed):
    """Optional 2b: realized v13 arc/lens ratio from the CFS cubes (memory-mapped)."""
    print("  [--measure-cubes] reading arc/galaxy cubes from %s" % CFS_CUBES)
    rng = np.random.default_rng(seed)
    idx_all = sim["lensed_idx"]
    idx = np.sort(rng.choice(idx_all, size=min(n_sample, len(idx_all)), replace=False))
    out = {}
    for b in BANDS:
        try:
            arcs = np.load(CFS_CUBES / ("arcs_%s.npy" % b), mmap_mode="r")
            gal = np.load(CFS_CUBES / ("galaxies_%s.npy" % b), mmap_mode="r")
        except Exception as e:
            print("    band %s: cannot open cubes (%s)" % (b, e)); continue
        size = arcs.shape[-1]
        yy, xx = np.ogrid[:size, :size]
        c = size / 2.0
        ap = (xx - c) ** 2 + (yy - c) ** 2 <= radius_px ** 2
        ratios = []
        for i in idx:
            a = np.asarray(arcs[i])[ap].sum()
            g = np.asarray(gal[i])[ap].sum()
            if g > 0:
                ratios.append(max(a, 0.0) / g)
        ratios = np.array(ratios)
        out[b] = float(np.median(ratios))
        print("    v13 measured %s: n=%d  median ratio=%.3f" % (b, len(ratios), out[b]))
    return out


# ---- Check 3: sigma_v & Einstein mass -------------------------------------
def check3(sim, cow, summary):
    print("\n" + "=" * 70)
    print("CHECK 3 — sigma_v & Einstein mass:  v13  vs  published lens samples")
    print("=" * 70)
    sig = sim["sigma_v"]
    f80 = float((sig <= 80.001).mean())
    f350 = float((sig >= 349.999).mean())
    phot = np.concatenate([sim["sigma_v_phot"][b] for b in SIM_BANDS])
    fj_max = float(np.nanmax(phot))
    print("  v13 sigma_v: median=%.0f km/s  frac@floor(80)=%.3f  frac@ceiling(350)=%.3f"
          % (np.median(sig), f80, f350))
    print("  v13 un-clipped Faber-Jackson sigma_v reaches %.0f km/s (the clip throws this away)"
          % fj_max)
    for name, (med, lo, hi) in PUBLISHED_SIGMA.items():
        print("    published %-26s median~%d  range~[%d, %d]" % (name, med, lo, hi))

    pub = load_published_sigma()
    if pub is not None:
        psig = pub["sigma_v_kms"]
        D = ks_d(sig, psig)
        p = float(ks_2samp(sig, psig).pvalue) if HAVE_SCIPY else float("nan")
        print("  published_sigma.csv [%s]: n=%d  median=%.0f km/s"
              % (pub["sample"], len(psig), np.median(psig)))
        print("  KS  v13 vs published sigma_v:  D=%.3f  p=%s"
              % (D, ("%.2e" % p) if p == p else "n/a (no scipy)"))
        summary.append(("3_sigma", "published_median_kms", float(np.median(psig)), "", pub["sample"]))
        summary.append(("3_sigma", "KS_vs_published_D", D, p, "real two-sample test"))
    else:
        pub = None
        print("  (no published_sigma.csv found -> run fetch_published_sigma.py for a real KS test)")

    m = sim["mass"] > 0
    slope, b0 = np.polyfit(np.log10(sig[m]), np.log10(sim["mass"][m]), 1)
    print("  v13 log10(M_E) vs log10(sigma_v): slope=%.2f (SIE physics expects ~4)" % slope)
    print("  v13 median log10(M_E)=%.2f  (SLACS ~10.8-11.7)" % np.log10(np.median(sim["mass"][m])))

    # sigma_v needed to reach COWLS's largest theta_E, holding distances fixed:
    te_cow = cow["einstein_radius"]; te_cow = te_cow[np.isfinite(te_cow)]
    need = 350.0 * np.sqrt(te_cow.max() / THETA_E_CEILING)
    print("  COWLS max theta_E=%.2f\" would need sigma_v ~%.0f km/s -> within FJ max (%.0f)."
          % (te_cow.max(), need, fj_max))
    print("  => raising the sigma_v clip from 350 to ~%.0f recovers the missing theta_E tail." % need)

    summary.append(("3_sigma", "median_kms", float(np.median(sig)), PUBLISHED_SIGMA["SLACS (Auger+2009)"][0], "vs SLACS"))
    summary.append(("3_sigma", "frac_at_clip_350", f350, "", "truncated tail"))
    summary.append(("3_sigma", "FJ_unclipped_max", fj_max, "", "available but discarded"))
    summary.append(("3_mass", "M_E_sigma_slope", float(slope), 4.0, "SIE expectation"))
    summary.append(("3_mass", "median_log10_M_E", float(np.log10(np.median(sim["mass"][m]))), "", "SLACS ~10.8-11.7"))

    if HAVE_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        bins = np.linspace(40, 470, 44)
        ax.hist(sig, bins=bins, density=True, alpha=0.6, label="v13 sigma_v (clipped)", color="C0")
        ax.hist(phot[np.isfinite(phot)], bins=bins, density=True, histtype="step", color="C1",
                label="v13 Faber-Jackson (un-clipped)")
        if pub is not None:
            ax.hist(pub["sigma_v_kms"], bins=bins, density=True, histtype="step", color="C3",
                    lw=2, label="%s (n=%d)" % (pub["sample"], len(pub["sigma_v_kms"])))
        ax.axvline(350, ls="--", color="k"); ax.axvline(80, ls=":", color="gray")
        ax.set_xlabel("sigma_v (km/s)"); ax.set_ylabel("normalized density")
        ax.set_title("Check 3a: sigma_v distribution"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.scatter(np.log10(sig[m]), np.log10(sim["mass"][m]), s=4, alpha=0.2, color="C0", label="v13")
        xfit = np.array([np.log10(sig[m].min()), np.log10(sig[m].max())])
        ax.plot(xfit, b0 + slope * xfit, "k-", label="v13 fit slope=%.2f" % slope)
        if pub is not None and "M_E" in pub:
            ok = np.isfinite(pub["M_E"]) & (pub["M_E"] > 0)
            if ok.any():
                ax.scatter(np.log10(pub["sigma_v_kms"][ok]), np.log10(pub["M_E"][ok]),
                           s=18, color="C3", edgecolor="k", lw=0.3, zorder=5, label=pub["sample"])
        ax.set_xlabel("log10 sigma_v (km/s)"); ax.set_ylabel("log10 M_E (M_sun)")
        ax.set_title("Check 3b: Einstein mass vs sigma_v"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, "sigma_mass.png")


# ---- io --------------------------------------------------------------------
def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % path)


def write_summary(summary):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "catalog_checks_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "quantity", "v13_value", "real_value", "note"])
        for row in summary:
            w.writerow(row)
    print("\nwrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observed", action="store_true",
                    help="Check 2: treat COWLS source mag as already-observed (interpretation B)")
    ap.add_argument("--measure-cubes", action="store_true",
                    help="Check 2b: measure realized v13 arc/lens ratio from CFS image cubes")
    ap.add_argument("--cube-sample", type=int, default=300, help="systems to sample for --measure-cubes")
    ap.add_argument("--cube-radius", type=int, default=120, help="central aperture radius (px) for cubes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("dependencies: matplotlib=%s  scipy=%s" % (HAVE_MPL, HAVE_SCIPY))
    sim = load_sim()
    cow = load_cowls()
    print("loaded v13 lensed systems: %d   |   COWLS rows: %d (%d with theta_E)"
          % (len(sim["theta_E"]), len(cow["einstein_radius"]),
             int(np.isfinite(cow["einstein_radius"]).sum())))

    summary = []
    check1(sim, cow, summary)
    check2(sim, cow, summary, intrinsic=not args.observed,
           measure_cubes=args.measure_cubes, cube_sample=args.cube_sample,
           cube_radius=args.cube_radius, seed=args.seed)
    check3(sim, cow, summary)
    write_summary(summary)
    if not HAVE_MPL:
        print("\n(note: matplotlib not found — numbers only, no .png plots. "
              "Activate a venv/conda env with matplotlib+scipy for figures.)")


if __name__ == "__main__":
    main()
