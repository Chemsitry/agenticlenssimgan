"""
export_csvs.py — write human-readable CSV catalogs of every galaxy used in the
v9_consistent pipeline.

Outputs:
  calibration_galaxies.csv  — the galaxies used to fit Faber-Jackson
  lens_scene_galaxies.csv   — the galaxies used as lens cutouts in simulations
  README_catalogs.md        — column documentation
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

V9_ROOT = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent')

# ─── Calibration sample ────────────────────────────────────────────────
sf_path = V9_ROOT / 'sample_final_extended.parquet'
if not sf_path.exists():
    sf_path = Path('/Users/nathankvinnesland/Desktop/desi_jwst_dev/cache/sample_final.parquet')

sf = pd.read_parquet(sf_path)
print(f'calibration sample: {len(sf)} galaxies from {sf_path.name}')

# Drop NaN-photometry rows (these are excluded from the FJ fit)
sf_clean = sf.dropna(subset=['m115', 'm150', 'm277', 'VDISP', 'Z']).reset_index(drop=True)
sf_clean.insert(0, 'role', 'calibration')
sf_clean = sf_clean.rename(columns={
    'RA': 'ra_deg', 'DEC': 'dec_deg',
    'Z': 'z_spec_desi', 'Z_PHOT_MEAN': 'z_phot_legacy',
    'VDISP': 'sigma_v_kms', 'VDISP_IVAR': 'sigma_v_ivar',
    'm115': 'F115W_AB', 'm150': 'F150W_AB', 'm277': 'F277W_AB',
    'm115_err': 'F115W_AB_err', 'm150_err': 'F150W_AB_err', 'm277_err': 'F277W_AB_err',
    'TARGETID': 'desi_targetid',
    'field': 'source_field',
    'jwst_match_arcsec': 'jwst_match_arcsec',
    'match_arcsec':      'desi_legacy_match_arcsec',
})
cal_cols = ['role','source_field','ra_deg','dec_deg','z_spec_desi','z_phot_legacy',
            'sigma_v_kms','sigma_v_ivar',
            'F115W_AB','F115W_AB_err','F150W_AB','F150W_AB_err','F277W_AB','F277W_AB_err',
            'desi_targetid','desi_legacy_match_arcsec','jwst_match_arcsec']
cal_cols = [c for c in cal_cols if c in sf_clean.columns]
cal_csv = V9_ROOT / 'calibration_galaxies.csv'
sf_clean[cal_cols].to_csv(cal_csv, index=False, float_format='%.6f')
print(f'  wrote {cal_csv}  ({len(sf_clean)} rows)')

# ─── Lens scenes ────────────────────────────────────────────────────────
manifest_path = V9_ROOT / 'prepped_scenes_v10' / 'manifest.parquet'
if not manifest_path.exists():
    manifest_path = V9_ROOT / 'prepped_scenes_v9' / 'manifest.parquet'

mf = pd.read_parquet(manifest_path)
print(f'\nlens scenes: {len(mf)} galaxies from {manifest_path}')
mf2 = mf.copy()
mf2.insert(0, 'role', 'lens_scene')
mf2 = mf2.rename(columns={
    'RA': 'ra_deg', 'DEC': 'dec_deg',
    'Z_PHOT_MEAN': 'z_phot_legacy', 'Z_PHOT_STD': 'z_phot_err',
    'TYPE': 'legacy_morphology',
    'field': 'source_field',
    'F444W_peak': 'F444W_peak_sim_units',
    'compactness': 'compactness_F444W',
})
ls_cols = ['role','source_field','scene_idx','ra_deg','dec_deg','z_phot_legacy','z_phot_err',
           'legacy_morphology','F444W_peak_sim_units','compactness_F444W','has_F444W']
ls_cols = [c for c in ls_cols if c in mf2.columns]
ls_csv = V9_ROOT / 'lens_scene_galaxies.csv'
mf2[ls_cols].to_csv(ls_csv, index=False, float_format='%.6f')
print(f'  wrote {ls_csv}  ({len(mf2)} rows)')

# ─── Documentation README ──────────────────────────────────────────────
readme = V9_ROOT / 'README_catalogs.md'
readme.write_text(f"""# Galaxy catalogs used by data prep v9_consistent

Two CSV files describe every galaxy used in the pipeline:

## 1. `calibration_galaxies.csv`
{len(sf_clean)} galaxies that supplied **DESI σ_v + JWST photometry + z_spec** for fitting the Faber-Jackson relation in `fit_calibration.py`.

Columns:
- **role** — always `calibration`
- **source_field** — which JWST field the galaxy came from (`jades_gdn`, `ceers_egs`, `primer_uds`, `primer_cosmos`, or `cosmos_web_mosaic`/`primer_cosmos_mosaic` if mag was measured from the mosaic rather than the published catalog)
- **ra_deg / dec_deg** — sky position (J2000, degrees)
- **z_spec_desi** — spectroscopic redshift from DESI DR1
- **z_phot_legacy** — photometric redshift from Legacy Survey (Zhou+ VAC)
- **sigma_v_kms** — stellar velocity dispersion in km/s (DESI FastSpecFit `VDISP`)
- **sigma_v_ivar** — inverse variance on σ_v
- **F115W_AB / F150W_AB / F277W_AB** — JWST NIRCam AB magnitudes
- **F***_AB_err** — magnitude uncertainties (NaN where measured from mosaic without a catalog error)
- **desi_targetid** — DESI internal target ID
- **desi_legacy_match_arcsec** — angular separation between DESI position and matched Legacy Survey object
- **jwst_match_arcsec** — angular separation to matched JWST catalog source (0 if mag came from direct mosaic photometry)

## 2. `lens_scene_galaxies.csv`
{len(mf2)} galaxies that supplied **JWST cutouts** for use as the foreground lens in simulations. These do *not* need a DESI σ_v — σ_v is derived from photometry via the FJ calibration above.

Columns:
- **role** — always `lens_scene`
- **source_field** — JWST mosaic the cutout was taken from
- **scene_idx** — index into the prepped scene arrays (`prepped_scenes_v10/scenes_*.npy`)
- **ra_deg / dec_deg** — sky position
- **z_phot_legacy / z_phot_err** — Legacy Survey photo-z + uncertainty
- **legacy_morphology** — `DEV / EXP / SER / REX` (the LS galaxy-type classification used to construct the parent sample; no morphology filtering beyond "any galaxy shape" was applied)
- **F444W_peak_sim_units** — central F444W brightness (used as the brightness-cut threshold; > 100 sim units to make sure lens is visible)
- **compactness_F444W** — sum(r<3px) / sum(r<15px) in F444W (< 0.35 to reject stars/AGN point sources)
- **has_F444W** — whether F444W data was available (false → that band was zero-padded)

## Where each sample is used
- **`fit_calibration.py`** reads `sample_final_extended.parquet` → produces `fj_params.json`
- **`simulate_v9_consistent.py`** reads `prepped_scenes_v10/manifest.parquet` → uses each scene's photo-z as `z_lens` and derives σ_v from its photometry + the fitted FJ calibration
""")
print(f'wrote {readme}')

# Print quick summaries
print()
print('Calibration breakdown:')
print(sf_clean['source_field'].value_counts().to_string())
print()
print('Lens scene breakdown:')
print(mf2['source_field'].value_counts().to_string())
