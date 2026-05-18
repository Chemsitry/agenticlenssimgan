# Galaxy catalogs used by data prep v9_consistent

Two CSV files describe every galaxy used in the pipeline:

## 1. `calibration_galaxies.csv`
49 galaxies that supplied **DESI σ_v + JWST photometry + z_spec** for fitting the Faber-Jackson relation in `fit_calibration.py`.

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
1015 galaxies that supplied **JWST cutouts** for use as the foreground lens in simulations. These do *not* need a DESI σ_v — σ_v is derived from photometry via the FJ calibration above.

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
