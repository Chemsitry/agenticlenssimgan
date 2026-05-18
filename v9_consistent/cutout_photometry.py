"""
cutout_photometry.py — convert a lens cutout (sim units) into an AB magnitude.

Lens cutouts in data prep v8 are stored in 'sim units', defined by
  sim_value = MJy_per_sr × pixar_sr × 1e15 × sum_to_flux
with sum_to_flux = 6.501853565914121.

Equivalently, the total flux (in MJy) of a region is
  F_MJy = sum(sim_values_in_region) / (1e15 × sum_to_flux)

To get AB magnitude:
  F_uJy = F_MJy × 1e12
  m_AB  = 23.9 - 2.5 × log10(F_uJy)

Aperture: by default sum a centered circular aperture of given radius.
This excludes most flux from neighboring objects that occasionally land
in the 201x201 cutout while capturing the central galaxy's halo.
"""

from __future__ import annotations
import numpy as np

SUM_TO_FLUX = 6.501853565914121          # same constant simulate_v8 / prep_lenses_v8 use
SIM_TO_MJY  = 1.0 / (1e15 * SUM_TO_FLUX)  # multiply a sum-of-sim-units by this → MJy
MJY_TO_UJY  = 1e12                        # MJy → microJy
AB_ZP_UJY   = 23.9                        # m_AB = 23.9 - 2.5 log10(flux/µJy)


def circular_aperture_mask(shape, radius_pix, center=None):
    """Boolean mask of pixels inside a circle of given radius (in pixels)."""
    h, w = shape
    if center is None:
        center = (h / 2.0 - 0.5, w / 2.0 - 0.5)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = center
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_pix ** 2


def cutout_flux_mjy(stamp, aperture_radius_pix=60, center=None,
                    sky_annulus=(70, 95)):
    """
    Total flux (MJy) inside a circular aperture, with optional local-sky
    subtraction from an outer annulus. Local sky should be near zero
    because prep_lenses_v8 subtracts the global background median, but we
    still do a residual-sky pass to be safe.

    Parameters
    ----------
    stamp : 2-D ndarray in sim units
    aperture_radius_pix : radius of source aperture (default 60 px = 1.8")
    sky_annulus : (r_in, r_out) for residual-sky annulus, or None to skip

    Returns
    -------
    flux_mjy, sky_per_pix_mjy
    """
    src_mask = circular_aperture_mask(stamp.shape, aperture_radius_pix, center)
    sky_per_pix_sim = 0.0
    if sky_annulus is not None:
        r_in, r_out = sky_annulus
        sky_mask = circular_aperture_mask(stamp.shape, r_out, center) & \
                   ~circular_aperture_mask(stamp.shape, r_in, center)
        if sky_mask.sum() > 0:
            sky_per_pix_sim = float(np.median(stamp[sky_mask]))
    src_sum_sim = float((stamp[src_mask] - sky_per_pix_sim).sum())
    flux_mjy = src_sum_sim * SIM_TO_MJY
    sky_mjy  = sky_per_pix_sim * SIM_TO_MJY
    return flux_mjy, sky_mjy


def cutout_ab_mag(stamp, aperture_radius_pix=60, center=None, sky_annulus=(70, 95)):
    """AB magnitude of source inside aperture. Returns np.nan if flux <= 0."""
    flux_mjy, _ = cutout_flux_mjy(stamp, aperture_radius_pix, center, sky_annulus)
    if flux_mjy <= 0:
        return float('nan')
    flux_ujy = flux_mjy * MJY_TO_UJY
    return AB_ZP_UJY - 2.5 * np.log10(flux_ujy)


def cutout_ab_mags_all_bands(stamps_per_band, **kwargs):
    """Convenience: dict of {band: stamp} → dict of {band: m_AB}."""
    return {band: cutout_ab_mag(stamp, **kwargs) for band, stamp in stamps_per_band.items()}
