"""
plot_seds.py — Plot SEDs for 10 lens galaxies + 10 source galaxies
across the 4 NIRCam bands, sampled from COWLS distributions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Band effective wavelengths (microns)
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
BAND_WAVES = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}
waves = [BAND_WAVES[b] for b in BANDS]

# SED functions (from simulate_v4.py)
def elliptical_color_ratios(z_lens):
    f150w = 1.3 + 0.4 * z_lens
    f277w = 1.8 + 1.2 * z_lens
    f444w = 2.0 + 1.8 * z_lens
    return {'F115W': 1.0, 'F150W': f150w, 'F277W': f277w, 'F444W': f444w}

def starforming_color_ratios(z_source, uv_slope=0.5):
    ly_break_um = 0.1216 * (1 + z_source)
    band_waves = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}
    ratios = {}
    for band, lam_eff in band_waves.items():
        if lam_eff < ly_break_um:
            suppression = max(0.0, (lam_eff / ly_break_um) ** 3)
            ratio = suppression * 0.05
        else:
            ratio = (1.15 / lam_eff) ** uv_slope
        ratios[band] = max(ratio, 0.01)
    f115w_val = ratios['F115W']
    if f115w_val > 0.01:
        for band in ratios:
            ratios[band] /= f115w_val
    else:
        max_val = max(ratios.values())
        if max_val > 0.01:
            for band in ratios:
                ratios[band] /= max_val
    return ratios

# Sample 10 lens redshifts from COWLS distribution
rng = np.random.default_rng(42)
z_lenses = []
for _ in range(10):
    z = float(truncnorm.rvs(
        a=(0.05 - 0.7) / 0.4, b=(2.5 - 0.7) / 0.4,
        loc=0.7, scale=0.4,
        random_state=int(rng.integers(int(1e9)))))
    z_lenses.append(z)
z_lenses.sort()

# Sample 10 source redshifts from COWLS distribution
z_sources = []
uv_slopes = []
for _ in range(10):
    z = float(truncnorm.rvs(
        a=(0.5 - 2.5) / 1.5, b=(7.0 - 2.5) / 1.5,
        loc=2.5, scale=1.5,
        random_state=int(rng.integers(int(1e9)))))
    z_sources.append(z)
    uv_slopes.append(float(np.clip(rng.normal(-0.1, 0.7), -1.5, 1.5)))
z_sources.sort()

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Lens SEDs (red/warm colormap)
cmap_lens = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 10))
for i, z in enumerate(z_lenses):
    ratios = elliptical_color_ratios(z)
    fluxes = [ratios[b] for b in BANDS]
    ax1.plot(waves, fluxes, 'o-', color=cmap_lens[i], linewidth=2, markersize=8,
             label=f'z = {z:.2f}')

ax1.set_xlabel('Wavelength (µm)', fontsize=13)
ax1.set_ylabel('Relative flux (normalized to F115W)', fontsize=13)
ax1.set_title('Lens galaxies (Elliptical SED)', fontsize=14)
ax1.legend(fontsize=9, title='$z_{lens}$', title_fontsize=10)
ax1.set_xticks(waves)
ax1.set_xticklabels(BANDS)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Source SEDs (blue/cool colormap)
cmap_src = plt.cm.cool(np.linspace(0.1, 0.9, 10))
for i, (z, uv) in enumerate(zip(z_sources, uv_slopes)):
    ratios = starforming_color_ratios(z, uv_slope=uv)
    fluxes = [ratios[b] for b in BANDS]
    style = '-' if fluxes[0] > 0.05 else '--'  # dashed if F115W is dropout
    ax2.plot(waves, fluxes, f'o{style}', color=cmap_src[i], linewidth=2, markersize=8,
             label=f'z = {z:.2f}, β = {uv:.1f}')

ax2.set_xlabel('Wavelength (µm)', fontsize=13)
ax2.set_ylabel('Relative flux (normalized to brightest band)', fontsize=13)
ax2.set_title('Source galaxies (Star-forming SED)', fontsize=14)
ax2.legend(fontsize=8, title='$z_{source}$, UV slope', title_fontsize=9)
ax2.set_xticks(waves)
ax2.set_xticklabels(BANDS)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

# Add Lyman break annotation
ax2.annotate('Lyman break\nenters F115W\nat z ~ 3.5',
             xy=(1.15, 0.05), fontsize=9, color='gray',
             ha='center', style='italic')

plt.suptitle('SED Color Ratios — COWLS-calibrated lens & source populations\n'
             '(10 lenses + 10 sources sampled from simulation distributions)',
             fontsize=14)
plt.tight_layout()
plt.savefig('output/validation/sed_plot.png', dpi=150, bbox_inches='tight')
print('Saved -> output/validation/sed_plot.png')
