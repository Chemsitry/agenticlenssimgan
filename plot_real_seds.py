"""
plot_real_seds.py — Side-by-side comparison of real vs simulated SED color ratios.

Left column:  Real COWLS II lenses & sources (from PyAutoLens catalogue)
Right column: Simulated lenses & sources (sampled from our pipeline distributions)
"""

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import os

# ── Config ─────────────────────────────────────────────────────────────
BANDS = ['F115W', 'F150W', 'F277W', 'F444W']
BAND_WAVES = {'F115W': 1.15, 'F150W': 1.50, 'F277W': 2.77, 'F444W': 4.44}
waves = np.array([BAND_WAVES[b] for b in BANDS])

# ── Our SED models (from simulate_v4.py) ──────────────────────────────
def elliptical_color_ratios(z_lens):
    # Recalibrated against COWLS II PyAutoLens photometry (15 lenses)
    f150w = 1.68 + 0.09 * z_lens
    f277w = 8.79 + 8.40 * z_lens
    f444w = 0.42 + 18.11 * z_lens
    return [1.0, f150w, f277w, f444w]

def starforming_color_ratios(z_source, uv_slope=0.5):
    # Recalibrated against COWLS II PyAutoLens source photometry (15 sources)
    ly_break_um = 0.1216 * (1 + z_source)
    ratios = {}
    for band, lam_eff in BAND_WAVES.items():
        if lam_eff < ly_break_um:
            suppression = max(0.0, (lam_eff / ly_break_um) ** 3)
            ratio = suppression * 0.05
        else:
            lam_rest = lam_eff / (1 + z_source)
            if lam_rest > 0.35:
                break_boost = 1.0 + 3.0 * min((lam_rest - 0.35) / 0.3, 1.0)
            else:
                break_boost = 1.0
            uv_part = (1.15 / lam_eff) ** uv_slope
            ratio = uv_part * break_boost
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
    return [ratios[b] for b in BANDS]

# ── Load real COWLS catalogue ──────────────────────────────────────────
id_map = {
    'COSJ095914+021219': 'A', 'COSJ095917+015424': 'B', 'COSJ095920+015851': 'C',
    'COSJ095921+020638': 'D', 'COSJ095950+022057': 'E', 'COSJ095953+023319': 'F',
    'COSJ095955+021900': 'G', 'COSJ100012+022015': 'H', 'COSJ100013+023424': 'I',
    'COSJ100018+022138': 'J', 'COSJ100024+015334': 'K', 'COSJ100024+021749': 'L',
    'COSJ100025+015245': 'M', 'COSJ100028+021928': 'N', 'COSJ100047+015023': 'O',
    'COSJ100119+014849': 'P', 'COSJ100121+022740': 'Q',
}

def safe_float(s):
    try:
        v = float(s)
        return v if v > 0 and v < 90 else None
    except (ValueError, TypeError):
        return None

def get_flux_ratios(row, prefix):
    mags = []
    for b in BANDS:
        m = safe_float(row.get(f'{b}_{prefix}_magnitude_ab', ''))
        if m is None:
            return None
        mags.append(m)
    fluxes = [10**(-0.4 * m) for m in mags]
    return [f / fluxes[0] for f in fluxes]

rows = []
with open('cowls_catalogue.csv') as f:
    for row in csv.DictReader(f):
        if row['score'] == 'M25':
            rows.append(row)

real_lens = []   # (label, z, flux_ratios)
real_source = [] # (label, flux_ratios)

for row in rows:
    code = row['code']
    sid = id_map.get(code, '?')
    z_spec = safe_float(row.get('lens_spec_z', ''))
    z_phot = safe_float(row.get('lens_cw_photo_z_med', ''))
    z = z_spec if z_spec else z_phot

    lr = get_flux_ratios(row, 'lens')
    if lr is not None and z is not None:
        real_lens.append((sid, z, lr))

    sr = get_flux_ratios(row, 'source')
    if sr is not None:
        # Skip F115W dropouts (mag > 30 → normalizing to F115W blows up)
        f115_mag = safe_float(row.get('F115W_source_magnitude_ab', ''))
        if f115_mag is not None and f115_mag > 30:
            print(f'  Skipping source {sid}: F115W dropout (mag={f115_mag:.1f})')
            continue
        real_source.append((sid, z, sr))

real_lens.sort(key=lambda x: x[1])
real_source.sort(key=lambda x: (x[1] if x[1] is not None else 99))

print(f'Real: {len(real_lens)} lens SEDs, {len(real_source)} source SEDs')

# ── Generate simulated SEDs (same distributions as pipeline) ───────────
rng = np.random.default_rng(42)
N_SIM = 15  # similar count to the 15-16 real ones

sim_lens = []
for _ in range(N_SIM):
    z = float(truncnorm.rvs(
        a=(0.05 - 0.7) / 0.4, b=(2.5 - 0.7) / 0.4,
        loc=0.7, scale=0.4,
        random_state=int(rng.integers(int(1e9)))))
    sim_lens.append((z, elliptical_color_ratios(z)))
sim_lens.sort(key=lambda x: x[0])

sim_source = []
for _ in range(N_SIM):
    z = float(truncnorm.rvs(
        a=(0.5 - 2.5) / 1.5, b=(7.0 - 2.5) / 1.5,
        loc=2.5, scale=1.5,
        random_state=int(rng.integers(int(1e9)))))
    uv = float(np.clip(rng.normal(-0.5, 1.0), -2.5, 1.5))
    sim_source.append((z, uv, starforming_color_ratios(z, uv)))
sim_source.sort(key=lambda x: x[0])

print(f'Simulated: {N_SIM} lens SEDs, {N_SIM} source SEDs')

# ── Plot: 2x2 ─────────────────────────────────────────────────────────
os.makedirs('output/validation', exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# ── Top-left: Real lens SEDs ──────────────────────────────────────────
ax = axes[0, 0]
cmap_l = plt.cm.YlOrRd
z_min_l = min(d[1] for d in real_lens)
z_max_l = max(d[1] for d in real_lens)
for sid, z, ratios in real_lens:
    c = cmap_l(0.25 + 0.65 * (z - z_min_l) / (z_max_l - z_min_l + 1e-6))
    ax.plot(waves, ratios, 'o-', color=c, linewidth=2, markersize=7,
            label=f'{sid}: z={z:.2f}', alpha=0.85)
ax.set_ylabel('Relative flux (norm. to F115W)', fontsize=12)
ax.set_title('Real lens galaxies (COWLS II, Mahler+2025)', fontsize=13)
ax.legend(fontsize=7, ncol=2, title='$z_{lens}$', title_fontsize=9)
ax.set_xticks(waves); ax.set_xticklabels(BANDS)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ── Top-right: Simulated lens SEDs ────────────────────────────────────
ax = axes[0, 1]
z_min_sl = min(d[0] for d in sim_lens)
z_max_sl = max(d[0] for d in sim_lens)
for z, ratios in sim_lens:
    c = cmap_l(0.25 + 0.65 * (z - z_min_sl) / (z_max_sl - z_min_sl + 1e-6))
    ax.plot(waves, ratios, 'o-', color=c, linewidth=2, markersize=7,
            label=f'z={z:.2f}', alpha=0.85)
ax.set_title('Simulated lens galaxies (our pipeline)', fontsize=13)
ax.legend(fontsize=7, ncol=2, title='$z_{lens}$', title_fontsize=9)
ax.set_xticks(waves); ax.set_xticklabels(BANDS)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ── Bottom-left: Real source SEDs ─────────────────────────────────────
ax = axes[1, 0]
cmap_s = plt.cm.cool
z_vals_s = [d[1] for d in real_source if d[1] is not None]
z_min_s, z_max_s = min(z_vals_s), max(z_vals_s)
for sid, z, ratios in real_source:
    zc = z if z is not None else 1.0
    c = cmap_s(0.1 + 0.8 * (zc - z_min_s) / (z_max_s - z_min_s + 1e-6))
    zl = f'z_l={z:.2f}' if z else 'z=?'
    style = '-' if ratios[0] > 0.05 else '--'
    ax.plot(waves, ratios, f'o{style}', color=c, linewidth=2, markersize=7,
            label=f'{sid}: {zl}', alpha=0.85)
ax.set_xlabel('Wavelength (µm)', fontsize=12)
ax.set_ylabel('Relative flux (norm. to F115W)', fontsize=12)
ax.set_title('Real source galaxies (COWLS II, Mahler+2025)', fontsize=13)
ax.legend(fontsize=6.5, ncol=2, title='Source (colored by $z_{lens}$)', title_fontsize=8)
ax.set_xticks(waves); ax.set_xticklabels(BANDS)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ── Bottom-right: Simulated source SEDs ───────────────────────────────
ax = axes[1, 1]
z_min_ss = min(d[0] for d in sim_source)
z_max_ss = max(d[0] for d in sim_source)
for z, uv, ratios in sim_source:
    c = cmap_s(0.1 + 0.8 * (z - z_min_ss) / (z_max_ss - z_min_ss + 1e-6))
    style = '-' if ratios[0] > 0.05 else '--'
    ax.plot(waves, ratios, f'o{style}', color=c, linewidth=2, markersize=7,
            label=f'z={z:.2f}, β={uv:.1f}', alpha=0.85)
ax.set_xlabel('Wavelength (µm)', fontsize=12)
ax.set_title('Simulated source galaxies (our pipeline)', fontsize=13)
ax.legend(fontsize=6.5, ncol=2, title='$z_{source}$, UV slope', title_fontsize=8)
ax.set_xticks(waves); ax.set_xticklabels(BANDS)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.suptitle('SED Color Ratios: Real COWLS II Data vs Our Simulation Pipeline',
             fontsize=15, y=1.01)
plt.tight_layout()
outpath = 'output/validation/real_vs_sim_seds.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'\nSaved -> {outpath}')
