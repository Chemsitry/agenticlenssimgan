"""Render a flowchart explaining how Faber-Jackson calibration drives the v9 simulation."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / 'fj_flowchart.png'
PARAMS = json.loads((Path(__file__).parent / 'fj_params.json').read_text())

fig, ax = plt.subplots(figsize=(13, 14))
ax.set_xlim(0, 12); ax.set_ylim(0, 18); ax.axis('off')

# ---------- helpers
def box(cx, cy, w, h, text, color, fontsize=10, fontweight='normal', edge='#333'):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.15', linewidth=1.6,
                       edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

def arrow(x1, y1, x2, y2, label=None, label_pos='right', color='#222'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=22, lw=1.8, color=color))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx, dy = (0.15, 0) if label_pos == 'right' else (-0.15, 0)
        ha = 'left' if label_pos == 'right' else 'right'
        ax.text(mx + dx, my + dy, label, fontsize=9, style='italic',
                color='#444', ha=ha, va='center')

def banner(cy, text, color='#222'):
    ax.text(6, cy, text, ha='center', va='center', fontsize=15,
            fontweight='bold', color=color)

# ---------- title
ax.text(6, 17.4, 'How v9 uses Faber-Jackson to set the lens mass',
        ha='center', fontsize=17, fontweight='bold')
ax.text(6, 17.0, 'Two stages: calibrate b once from the 5 DESI×JWST galaxies, then use it for every simulated lens.',
        ha='center', fontsize=10.5, style='italic', color='#555')

# =========================================================================
# STAGE A — CALIBRATION (one-time)
# =========================================================================
ax.add_patch(FancyBboxPatch((0.2, 9.5), 11.6, 6.9,
             boxstyle='round,pad=0.1', linewidth=2.4,
             edgecolor='#1a5fb4', facecolor='#eaf2fc', zorder=0))
banner(16.0, 'STAGE A  —  Calibrate the FJ relation (one-time, already done)', color='#1a5fb4')

# Inputs row
box(2.0, 15.0, 3.2, 0.9,
    "5 galaxies with measured\nσ_v (DESI) + JWST mags + z",
    color='#cfe8ff', fontweight='bold', fontsize=10)
box(6.0, 15.0, 3.2, 0.9,
    "Faber-Jackson equation\nM_abs = -10·log₁₀(σ_v) + b",
    color='#ffe2cc', fontweight='bold', fontsize=10)
box(10.0, 15.0, 3.2, 0.9,
    "Planck18 cosmology\n→ distance modulus μ(z)",
    color='#fff2cf', fontweight='bold', fontsize=10)

# Process
box(6.0, 13.0, 11.0, 1.4,
    "For each galaxy and each filter F (115, 150, 277):\n"
    "  M_abs = m_apparent − μ(z)         (account for distance)\n"
    "  b_galaxy = M_abs + 10·log₁₀(σ_v)   (solve FJ for b)",
    color='#ffffff', fontsize=10.5)

arrow(2.0, 14.55, 5.0, 13.75, label=None)
arrow(6.0, 14.55, 6.0, 13.75)
arrow(10.0, 14.55, 7.0, 13.75)

# Result: per-band b
b115 = PARAMS['bands']['F115W']['intercept_median']
b150 = PARAMS['bands']['F150W']['intercept_median']
b277 = PARAMS['bands']['F277W']['intercept_median']
box(6.0, 11.2, 11.0, 1.5,
    "Take median b across the 5 galaxies → one number per filter:\n"
    f"      b(F115W) = {b115:+.2f}      b(F150W) = {b150:+.2f}      b(F277W) = {b277:+.2f}\n"
    "(Stored in fj_params.json — the calibration sample isn't needed again.)",
    color='#fcd5d5', fontweight='bold', fontsize=10.5)

arrow(6.0, 12.30, 6.0, 11.95)

# Arrow from Stage A down to Stage B
ax.add_patch(FancyArrowPatch((6.0, 9.6), (6.0, 8.4),
             arrowstyle='-|>', mutation_scale=32, lw=2.5, color='#1a5fb4'))
ax.text(6.4, 9.0, 'fj_params.json', fontsize=10, style='italic',
        color='#1a5fb4', ha='left', va='center')

# =========================================================================
# STAGE B — PER-SYSTEM USE (every simulated image)
# =========================================================================
ax.add_patch(FancyBboxPatch((0.2, 0.4), 11.6, 7.6,
             boxstyle='round,pad=0.1', linewidth=2.4,
             edgecolor='#1c7c2e', facecolor='#eafaee', zorder=0))
banner(7.6, 'STAGE B  —  For every simulated system, use b to set σ_v', color='#1c7c2e')

# Step 1: pick cutout
box(2.0, 6.5, 3.2, 1.0,
    "1. Pick a real\nJWST cutout from\nour 46 scenes",
    color='#cfe8ff', fontweight='bold', fontsize=10)

# Step 2: measure mags
box(6.0, 6.5, 3.4, 1.0,
    "2. Sum cutout pixels →\napparent AB mag in\nF115W, F150W, F277W",
    color='#cfe8ff', fontweight='bold', fontsize=10)

# Step 3: predict sigma
box(10.0, 6.5, 3.4, 1.0,
    "3. Solve FJ for σ_v:\nlog σ_v = (M_abs − b) / −10\n(one σ_v per filter)",
    color='#ffe2cc', fontweight='bold', fontsize=10)

arrow(3.6, 6.5, 4.3, 6.5)
arrow(7.7, 6.5, 8.3, 6.5)

# Step 4: median across bands
box(3.5, 4.5, 4.4, 1.0,
    "4. Median across 3 bands\n→ one σ_v per system\n(rejects cutouts with inconsistent colors)",
    color='#ffe2cc', fontweight='bold', fontsize=10)
arrow(10.0, 5.95, 5.5, 5.05)

# Step 5: SIE → theta_E
box(8.5, 4.5, 4.4, 1.0,
    "5. lenstronomy SIE:\nσ_v + z_lens + z_source → θ_E\n(Einstein radius)",
    color='#ffe2cc', fontweight='bold', fontsize=10)
arrow(5.7, 4.5, 6.3, 4.5)

# Step 6: filter
box(3.5, 2.4, 4.4, 1.0,
    "6. Reject if θ_E outside\n[0.5″, 1.5″] → retry with\na different cutout",
    color='#fff2cf', fontsize=10)
arrow(8.5, 3.95, 5.5, 2.95)

# Step 7: lens light + source light + render
box(8.5, 2.4, 4.4, 1.0,
    "7. Paint lensed arcs at\nthe Einstein radius onto\nthe real cutout → final image",
    color='#d4f0d4', fontweight='bold', fontsize=10)
arrow(5.7, 2.4, 6.3, 2.4)

# Final
box(6.0, 0.9, 7.0, 0.8,
    "Final image: a real lens galaxy + arcs whose size is physically consistent with it",
    color='#fcd5d5', fontweight='bold', fontsize=11)
arrow(6.0, 1.85, 6.0, 1.45)

plt.tight_layout()
plt.savefig(OUT, dpi=170, bbox_inches='tight')
print('wrote', OUT)
