#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_audit.sh — NERSC Perlmutter sbatch script
# Stage 0, Deliverable 2: compare sim vs real pixel distributions.
#
# Produces four figures in output/gan/baselines/:
#   pixel_histograms.png   radial_profiles.png
#   centroid_positions.png  power_spectra.png
#
# This is CPU-only and fast: ~5-15 minutes for 5000 images per class.
# No GPU, no special memory.
#
# Prerequisites:
#   - output/v3/images_{band}.npy      (run simulate_v3.py first)
#   - output/gan/real_cutouts/         (run run_prep_real_targets.sh first)
#
# Submit:
#   cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan
#   sbatch gan/slurm/run_audit.sh
#
# To run against the jitter-corrected sim (Stage 1 output):
#   sbatch gan/slurm/run_audit.sh --sim-dir output/v3_jitter
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=deepsrch
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB                # 5k x 4 bands x 125x125 x float32 ~ 1.5 GB each set
#SBATCH --time=00:30:00
#SBATCH --job-name=audit_dist
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.err

set -euo pipefail

cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan

PYTHON=/global/homes/f/forrestc/.conda/envs/lenssim/bin/python

echo "Python: $($PYTHON --version)"
echo "matplotlib: $($PYTHON -c 'import matplotlib; print(matplotlib.__version__)')"
echo "Start: $(date)"

# Check prerequisites
if [ ! -f "output/v3/images_F115W.npy" ]; then
    echo "ERROR: output/v3/images_F115W.npy not found."
    echo "       Run simulate_v3.py --n 5000 first."
    exit 1
fi
if [ ! -d "output/gan/real_cutouts" ]; then
    echo "ERROR: output/gan/real_cutouts/ not found."
    echo "       Run run_prep_real_targets.sh first."
    exit 1
fi

mkdir -p logs output/gan/baselines

$PYTHON -u -m gan.data.audit_distributions \
    --sim-dir output/v3 \
    --real-dir output/gan/real_cutouts \
    --out-dir output/gan/baselines \
    --n-sim 5000 \
    --n-real 5000 \
    --norm \
    "$@"

echo "Done: $(date)"
echo "Figures in output/gan/baselines/"
