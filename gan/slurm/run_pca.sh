#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pca.sh — NERSC Perlmutter sbatch script
# Stage 0, Deliverable 3: PCA + logistic regression baseline.
#
# Produces:
#   output/gan/baselines/pca_results.json       — THE BASELINE ACCURACY
#   output/gan/baselines/pca_top_components.png — what PCA "sees"
#   output/gan/baselines/pca_projection.png     — scatter of sim vs real in PC space
#
# The CV accuracy in pca_results.json is the number every later stage must beat.
# Read the printed interpretation at the end of the job output file.
#
# Memory: 10k images x 4 bands x 125x125 x 4 bytes = ~3.1 GB per class.
# PCA on a 20k x 62500 matrix uses ~10 GB RAM peak (randomized SVD).
# 64 GB is safe.  Do NOT reduce --mem below 32 GB or the job will OOM silently.
#
# Prerequisites:
#   - output/v3/images_{band}.npy          (run simulate_v3.py first)
#   - output/gan/real_cutouts/             (run run_prep_real_targets.sh first)
#   - output/gan/real_cutouts/normalization.json
#
# Submit:
#   cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan
#   sbatch gan/slurm/run_pca.sh
#
# For a quick check with fewer images (less memory, faster):
#   sbatch gan/slurm/run_pca.sh --n-sim 2000 --n-real 2000
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=deepsrch
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # sklearn's randomized PCA benefits from threads
#SBATCH --mem=64GB                # 20k x 62500 float32 matrix ≈ 5 GB; PCA SVD ≈ 10 GB peak
#SBATCH --time=01:00:00
#SBATCH --job-name=pca_baseline
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err

set -euo pipefail

cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan

PYTHON=/global/homes/f/forrestc/.conda/envs/lenssim/bin/python

echo "Python: $($PYTHON --version)"
echo "sklearn: $($PYTHON -c 'import sklearn; print(sklearn.__version__)')"
echo "numpy:   $($PYTHON -c 'import numpy; print(numpy.__version__)')"
echo "Start: $(date)"

# Check prerequisites
if [ ! -f "output/v3/images_F115W.npy" ]; then
    echo "ERROR: output/v3/images_F115W.npy not found."
    exit 1
fi
if [ ! -f "output/gan/real_cutouts/normalization.json" ]; then
    echo "ERROR: output/gan/real_cutouts/normalization.json not found."
    echo "       Run run_prep_real_targets.sh first."
    exit 1
fi

mkdir -p logs output/gan/baselines

# Set threading for numpy/sklearn — use the CPUs we asked for
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON -u -m gan.baselines.pca \
    --sim-dir output/v3 \
    --real-dir output/gan/real_cutouts \
    --out-dir output/gan/baselines \
    --n-sim 10000 \
    --n-real 10000 \
    --n-components 50 \
    --cv-folds 5 \
    --seed 42 \
    "$@"

echo "Done: $(date)"
echo "Baseline accuracy in output/gan/baselines/pca_results.json"
