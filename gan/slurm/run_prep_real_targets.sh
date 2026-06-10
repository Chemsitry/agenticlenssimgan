#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_prep_real_targets.sh — NERSC Perlmutter sbatch script
# Stage 0, Deliverable 1: extract ~30k COSMOS-Web real cutouts.
#
# This job is CPU-only.  It opens 4 large FITS mosaics (memory-mapped, so
# they don't all load into RAM at once) and writes ~7.5 GB of .npy output.
# 64 GB of RAM is ample.  Estimated wall time: 30-90 minutes depending on
# how many cutouts you request and mosaic I/O speed.
#
# Submit:
#   cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan
#   sbatch gan/slurm/run_prep_real_targets.sh
#
# Override defaults at submit time:
#   sbatch gan/slurm/run_prep_real_targets.sh --n 10000 --size 125
#
# To run a quick smoke test first (50 cutouts, no queue):
#   srun --account=deepsrch --constraint=cpu --qos=interactive --time=00:15:00 \
#        --nodes=1 --cpus-per-task=4 --mem=32GB \
#        /global/homes/f/forrestc/.conda/envs/lenssim/bin/python \
#        -m gan.data.prep_real_targets --smoke-test
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=deepsrch
#SBATCH --constraint=cpu          # CPU partition — no GPU needed for data prep
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # 4 CPUs: one per band for I/O concurrency
#SBATCH --mem=64GB                # 4 FITS mosaics (memmap) + 7.5 GB output
#SBATCH --time=02:00:00           # 2 hours is conservative; typically ~1 hour
#SBATCH --job-name=prep_real
#SBATCH --output=logs/prep_real_%j.out
#SBATCH --error=logs/prep_real_%j.err

set -euo pipefail   # exit on error, undefined variable, or pipe failure

# ── Working directory ─────────────────────────────────────────────────────────
# All paths in the Python scripts are relative to here.
cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan

# ── Python interpreter ────────────────────────────────────────────────────────
PYTHON=/global/homes/f/forrestc/.conda/envs/lenssim/bin/python

# Verify the interpreter and key packages exist before waiting in queue
echo "Python: $($PYTHON --version)"
echo "numpy: $($PYTHON -c 'import numpy; print(numpy.__version__)')"
echo "astropy: $($PYTHON -c 'import astropy; print(astropy.__version__)')"
echo "Working dir: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Verify raw_data exists before queuing was a waste of time
if [ ! -d "raw_data/1727_mosaic" ]; then
    echo "ERROR: raw_data/1727_mosaic/ not found."
    echo "       The COSMOS-Web DR0.5 mosaics must be present at this path."
    exit 1
fi

# Verify prepped_mosaic/band_info.json exists
if [ ! -f "prepped_mosaic/band_info.json" ]; then
    echo "ERROR: prepped_mosaic/band_info.json not found."
    echo "       Run: sbatch gan/slurm/run_prep_mosaic.sh  (or prep_mosaic.py directly)"
    exit 1
fi

mkdir -p logs output/gan/real_cutouts

# ── Run ───────────────────────────────────────────────────────────────────────
# Pass any additional arguments from the sbatch command line through "$@"
$PYTHON -u -m gan.data.prep_real_targets \
    --n 30000 \
    --size 125 \
    --seed 99 \
    "$@"

echo "Done: $(date)"
