#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_prep_mosaic.sh — NERSC Perlmutter sbatch script
# Runs prep_mosaic.py to extract backgrounds and PSFs from the COSMOS-Web
# DR0.5 mosaics.  This must run ONCE before anything else.
#
# Prerequisites:
#   raw_data/1727_mosaic/ must contain the COSMOS-Web FITS mosaics, either
#   directly or via a symlink to scratch:
#     ln -s /pscratch/sd/f/forrestc/cosmos_web_dr0.5 raw_data/1727_mosaic
#
# Outputs (in the working directory):
#   prepped_mosaic/band_info.json
#   prepped_mosaic/{band}/backgrounds.npy   (2000 sky patches per band)
#   prepped_mosaic/{band}/psf_median.npy    (empirical PSF per band)
#
# These outputs are ~500 MB total and live permanently on u2 — you do NOT
# need the raw mosaics again after this script completes.
#
# Runtime: ~1-2 hours (mostly FITS I/O across 4 large files).
# Memory:  the mosaics are memory-mapped, so RAM usage stays low (~8 GB).
#
# Submit:
#   cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan
#   sbatch gan/slurm/run_prep_mosaic.sh
#
# For 224-pixel patches (recommended by gan_plan.md Section 9):
#   sbatch gan/slurm/run_prep_mosaic.sh --size 224
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=deepsrch
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB               # mosaics are memory-mapped; peak ~8 GB
#SBATCH --time=03:00:00          # conservative; typically ~1.5 hours
#SBATCH --job-name=prep_mosaic
#SBATCH --output=logs/prep_mosaic_%j.out
#SBATCH --error=logs/prep_mosaic_%j.err

set -euo pipefail

cd /global/u2/f/forrestc/agenticlenssim/agenticlenssimgan

PYTHON=/global/homes/f/forrestc/.conda/envs/lenssim/bin/python

echo "Python: $($PYTHON --version)"
echo "astropy: $($PYTHON -c 'import astropy; print(astropy.__version__)')"
echo "Start: $(date)"
echo "Node: $SLURMD_NODENAME"

# Verify raw data exists before wasting queue time
if [ ! -d "raw_data/1727_mosaic" ]; then
    echo "ERROR: raw_data/1727_mosaic/ not found."
    echo "  Either download mosaics to /pscratch/sd/f/forrestc/cosmos_web_dr0.5/"
    echo "  and symlink: ln -s /pscratch/sd/f/forrestc/cosmos_web_dr0.5 raw_data/1727_mosaic"
    exit 1
fi

# Check at least one band exists
if [ -z "$(ls raw_data/1727_mosaic/F115W/mosaic*.fits 2>/dev/null)" ]; then
    echo "ERROR: No mosaic*.fits found in raw_data/1727_mosaic/F115W/"
    echo "  Expected structure:"
    echo "    raw_data/1727_mosaic/F115W/mosaic_F115W.fits"
    echo "    raw_data/1727_mosaic/F150W/mosaic_F150W.fits"
    echo "    raw_data/1727_mosaic/F277W/mosaic_F277W.fits"
    echo "    raw_data/1727_mosaic/F444W/mosaic_F444W.fits"
    exit 1
fi

mkdir -p logs prepped_mosaic

$PYTHON -u prep_mosaic.py "$@"

echo "Done: $(date)"
echo ""
echo "Next steps:"
echo "  1. Run simulate_v3.py to generate sim images"
echo "  2. sbatch gan/slurm/run_prep_real_targets.sh"
