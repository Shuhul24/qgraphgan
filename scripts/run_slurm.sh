#!/bin/bash
#SBATCH --job-name=qgraphgan
#SBATCH --output=qgraphgan_%j.out
#SBATCH --error=qgraphgan_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
# Add --partition / --account lines for your cluster if required.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qenv

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python -u run_experiments.py "$@"
