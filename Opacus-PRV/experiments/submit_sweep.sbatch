#!/bin/bash
#SBATCH --job-name=beta-dpsgd
#SBATCH --array=0-3071%64
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#
# Total jobs: 4 datasets × 8 β × 6 σ × 2 batch × 2 LR × 4 clip = 3072
# Adjust --array upper bound if grid changes. Run:
#   python run_sweep.py --list-jobs
# to verify the count.
#
# Adjust %64 to control max concurrent jobs.
#
# Usage:
#   mkdir -p logs results
#   sbatch submit_sweep.sbatch

# ---- load your environment here ----
# module load cuda/11.x python/3.9
# source /path/to/your/venv/bin/activate

cd "$(dirname "$0")"

python run_sweep.py \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir results
