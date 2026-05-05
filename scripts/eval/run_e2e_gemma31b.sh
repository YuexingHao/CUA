#!/bin/bash
#SBATCH --job-name=e2e-gemma
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-gemma-%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== Gemma-4-31B — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "google/gemma-4-31B-it" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
