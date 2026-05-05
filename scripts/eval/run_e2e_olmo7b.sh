#!/bin/bash
#SBATCH --job-name=e2e-olmo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-olmo-%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== OLMo-3-7B — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "allenai/Olmo-3-1025-7B" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
