#!/bin/bash
#SBATCH --job-name=e2e-qwen0
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-qwen0-%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== Qwen3-8B zero-shot — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
