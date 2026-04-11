#!/bin/bash
#SBATCH --job-name=e2e-gemma
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-gemma-%j.out

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== Gemma-4-31B — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "google/gemma-4-31B-it" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
