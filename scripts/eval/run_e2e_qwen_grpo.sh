#!/bin/bash
#SBATCH --job-name=e2e-grpo
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-grpo-%j.out

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== Qwen3-8B GRPO — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" --adapter "results/qwen3_grpo/final_adapter" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
