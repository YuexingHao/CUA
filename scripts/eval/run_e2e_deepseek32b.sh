#!/bin/bash
#SBATCH --job-name=e2e-dsr1
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-dsr1-%j.out

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

echo "=== DeepSeek-R1-Distill-32B — composition (test_task) ==="
echo "Node: $(hostname) | Date: $(date)"
python -m interaskill.eval_mind2web \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --split test_task --max-tasks 200 --resume --composition
echo "Done: $(date)"
