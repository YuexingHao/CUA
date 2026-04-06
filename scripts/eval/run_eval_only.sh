#!/bin/bash
#SBATCH --job-name=qwen3-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=4:00:00
#SBATCH --output=results/slurm_logs/slurm-eval-%j.out

echo "=== Qwen3-8B LoRA Evaluation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1
python -m interaskill.eval_qwen

echo ""
echo "=== Done ==="
echo "Date: $(date)"
