#!/bin/bash
#SBATCH --job-name=qwen3-zs-bc
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-qwen3-zs-bc-%j.out

echo "=== Qwen3-8B Zero-Shot on BrowseComp-Plus (thinking fix) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

python -m interaskill.eval_model \
    --model Qwen/Qwen3-8B \
    --dataset bc --max-convs 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"
