#!/bin/bash
#SBATCH --job-name=gemma31bc
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/slurm-gemma31bc-%j.out

echo "=== Gemma-4-31B on BrowseComp-Plus ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

python -m interaskill.eval_model \
    --model google/gemma-4-31B-it \
    --dataset bc --max-convs 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"
