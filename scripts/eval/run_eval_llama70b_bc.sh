#!/bin/bash
#SBATCH --job-name=llama70bc
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-llama70bc-%j.out

echo "=== Llama-3.1-70B on BrowseComp-Plus ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

python -m interaskill.eval_model \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset bc --max-convs 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"
