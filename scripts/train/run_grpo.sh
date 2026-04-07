#!/bin/bash
#SBATCH --job-name=qwen3-grpo
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-grpo-%j.out

echo "=== GRPO Fine-Tuning: Qwen3-8B ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Train with GRPO
python -m interaskill.finetune_grpo

# Evaluate GRPO adapter on all datasets
echo ""
echo "=== Evaluating GRPO adapter ==="

for DATASET in iw wa bc; do
    echo ""
    echo "=== GRPO on $DATASET ==="
    python -m interaskill.eval_model \
        --model Qwen/Qwen3-8B \
        --adapter results/qwen3_grpo/final_adapter \
        --dataset $DATASET --max-convs 200 --resume
done

echo ""
echo "=== Done ==="
echo "Date: $(date)"
