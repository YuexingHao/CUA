#!/bin/bash
#SBATCH --job-name=qwen3-lora
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=7-00:00:00
#SBATCH --output=results/slurm-finetune-%j.out

echo "=== Qwen3-8B LoRA Fine-Tuning on Multi-Turn Conversations ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

# Activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda activate base

# Step 1: Generate conversation data (if not already generated)
if [ ! -f data/interaskill/conversations/train_conversations.jsonl ]; then
    echo "=== Step 1: Generating conversation data ==="
    python scripts/data/generate_conversations.py --num 1000
    echo ""
fi

# Step 2: LoRA Fine-Tuning
echo "=== Step 2: LoRA Fine-Tuning ==="
python -m interaskill.finetune_qwen

# Step 3: Evaluation
echo ""
echo "=== Step 3: Evaluation ==="
python -m interaskill.eval_qwen

echo ""
echo "=== Done ==="
echo "Date: $(date)"
