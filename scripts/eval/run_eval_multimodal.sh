#!/bin/bash
#SBATCH --job-name=mm-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-multimodal-%j.out

echo "=== Multimodal Evaluation Suite ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1
export HF_HOME=/orcd/compute/mghassem/001/gobi1/huggingface

# ── Experiment 1: Qwen3-VL-8B Screenshot Skill Prediction ───────────
echo "=== Qwen3-VL-8B: Screenshot → Skill (teacher forcing) ==="
python -m interaskill.eval_multimodal \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --mode vlm --max-tasks 200

echo ""
echo "=== Qwen3-VL-8B: Screenshot → Skill (composition) ==="
python -m interaskill.eval_multimodal \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --mode vlm --max-tasks 200 --composition

# ── Experiment 2: CLIP Zero-Shot Baseline ────────────────────────────
echo ""
echo "=== CLIP ViT-L/14: Zero-Shot Skill Classification ==="
python -m interaskill.eval_multimodal \
    --mode clip-zeroshot --max-tasks 200

# ── Experiment 3: Cross-Domain Visual Transfer Analysis ──────────────
echo ""
echo "=== Cross-Domain Visual Transfer (CLIP) ==="
python -m interaskill.eval_multimodal \
    --mode cross-domain --max-tasks 500

echo ""
echo "=== Done ==="
echo "Date: $(date)"
