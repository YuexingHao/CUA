#!/bin/bash
#SBATCH --job-name=e2e-small
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-small-%j.out

echo "=== E2E Composition: Small Models (7-8B) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# ── OLMo-3-7B ────────────────────────────────────────────────────────
echo "=== [1/5] OLMo-3-7B — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "allenai/Olmo-3-1025-7B" \
    --split test_task --max-tasks 200 --resume --composition

# ── Qwen3-8B zero-shot ───────────────────────────────────────────────
echo ""
echo "=== [2/5] Qwen3-8B zero-shot — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" \
    --split test_task --max-tasks 200 --resume --composition

# ── Qwen3-8B LoRA ────────────────────────────────────────────────────
echo ""
echo "=== [3/5] Qwen3-8B LoRA — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" --adapter "results/qwen3_lora/final_adapter" \
    --split test_task --max-tasks 200 --resume --composition

# ── Qwen3-8B GRPO ────────────────────────────────────────────────────
echo ""
echo "=== [4/5] Qwen3-8B GRPO — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" --adapter "results/qwen3_grpo/final_adapter" \
    --split test_task --max-tasks 200 --resume --composition

# ── Qwen3-8B LoRA cross-domain ───────────────────────────────────────
echo ""
echo "=== [5/5] Qwen3-8B LoRA — composition (test_domain) ==="
python -m interaskill.eval_mind2web \
    --model "Qwen/Qwen3-8B" --adapter "results/qwen3_lora/final_adapter" \
    --split test_domain --max-tasks 200 --resume --composition

echo ""
echo "=== Small Models Done ==="
echo "Date: $(date)"
