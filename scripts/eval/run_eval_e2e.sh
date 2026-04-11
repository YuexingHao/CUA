#!/bin/bash
#SBATCH --job-name=e2e-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-%j.out

echo "=== End-to-End Task Completion Evaluation ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# ── Step 0: Download Mind2Web data if needed ─────────────────────────
if [ ! -f data/mind2web_test_task.json ]; then
    echo "=== Downloading Mind2Web dataset ==="
    python data/download_mind2web.py --max-tasks 500
    echo ""
fi

MODEL="Qwen/Qwen3-8B"
LORA_ADAPTER="results/qwen3_lora/final_adapter"
GRPO_ADAPTER="results/qwen3_grpo/final_adapter"

# ══════════════════════════════════════════════════════════════════════
# FULL TRAJECTORY EVALUATION (Composition Mode)
# Model uses its OWN predicted actions in history (autoregressive)
# ══════════════════════════════════════════════════════════════════════

# ── 1. LoRA — composition (the main result) ──────────────────────────
echo "=== [1/7] Qwen3-8B LoRA — COMPOSITION (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --split test_task --max-tasks 200 --resume --composition

# ── 2. GRPO — composition ────────────────────────────────────────────
echo ""
echo "=== [2/7] Qwen3-8B GRPO — COMPOSITION (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$GRPO_ADAPTER" \
    --split test_task --max-tasks 200 --resume --composition

# ── 3. Zero-shot — composition ───────────────────────────────────────
echo ""
echo "=== [3/7] Qwen3-8B zero-shot — COMPOSITION (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" \
    --split test_task --max-tasks 200 --resume --composition

# ── 4. Cross-domain — LoRA composition ───────────────────────────────
echo ""
echo "=== [4/7] Qwen3-8B LoRA — composition (test_domain) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --split test_domain --max-tasks 200 --resume --composition

# ══════════════════════════════════════════════════════════════════════
# ABLATION: Does the skill layer help full task completion?
# ══════════════════════════════════════════════════════════════════════

# ── 5. Direct action (no skill stage) ────────────────────────────────
echo ""
echo "=== [5/7] ABLATION: direct action, no skill stage (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --composition --grounding-mode direct

# ── 6. Oracle skills (upper bound) ───────────────────────────────────
echo ""
echo "=== [6/7] ABLATION: oracle skills (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --composition --skill-mode oracle

# ── 7. Teacher forcing baseline (for comparison) ─────────────────────
echo ""
echo "=== [7/7] Qwen3-8B LoRA — teacher forcing baseline (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --split test_task --max-tasks 200 --resume

echo ""
echo "=== All Done ==="
echo "Date: $(date)"
