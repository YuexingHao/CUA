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
ADAPTER="results/qwen3_lora/final_adapter"

# ══════════════════════════════════════════════════════════════════════
# KEY EXPERIMENT: Next-Skill Prediction vs Full Task Completion
# ══════════════════════════════════════════════════════════════════════

# ── 1. Teacher forcing (baseline, like current eval) ─────────────────
# Model sees ground-truth action history at each step.
# Tests: can the model predict the right skill at each step?
echo "=== [1/6] Qwen3-8B LoRA — teacher forcing (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume

# ── 2. Composition mode (the real test) ──────────────────────────────
# Model sees its OWN previous predictions in the history.
# Tests: can errors compound? Can the agent recover?
echo ""
echo "=== [2/6] Qwen3-8B LoRA — COMPOSITION (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume --composition

# ── 3. Zero-shot composition (no fine-tuning) ────────────────────────
echo ""
echo "=== [3/6] Qwen3-8B zero-shot — COMPOSITION (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" \
    --split test_task --max-tasks 200 --resume --composition

# ── 4. Cross-domain generalization (unseen domains) ──────────────────
echo ""
echo "=== [4/6] Qwen3-8B LoRA — composition (test_domain) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_domain --max-tasks 200 --resume --composition

# ══════════════════════════════════════════════════════════════════════
# ABLATION: Does the skill layer help task completion?
# ══════════════════════════════════════════════════════════════════════

# ── 5. Direct action (no skill stage) ────────────────────────────────
echo ""
echo "=== [5/6] ABLATION: direct action, no skill stage (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --composition --grounding-mode direct

# ── 6. Oracle skills (upper bound) ───────────────────────────────────
echo ""
echo "=== [6/6] ABLATION: oracle skills (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --composition --skill-mode oracle

echo ""
echo "=== All Done ==="
echo "Date: $(date)"
