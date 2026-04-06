#!/bin/bash
#SBATCH --job-name=webshop-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-webshop-%j.out

echo "=== WebShop End-to-End Evaluation (Full Suite) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Setup WebShop if not present
if [ ! -d third_party/WebShop ]; then
    echo "=== Setting up WebShop ==="
    python data/setup_webshop.py
    echo ""
fi

MODEL="Qwen/Qwen3-8B"
ADAPTER="results/qwen3_lora/final_adapter"

# ── Primary evaluation ───────────────────────────────────────────────

echo "=== Qwen3-8B LoRA: skill+LLM (full pipeline) ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" --adapter "$ADAPTER" \
    --max-tasks 200 --resume

echo ""
echo "=== Qwen3-8B Zero-Shot: skill+LLM ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" \
    --max-tasks 200 --resume

# ── Ablation study (#5): does the skill layer add value? ─────────────

echo ""
echo "=== ABLATION: direct action (no skill stage) ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" --adapter "$ADAPTER" \
    --max-tasks 200 --resume \
    --grounding-mode direct

echo ""
echo "=== ABLATION: heuristic skill + LLM action ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" --adapter "$ADAPTER" \
    --max-tasks 200 --resume \
    --skill-mode heuristic

echo ""
echo "=== ABLATION: LLM skill + heuristic action ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" --adapter "$ADAPTER" \
    --max-tasks 200 --resume \
    --grounding-mode heuristic

echo ""
echo "=== ABLATION: oracle skill + LLM action (upper bound) ==="
python -m interaskill.eval_webshop \
    --model "$MODEL" --adapter "$ADAPTER" \
    --max-tasks 200 --resume \
    --skill-mode oracle

echo ""
echo "=== Done ==="
echo "Date: $(date)"
