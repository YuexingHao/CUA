#!/bin/bash
#SBATCH --job-name=m2w-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-mind2web-%j.out

echo "=== Mind2Web Evaluation (Full Suite) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Download Mind2Web data if not cached
if [ ! -f data/mind2web_test_task.json ]; then
    echo "=== Downloading Mind2Web dataset ==="
    python data/download_mind2web.py --max-tasks 500
    echo ""
fi

MODEL="Qwen/Qwen3-8B"
ADAPTER="results/qwen3_lora/final_adapter"

# ── Cross-domain generalization (#3): all 3 Mind2Web splits ──────────

for SPLIT in test_task test_website test_domain; do
    echo ""
    echo "=== Qwen3-8B LoRA on Mind2Web ($SPLIT) — teacher forcing ==="
    python -m interaskill.eval_mind2web \
        --model "$MODEL" --adapter "$ADAPTER" \
        --split $SPLIT --max-tasks 200 --resume

    echo ""
    echo "=== Qwen3-8B LoRA on Mind2Web ($SPLIT) — composition ==="
    python -m interaskill.eval_mind2web \
        --model "$MODEL" --adapter "$ADAPTER" \
        --split $SPLIT --max-tasks 200 --resume --composition
done

# ── Ablation study (#5): skill value ─────────────────────────────────

echo ""
echo "=== ABLATION: direct action (no skill stage) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --grounding-mode direct

echo ""
echo "=== ABLATION: heuristic skill + LLM action ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --skill-mode heuristic

echo ""
echo "=== ABLATION: LLM skill + heuristic action ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --grounding-mode heuristic

echo ""
echo "=== ABLATION: oracle skill + LLM action (upper bound) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" --adapter "$ADAPTER" \
    --split test_task --max-tasks 200 --resume \
    --skill-mode oracle

# ── Zero-shot baseline ───────────────────────────────────────────────

echo ""
echo "=== Qwen3-8B Zero-Shot on Mind2Web (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "$MODEL" \
    --split test_task --max-tasks 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"
