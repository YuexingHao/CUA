#!/bin/bash
#SBATCH --job-name=online-wa
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-online-wa-%j.out

echo "=== InteraSkill Online Agent: WebArena ==="
echo "Node: $(hostname) | Date: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# ── WebArena server config ──────────────────────────────────────────
# UPDATE THESE to point to your WebArena Docker services
# export WA_SHOPPING="http://your-server:8082/"
# export WA_REDDIT="http://your-server:8080"
# export WA_GITLAB="http://your-server:9001"
# export WA_MAP="http://your-server:443"

MODEL="Qwen/Qwen3-8B"
ADAPTER="results/qwen3_lora/final_adapter"

# ── 1. LoRA agent on WebArena ───────────────────────────────────────
echo "=== [1/3] Qwen3-8B LoRA — online WebArena ==="
python -m interaskill.agent_online \
    --model "$MODEL" --adapter "$ADAPTER" \
    --benchmark webarena --max-tasks 50 --max-steps 15

# ── 2. Zero-shot agent ──────────────────────────────────────────────
echo ""
echo "=== [2/3] Qwen3-8B zero-shot — online WebArena ==="
python -m interaskill.agent_online \
    --model "$MODEL" \
    --benchmark webarena --max-tasks 50 --max-steps 15

# ── 3. Direct action (no skill, ablation) ───────────────────────────
echo ""
echo "=== [3/3] Ablation: direct action (no skill stage) ==="
python -m interaskill.agent_online \
    --model "$MODEL" --adapter "$ADAPTER" \
    --benchmark webarena --max-tasks 50 --max-steps 15 \
    --grounding-mode direct

echo ""
echo "=== Done ==="
echo "Date: $(date)"
