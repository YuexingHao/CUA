#!/bin/bash
#SBATCH --job-name=e2e-large
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-e2e-large-%j.out

echo "=== E2E Composition: Large Models (31-70B) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# ── Llama-3.1-70B ────────────────────────────────────────────────────
echo "=== [1/3] Llama-3.1-70B — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --split test_task --max-tasks 200 --resume --composition

# ── Gemma-4-31B ──────────────────────────────────────────────────────
echo ""
echo "=== [2/3] Gemma-4-31B — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "google/gemma-4-31B-it" \
    --split test_task --max-tasks 200 --resume --composition

# ── DeepSeek-R1-Distill-32B ──────────────────────────────────────────
echo ""
echo "=== [3/3] DeepSeek-R1-Distill-32B — composition (test_task) ==="
python -m interaskill.eval_mind2web \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --split test_task --max-tasks 200 --resume --composition

echo ""
echo "=== Large Models Done ==="
echo "Date: $(date)"
