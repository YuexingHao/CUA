#!/bin/bash
#SBATCH --job-name=qwen3-wa-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=6:00:00
#SBATCH --output=results/slurm_logs/slurm-eval-wa-%j.out

echo "=== Qwen3-8B LoRA: WebArena + IW Edit Distance Evaluation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Step 1: Generate WebArena conversations (fast, no GPU needed)
echo "=== Step 1: Generating WebArena conversations ==="
python data/generate_wa_conversations.py --max-trajs 200 --seed 42

# Step 2: Evaluate on WebArena conversations
echo ""
echo "=== Step 2: Evaluating on WebArena ==="
python -m interaskill.eval_qwen --dataset wa --max-convs 200

# Step 3: Re-evaluate on IW with edit distance (reuses existing model load)
echo ""
echo "=== Step 3: Evaluating on IW (with edit distance) ==="
python -m interaskill.eval_qwen --dataset iw --max-convs 50

echo ""
echo "=== Done ==="
echo "Date: $(date)"

echo ""
echo "=== Results Summary ==="
echo "WebArena:"
cat results/qwen3_eval_metrics_wa.json 2>/dev/null
echo ""
echo "IW:"
cat results/qwen3_eval_metrics.json 2>/dev/null
