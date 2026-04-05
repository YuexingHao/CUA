#!/bin/bash
#SBATCH --job-name=gemma31b-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=8:00:00
#SBATCH --output=results/slurm-gemma31b-%j.out

echo "=== Gemma-4-31B-IT: Skill Prediction Evaluation ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

MODEL="google/gemma-4-31B-it"

# Step 1: Evaluate on IW data
echo "=== Step 1: Evaluating on IW ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset iw --max-convs 50

# Step 2: Evaluate on WebArena data
echo ""
echo "=== Step 2: Evaluating on WebArena ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset wa --max-convs 200

echo ""
echo "=== Done ==="
echo "Date: $(date)"

echo ""
echo "=== Results Summary ==="
echo "IW:"
cat results/gemma4-31b_eval_metrics.json 2>/dev/null
echo ""
echo "WebArena:"
cat results/gemma4-31b_eval_metrics_wa.json 2>/dev/null
