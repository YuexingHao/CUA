#!/bin/bash
#SBATCH --job-name=phi4-eval
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --time=6:00:00
#SBATCH --output=results/slurm_logs/slurm-phi4mini-%j.out

echo "=== Phi-4-mini-instruct: Skill Prediction Evaluation ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

MODEL="microsoft/Phi-4-mini-instruct"

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
cat results/phi-4-mini_eval_metrics.json 2>/dev/null
echo ""
echo "WebArena:"
cat results/phi-4-mini_eval_metrics_wa.json 2>/dev/null
