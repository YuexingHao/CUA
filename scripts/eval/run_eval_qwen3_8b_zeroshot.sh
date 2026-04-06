#!/bin/bash
#SBATCH --job-name=qwen3-zs
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/slurm-qwen3-zs-%j.out

echo "=== Qwen3-8B Zero-Shot: Skill Prediction Evaluation ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

MODEL="Qwen/Qwen3-8B"

# Step 1: Evaluate on IW data
echo "=== Step 1: Evaluating on IW ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset iw --max-convs 50 --resume

# Step 2: Evaluate on WebArena data
echo ""
echo "=== Step 2: Evaluating on WebArena ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset wa --max-convs 200 --resume

# Step 3: Evaluate on BrowseComp-Plus data
echo ""
echo "=== Step 3: Evaluating on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset bc --max-convs 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"

echo ""
echo "=== Results Summary ==="
echo "IW:"
cat results/qwen3-8b_eval_metrics.json 2>/dev/null
echo ""
echo "WebArena:"
cat results/qwen3-8b_eval_metrics_wa.json 2>/dev/null
echo ""
echo "BrowseComp-Plus:"
cat results/qwen3-8b_eval_metrics_bc.json 2>/dev/null
