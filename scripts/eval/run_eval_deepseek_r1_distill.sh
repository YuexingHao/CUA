#!/bin/bash
#SBATCH --job-name=dsr1-32b
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-dsr1-distill-%j.out

echo "=== DeepSeek-R1-Distill-Qwen-32B: Skill Prediction ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Step 1: IW
echo "=== Step 1: Evaluating on IW ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset iw --max-convs 50 --max-new-tokens 200 --resume

# Step 2: WebArena
echo ""
echo "=== Step 2: Evaluating on WebArena ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset wa --max-convs 200 --max-new-tokens 200 --resume

# Step 3: BrowseComp-Plus
echo ""
echo "=== Step 3: Evaluating on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model "$MODEL" \
    --dataset bc --max-convs 200 --max-new-tokens 200 --resume

echo ""
echo "=== Done ==="
echo "Date: $(date)"
