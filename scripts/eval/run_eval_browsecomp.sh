#!/bin/bash
#SBATCH --job-name=bc-eval-all
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-browsecomp-%j.out

echo "=== BrowseComp-Plus Evaluation: All Models ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Step 0: Generate BC conversations if not already done
if [ ! -f data/bc_conversations.jsonl ]; then
    echo "=== Generating BrowseComp-Plus conversations ==="
    python data/generate_bc_conversations.py --max-queries 200 --seed 42
    echo ""
fi

# Step 1: Qwen3-8B LoRA (fine-tuned)
echo "=== Qwen3-8B LoRA on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model Qwen/Qwen3-8B \
    --adapter results/qwen3_lora/final_adapter \
    --dataset bc --max-convs 200

# Step 2: Llama-3.1-70B (zero-shot)
echo ""
echo "=== Llama-3.1-70B on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset bc --max-convs 200

# Step 3: Gemma-4-31B (zero-shot)
echo ""
echo "=== Gemma-4-31B on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model google/gemma-4-31B-it \
    --dataset bc --max-convs 200

# Step 4: OLMo-3-7B (zero-shot)
echo ""
echo "=== OLMo-3-7B on BrowseComp-Plus ==="
python -m interaskill.eval_model \
    --model allenai/Olmo-3-1025-7B \
    --dataset bc --max-convs 200

echo ""
echo "=== Done ==="
echo "Date: $(date)"

echo ""
echo "=== Results Summary ==="
for f in results/*_eval_metrics_bc.json; do
    echo "--- $(basename $f) ---"
    cat $f 2>/dev/null
    echo ""
done
