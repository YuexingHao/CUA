#!/bin/bash
#SBATCH --job-name=interaskill
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=7-00:00:00
#SBATCH --output=results/slurm-%j.out

echo "=== InteraSkill Pipeline ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026

# Activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda activate base

python -m interaskill.main

echo ""
echo "=== Done ==="
echo "Date: $(date)"
