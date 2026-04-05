#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=1:00:00
#SBATCH --output=results/slurm-baselines-%j.out

echo "=== Baseline Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

python -c "
import json
from pathlib import Path
from interaskill.data import load_and_featurize
from interaskill.baselines import run_all_baselines

# ── Fabricated (IW) Data ──────────────────────────────────────
print('=' * 70)
print('DATASET: Fabricated (IW Benchmark)')
print('=' * 70)
traj_iw = load_and_featurize('data/fabricated_trajectories.json')
print(f'Loaded {len(traj_iw)} trajectories')
results_iw = run_all_baselines(traj_iw)

with open('results/baseline_metrics_iw.json', 'w') as f:
    json.dump(results_iw, f, indent=2)
print(f'\nSaved to results/baseline_metrics_iw.json')

# ── WebArena Data ─────────────────────────────────────────────
print('\n\n')
print('=' * 70)
print('DATASET: WebArena (Go-Browse-WA)')
print('=' * 70)
try:
    traj_wa = load_and_featurize('data/webarena_trajectories.json')
    print(f'Loaded {len(traj_wa)} trajectories')
    results_wa = run_all_baselines(traj_wa)

    with open('results/baseline_metrics_wa.json', 'w') as f:
        json.dump(results_wa, f, indent=2)
    print(f'\nSaved to results/baseline_metrics_wa.json')
except Exception as e:
    print(f'WebArena data not available: {e}')

# ── Summary Table ─────────────────────────────────────────────
print('\n\n')
print('=' * 70)
print('SUMMARY: All Models on Fabricated Data')
print('=' * 70)
print(f'{\"Model\":<25s} {\"Exact Match\":>12s} {\"Edit Dist\":>10s} {\"Mean Pos Acc\":>12s}')
print('-' * 62)

# Baselines
for key in ['frequency', 'skillmd', 'awm']:
    r = results_iw[key]
    pa = r['per_position_accuracy']
    mean_pa = sum(v for k, v in pa.items() if int(k) > 0) / max(len(pa) - 1, 1)
    print(f'{r[\"name\"]:<25s} {r[\"exact_match\"]:>12.3f} {r[\"edit_distance\"]:>10.3f} {mean_pa:>12.3f}')

# Our models (from saved metrics)
try:
    with open('results/metrics.json') as f:
        our = json.load(f)

    # MLP
    mlp = our['phase3_mlp']
    mlp_pa = mlp['per_position_accuracy']
    mlp_mean = sum(v for k, v in mlp_pa.items() if int(k) > 0) / max(len(mlp_pa) - 1, 1)
    print(f'{\"MLP (ours)\":<25s} {mlp[\"sequence_exact_match\"]:>12.3f} {mlp[\"normalized_edit_distance\"]:>10.3f} {mlp_mean:>12.3f}')

    # Transformer
    tf = our['phase3_transformer']
    tf_pa = tf['per_position_accuracy']
    tf_mean = sum(v for k, v in tf_pa.items() if int(k) > 0) / max(len(tf_pa) - 1, 1)
    print(f'{\"Transformer (ours)\":<25s} {tf[\"sequence_exact_match\"]:>12.3f} {tf[\"normalized_edit_distance\"]:>10.3f} {tf_mean:>12.3f}')
except:
    pass

# Qwen3
try:
    with open('results/qwen3_eval_metrics.json') as f:
        qwen = json.load(f)
    print(f'{\"Qwen3-8B LoRA (ours)\":<25s} {\"—\":>12s} {\"—\":>10s} {qwen[\"overall_accuracy\"]:>12.3f}')
except:
    pass
"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
