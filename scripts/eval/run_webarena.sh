#!/bin/bash
#SBATCH --job-name=webarena
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50gb
#SBATCH --time=4:00:00
#SBATCH --output=results/slurm-webarena-%j.out

echo "=== WebArena Pipeline ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Step 1: Download WebArena trajectories
echo "=== Step 1: Downloading WebArena trajectories ==="
python data/download_webarena.py --max-trajs 1000 --success-only

# Step 2: Run InteraSkill pipeline on WebArena data
echo ""
echo "=== Step 2: Running InteraSkill pipeline ==="
# Temporarily point pipeline to WebArena data
python -c "
import json
from pathlib import Path
from interaskill.data import load_and_featurize, extract_all_segments, SKILL_TYPES, N_SKILLS
from interaskill.segment import sweep_theta
from interaskill.discover import segments_to_summaries, compute_distance_matrix, cluster_segments, train_encoder, encode_all
from interaskill.compose import build_sequence_dataset, train_transformer_policy, reconstruct_sequences_transformer
from interaskill.evaluate import (cluster_purity, sequence_exact_match, normalized_edit_distance, per_position_accuracy,
    plot_discontinuity_histogram, plot_tsne, plot_confusion_matrix, plot_training_curves, plot_per_position_accuracy)
import torch, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

DATA_PATH = 'data/webarena_trajectories.json'
RESULTS_DIR = Path('results/webarena')
RESULTS_DIR.mkdir(exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load
print('\n[1/5] Loading WebArena trajectories...')
traj_data = load_and_featurize(DATA_PATH)
segments, labels, label_ints = extract_all_segments(traj_data)
print(f'  {len(traj_data)} trajectories, {len(segments)} segments, {N_SKILLS} skill types')

# Segmentation
print('\n[2/5] Trajectory Segmentation')
best_theta, best_m, all_results, all_disc = sweep_theta(traj_data)
print(f'  Best theta={best_theta:.4f}: P={best_m[\"precision\"]:.3f} R={best_m[\"recall\"]:.3f} F1={best_m[\"f1\"]:.3f}')
plot_discontinuity_histogram(all_disc, best_theta, str(RESULTS_DIR / 'segmentation_hist.png'))

# Clustering + Encoder
print('\n[3/5] Skill Discovery')
summaries = segments_to_summaries(segments)
dist_matrix = compute_distance_matrix(summaries)
best_nmi, best_k = 0, 12
for k in [8, 10, 12, 14, 16]:
    cl = cluster_segments(dist_matrix, n_clusters=k)
    nmi = normalized_mutual_info_score(label_ints.numpy(), cl)
    pur = cluster_purity(cl, label_ints.numpy())
    print(f'  k={k}: NMI={nmi:.3f}, Purity={pur:.3f}')
    if nmi > best_nmi: best_nmi, best_k = nmi, k
plot_confusion_matrix(cluster_segments(dist_matrix, n_clusters=best_k), label_ints.numpy(), SKILL_TYPES, str(RESULTS_DIR / 'clustering_confusion.png'))

print('\n[4/5] InfoNCE Encoder')
encoder, tl, vl = train_encoder(summaries, label_ints, device=device, epochs=200, lr=1e-3, batch_size=256, verbose=True)
plot_training_curves(tl, vl, str(RESULTS_DIR / 'training_curves.png'))
z_all = encode_all(encoder, summaries, device=device)
km = KMeans(n_clusters=N_SKILLS, random_state=42, n_init=10).fit(z_all.numpy())
latent_nmi = normalized_mutual_info_score(label_ints.numpy(), km.labels_)
latent_sil = silhouette_score(z_all.numpy(), label_ints.numpy())
latent_pur = cluster_purity(km.labels_, label_ints.numpy())
print(f'  Latent: NMI={latent_nmi:.3f}, Sil={latent_sil:.3f}, Purity={latent_pur:.3f}')
plot_tsne(z_all.numpy(), label_ints.numpy(), SKILL_TYPES, str(RESULTS_DIR / 'tsne.png'))

# Composition
print('\n[5/5] Transformer Composition')
seq_ds = build_sequence_dataset(traj_data, encoder, summaries, label_ints, device=device)
print(f'  {len(seq_ds)} sequences')
policy, ptl, pvl = train_transformer_policy(seq_ds, device=device, epochs=500, lr=5e-4, batch_size=64, verbose=True)
pred_seqs = reconstruct_sequences_transformer(policy, encoder, traj_data, summaries, device=device)
gt_seqs = [td.skill_sequence for td in traj_data]
em = sequence_exact_match(pred_seqs, gt_seqs)
ed = normalized_edit_distance(pred_seqs, gt_seqs)
pa = per_position_accuracy(pred_seqs, gt_seqs)
print(f'  Exact match: {em:.3f}, Edit dist: {ed:.3f}')
print(f'  Per-pos accuracy: {pa}')
plot_per_position_accuracy(pa, str(RESULTS_DIR / 'position_accuracy.png'))

# Save
results = {
    'data_source': 'webarena (go-browse-wa)',
    'num_trajectories': len(traj_data),
    'num_segments': len(segments),
    'phase1': {'theta': float(best_theta), 'f1': best_m['f1']},
    'phase2_wasserstein': {'best_k': best_k, 'nmi': float(best_nmi)},
    'phase2_infonce': {'nmi': float(latent_nmi), 'silhouette': float(latent_sil), 'purity': float(latent_pur)},
    'phase3_transformer': {'exact_match': float(em), 'edit_distance': float(ed), 'per_position': {str(k):v for k,v in pa.items()}},
}
with open(RESULTS_DIR / 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {RESULTS_DIR}/')
torch.save(encoder.state_dict(), RESULTS_DIR / 'encoder.pt')
torch.save(policy.state_dict(), RESULTS_DIR / 'policy.pt')
"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
