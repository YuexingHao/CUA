"""InteraSkill: Full pipeline — segmentation, discovery, composition."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from .data import (
    load_and_featurize, extract_all_segments,
    SKILL_TYPES, SKILL_TO_IDX, N_SKILLS,
)
from .segment import sweep_theta
from .discover import (
    segments_to_summaries, compute_distance_matrix,
    cluster_segments, train_encoder, encode_all,
)
from .compose import (
    build_composition_dataset, build_sequence_dataset,
    train_macro_policy, train_transformer_policy,
    reconstruct_sequences, reconstruct_sequences_transformer,
)
from .evaluate import (
    cluster_purity,
    sequence_exact_match, normalized_edit_distance, per_position_accuracy,
    plot_discontinuity_histogram, plot_tsne, plot_confusion_matrix,
    plot_training_curves, plot_per_position_accuracy,
)

DATA_PATH = Path("data/fabricated_trajectories.json")
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ── Load Data ──────────────────────────────────────────────────
    print("\n[1/7] Loading and featurizing trajectories...")
    traj_data = load_and_featurize(str(DATA_PATH))
    segments, labels, label_ints = extract_all_segments(traj_data)
    print(f"  {len(traj_data)} trajectories, {len(segments)} segments, "
          f"{N_SKILLS} skill types")

    # ── Phase 1: Segmentation ─────────────────────────────────────
    print(f"\n[2/7] Phase 1: Trajectory Segmentation")
    best_theta, best_m, all_results, all_disc = sweep_theta(traj_data)
    print(f"  Best theta={best_theta:.4f}: "
          f"P={best_m['precision']:.3f} R={best_m['recall']:.3f} F1={best_m['f1']:.3f}")
    for k, v in all_results.items():
        print(f"    {k}: theta={v['theta']:.4f} P={v['precision']:.3f} "
              f"R={v['recall']:.3f} F1={v['f1']:.3f}")

    plot_discontinuity_histogram(all_disc, best_theta,
                                 str(FIGURES_DIR / "segmentation_hist.png"))

    # ── Phase 2a: Wasserstein Clustering ──────────────────────────
    print(f"\n[3/7] Phase 2a: Gaussian Representation + Wasserstein Clustering")
    summaries = segments_to_summaries(segments)
    dist_matrix = compute_distance_matrix(summaries)

    best_nmi, best_k = 0, 12
    for k in [8, 10, 12, 14, 16]:
        cl = cluster_segments(dist_matrix, n_clusters=k)
        nmi = normalized_mutual_info_score(label_ints.numpy(), cl)
        pur = cluster_purity(cl, label_ints.numpy())
        print(f"  k={k}: NMI={nmi:.3f}, Purity={pur:.3f}")
        if nmi > best_nmi:
            best_nmi, best_k = nmi, k

    best_cl = cluster_segments(dist_matrix, n_clusters=best_k)
    print(f"  Best k={best_k}: NMI={best_nmi:.3f}")
    plot_confusion_matrix(best_cl, label_ints.numpy(), SKILL_TYPES,
                          str(FIGURES_DIR / "clustering_confusion.png"))

    # ── Phase 2b: InfoNCE Encoder ─────────────────────────────────
    print(f"\n[4/7] Phase 2b: InfoNCE Contrastive Encoder Training")
    encoder, train_losses, val_losses = train_encoder(
        summaries, label_ints, device=device, epochs=200,
        lr=1e-3, batch_size=256, verbose=True,
    )
    plot_training_curves(train_losses, val_losses,
                         str(FIGURES_DIR / "training_curves.png"))

    z_all = encode_all(encoder, summaries, device=device)
    print(f"  Embeddings: {z_all.shape}")

    # Evaluate latent space
    km = KMeans(n_clusters=N_SKILLS, random_state=42, n_init=10).fit(z_all.numpy())
    latent_nmi = normalized_mutual_info_score(label_ints.numpy(), km.labels_)
    latent_sil = silhouette_score(z_all.numpy(), label_ints.numpy())
    latent_pur = cluster_purity(km.labels_, label_ints.numpy())
    print(f"  Latent KMeans: NMI={latent_nmi:.3f}, Silhouette={latent_sil:.3f}, "
          f"Purity={latent_pur:.3f}")

    plot_tsne(z_all.numpy(), label_ints.numpy(), SKILL_TYPES,
              str(FIGURES_DIR / "tsne.png"))

    # ── Phase 3a: MLP Composition (baseline) ──────────────────────
    print(f"\n[5/7] Phase 3a: MLP Skill Composition (baseline)")
    states, skill_targets, term_targets = build_composition_dataset(
        traj_data, encoder, summaries, label_ints, device=device,
    )
    print(f"  Composition dataset: {states.shape[0]} samples")

    policy_mlp, mlp_losses = train_macro_policy(
        states, skill_targets, term_targets,
        device=device, epochs=150, lr=1e-3, verbose=True,
    )

    pred_seqs_mlp = reconstruct_sequences(
        policy_mlp, encoder, traj_data, summaries, device=device,
    )
    gt_seqs = [td.skill_sequence for td in traj_data]

    mlp_exact = sequence_exact_match(pred_seqs_mlp, gt_seqs)
    mlp_edit = normalized_edit_distance(pred_seqs_mlp, gt_seqs)
    mlp_pos = per_position_accuracy(pred_seqs_mlp, gt_seqs)
    print(f"  [MLP] Exact match: {mlp_exact:.3f}")
    print(f"  [MLP] Edit distance: {mlp_edit:.3f}")
    print(f"  [MLP] Per-position accuracy: {mlp_pos}")

    # ── Phase 3b: Transformer Composition ─────────────────────────
    print(f"\n[6/7] Phase 3b: Transformer Skill Composition")
    seq_dataset = build_sequence_dataset(
        traj_data, encoder, summaries, label_ints, device=device,
    )
    print(f"  Sequence dataset: {len(seq_dataset)} sequences")

    policy_tf, tf_train_losses, tf_val_losses = train_transformer_policy(
        seq_dataset, device=device, epochs=500, lr=5e-4,
        batch_size=64, verbose=True,
    )

    pred_seqs_tf = reconstruct_sequences_transformer(
        policy_tf, encoder, traj_data, summaries, device=device,
    )

    tf_exact = sequence_exact_match(pred_seqs_tf, gt_seqs)
    tf_edit = normalized_edit_distance(pred_seqs_tf, gt_seqs)
    tf_pos = per_position_accuracy(pred_seqs_tf, gt_seqs)
    print(f"  [Transformer] Exact match: {tf_exact:.3f}")
    print(f"  [Transformer] Edit distance: {tf_edit:.3f}")
    print(f"  [Transformer] Per-position accuracy: {tf_pos}")

    plot_per_position_accuracy(tf_pos,
                               str(FIGURES_DIR / "position_accuracy.png"))

    # ── Phase 3 Comparison Plot ───────────────────────────────────
    print(f"\n[7/7] Generating comparison plots")
    plot_composition_comparison(mlp_pos, tf_pos,
                                str(FIGURES_DIR / "composition_comparison.png"))
    plot_training_curves(tf_train_losses, tf_val_losses,
                         str(FIGURES_DIR / "transformer_training.png"))

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    results = {
        "phase1_segmentation": {
            "best_theta": float(best_theta),
            "boundary_precision": best_m["precision"],
            "boundary_recall": best_m["recall"],
            "boundary_f1": best_m["f1"],
        },
        "phase2_clustering": {
            "wasserstein_best_k": best_k,
            "wasserstein_nmi": float(best_nmi),
        },
        "phase2_infonce": {
            "latent_nmi": float(latent_nmi),
            "latent_silhouette": float(latent_sil),
            "latent_purity": float(latent_pur),
        },
        "phase3_mlp": {
            "sequence_exact_match": float(mlp_exact),
            "normalized_edit_distance": float(mlp_edit),
            "per_position_accuracy": {str(k): v for k, v in mlp_pos.items()},
        },
        "phase3_transformer": {
            "sequence_exact_match": float(tf_exact),
            "normalized_edit_distance": float(tf_edit),
            "per_position_accuracy": {str(k): v for k, v in tf_pos.items()},
        },
    }

    for phase, metrics in results.items():
        print(f"\n  {phase}:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save results JSON
    with open(METRICS_DIR / "pipeline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved metrics to {METRICS_DIR / 'pipeline_metrics.json'}")
    print(f"  Plots saved to {FIGURES_DIR}/")

    # Save models
    torch.save(encoder.state_dict(), MODELS_DIR / "encoder.pt")
    torch.save(policy_mlp.state_dict(), MODELS_DIR / "policy_mlp.pt")
    torch.save(policy_tf.state_dict(), MODELS_DIR / "policy_transformer.pt")
    print(f"  Models saved to {MODELS_DIR}/")


def plot_composition_comparison(mlp_pos, tf_pos, save_path):
    """Side-by-side bar chart comparing MLP vs Transformer per-position accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = sorted(set(mlp_pos.keys()) | set(tf_pos.keys()))
    mlp_accs = [mlp_pos.get(p, 0) for p in positions]
    tf_accs = [tf_pos.get(p, 0) for p in positions]

    x = np.arange(len(positions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, mlp_accs, width, label="MLP (baseline)",
                   color="#94a3b8", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, tf_accs, width, label="Transformer",
                   color="#2563eb", alpha=0.85, edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_title("Phase 3: MLP vs Transformer Per-Position Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pos {p}" for p in positions])
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Value labels
    for bar, acc in zip(bars1, mlp_accs):
        if acc > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=10, color="#64748b")
    for bar, acc in zip(bars2, tf_accs):
        if acc > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=10,
                    color="#1e40af", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
