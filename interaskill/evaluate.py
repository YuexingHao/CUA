"""Evaluation metrics and plotting for InteraSkill pipeline."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)
from collections import Counter

# ── Global Plot Style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── Phase 1 Metrics ───────────────────────────────────────────────────

def cluster_purity(pred_labels, gt_labels) -> float:
    """Fraction of samples whose cluster majority class matches ground truth."""
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    total = 0
    for c in np.unique(pred):
        mask = pred == c
        most_common = Counter(gt[mask]).most_common(1)[0][1]
        total += most_common
    return total / len(gt)


# ── Phase 3 Metrics ───────────────────────────────────────────────────

def sequence_exact_match(pred_seqs, gt_seqs) -> float:
    """Fraction of trajectories with exact skill sequence match."""
    matches = sum(1 for p, g in zip(pred_seqs, gt_seqs) if p == g)
    return matches / len(gt_seqs) if gt_seqs else 0.0


def normalized_edit_distance(pred_seqs, gt_seqs) -> float:
    """Average normalized Levenshtein distance."""
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (0 if c1 == c2 else 1)))
            prev = curr
        return prev[-1]

    dists = []
    for p, g in zip(pred_seqs, gt_seqs):
        max_len = max(len(p), len(g), 1)
        dists.append(levenshtein(p, g) / max_len)
    return np.mean(dists) if dists else 0.0


def per_position_accuracy(pred_seqs, gt_seqs) -> dict:
    """Accuracy at each position across all trajectories."""
    pos_correct = {}
    pos_total = {}
    for p, g in zip(pred_seqs, gt_seqs):
        for i, (pi, gi) in enumerate(zip(p, g)):
            pos_total[i] = pos_total.get(i, 0) + 1
            pos_correct[i] = pos_correct.get(i, 0) + (1 if pi == gi else 0)
    return {i: pos_correct.get(i, 0) / pos_total[i]
            for i in sorted(pos_total.keys())}


# ── Plots ─────────────────────────────────────────────────────────────

def plot_discontinuity_histogram(all_disc, theta, save_path):
    """Plot histogram of action discontinuities with threshold line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_disc.numpy(), bins=80, alpha=0.75, color="#2563eb",
            edgecolor="white", linewidth=0.5)
    ax.axvline(theta, color="#dc2626", linestyle="--", linewidth=2.5,
               label=f"θ = {theta:.3f}")
    ax.set_xlabel("Action Discontinuity (L₂ norm)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 1: Action Discontinuity Distribution")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(embeddings, labels, label_names, save_path):
    """t-SNE visualization of skill embeddings."""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        name = label_names[int(lbl)] if isinstance(lbl, (int, np.integer)) else lbl
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=[colors[i]], label=name,
                   s=40, alpha=0.75, edgecolors="white", linewidths=0.3)

    ax.legend(fontsize=12, markerscale=1.8, loc="best", ncol=2,
              frameon=True, fancybox=True, shadow=True)
    ax.set_title("Phase 2: t-SNE of Learned Skill Embeddings (16-dim → 2-dim)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(pred_labels, gt_labels, label_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(gt_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=13)
    ax.set_yticklabels(label_names, fontsize=13)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("Ground Truth Skill")
    ax.set_title("Phase 2: Clustering Confusion Matrix (Normalized)")

    # Add value annotations
    for i in range(len(label_names)):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_losses, alpha=0.4, color="#2563eb", linewidth=1.5,
            label="Train (per step)")
    if val_losses:
        val_x = np.linspace(0, len(train_losses), len(val_losses))
        ax.plot(val_x, val_losses, color="#dc2626", linewidth=2.5,
                label="Validation")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Phase 2: InfoNCE Encoder Training")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_position_accuracy(pos_acc, save_path):
    """Bar chart of per-position skill prediction accuracy."""
    positions = sorted(pos_acc.keys())
    accs = [pos_acc[p] for p in positions]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar([f"Pos {p}" for p in positions], accs,
                  color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase 3: Per-Position Skill Prediction Accuracy")
    ax.axhline(y=np.mean(accs), color="#dc2626", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(accs):.3f}")

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
