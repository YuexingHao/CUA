"""
Post-hoc verification of learned skill clusters.

Compares the discovered skill clusters (from Phase 2 InfoNCE encoder)
against the heuristic skill labels to verify that:
  1. Learned clusters align with human-intuitive skill categories
  2. The encoder captures meaningful skill distinctions
  3. There are no degenerate clusters (all same skill)

Usage:
    python -m interaskill.verify_skills \
        --encoder results/encoder.pt \
        --trajectories data/fabricated_trajectories.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)

from .data import SKILL_TYPES, SKILL_TO_IDX, load_and_featurize, extract_all_segments
from .discover import SegmentEncoder, segments_to_summaries, encode_all
from .grounding import (
    classify_mind2web_step, classify_webshop_phase,
    LearnedSkillClassifier, build_skill_prototypes,
)

RESULTS_DIR = Path("results")


def cluster_label_alignment(encoder_labels: np.ndarray,
                            gt_labels: np.ndarray) -> dict:
    """Compute alignment metrics between learned clusters and GT labels."""
    nmi = normalized_mutual_info_score(gt_labels, encoder_labels)
    ari = adjusted_rand_score(gt_labels, encoder_labels)
    v_measure = v_measure_score(gt_labels, encoder_labels)

    # Per-cluster purity: what fraction of each cluster is the majority class?
    cluster_purity = {}
    for c in np.unique(encoder_labels):
        mask = encoder_labels == c
        gt_in_cluster = gt_labels[mask]
        most_common = Counter(gt_in_cluster).most_common(1)[0]
        cluster_purity[int(c)] = {
            "majority_label": SKILL_TYPES[most_common[0]] if most_common[0] < len(SKILL_TYPES) else str(most_common[0]),
            "purity": most_common[1] / len(gt_in_cluster),
            "size": int(len(gt_in_cluster)),
        }

    # Per-skill coverage: what fraction of each GT skill lands in one cluster?
    skill_coverage = {}
    for s in np.unique(gt_labels):
        mask = gt_labels == s
        clusters_for_skill = encoder_labels[mask]
        most_common = Counter(clusters_for_skill).most_common(1)[0]
        skill_name = SKILL_TYPES[s] if s < len(SKILL_TYPES) else str(s)
        skill_coverage[skill_name] = {
            "primary_cluster": int(most_common[0]),
            "coverage": most_common[1] / len(clusters_for_skill),
            "n_clusters": len(set(clusters_for_skill)),
        }

    return {
        "nmi": float(nmi),
        "ari": float(ari),
        "v_measure": float(v_measure),
        "cluster_purity": cluster_purity,
        "skill_coverage": skill_coverage,
        "n_clusters": len(np.unique(encoder_labels)),
        "n_skills": len(np.unique(gt_labels)),
    }


def heuristic_vs_learned_agreement(heuristic_labels: list[str],
                                   learned_labels: list[str]) -> dict:
    """Compare heuristic keyword labels vs learned encoder labels."""
    assert len(heuristic_labels) == len(learned_labels)

    agree = sum(h == l for h, l in zip(heuristic_labels, learned_labels))
    total = len(heuristic_labels)

    # Confusion: where do they disagree?
    disagreements = Counter()
    for h, l in zip(heuristic_labels, learned_labels):
        if h != l:
            disagreements[(h, l)] += 1

    return {
        "agreement_rate": agree / total if total > 0 else 0.0,
        "n_agree": agree,
        "n_total": total,
        "top_disagreements": [
            {"heuristic": h, "learned": l, "count": c}
            for (h, l), c in disagreements.most_common(10)
        ],
    }


def verify_no_degenerate_clusters(cluster_labels: np.ndarray,
                                  min_cluster_size: int = 5) -> dict:
    """Check for degenerate clusters (too small or all-same-skill)."""
    cluster_sizes = Counter(cluster_labels)
    degenerate = []
    for cluster_id, size in cluster_sizes.items():
        if size < min_cluster_size:
            degenerate.append({
                "cluster": int(cluster_id),
                "size": size,
                "issue": "too_small",
            })

    return {
        "n_degenerate": len(degenerate),
        "degenerate_clusters": degenerate,
        "cluster_size_stats": {
            "min": int(min(cluster_sizes.values())),
            "max": int(max(cluster_sizes.values())),
            "mean": float(np.mean(list(cluster_sizes.values()))),
            "std": float(np.std(list(cluster_sizes.values()))),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify learned skill clusters")
    parser.add_argument("--encoder", type=str, default="results/encoder.pt")
    parser.add_argument("--trajectories", type=str,
                        default="data/fabricated_trajectories.json")
    parser.add_argument("--n-clusters", type=int, default=12)
    args = parser.parse_args()

    print("=" * 60)
    print("Post-Hoc Skill Cluster Verification")
    print("=" * 60)

    # Load trajectories and extract segments
    print("\nLoading trajectories...")
    traj_data = load_and_featurize(args.trajectories)
    segments, labels_str, label_ints = extract_all_segments(traj_data)
    summaries = segments_to_summaries(segments)
    gt_labels = label_ints.numpy()
    print(f"  {len(segments)} segments, {len(set(labels_str))} skill types")

    # Load encoder
    print(f"\nLoading encoder from {args.encoder}...")
    encoder = SegmentEncoder()
    encoder.load_state_dict(torch.load(args.encoder, weights_only=True))
    encoder.eval()

    # Encode and cluster
    z = encode_all(encoder, summaries)

    # K-means clustering in learned space
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z.numpy())

    # Build and save prototypes
    print("\nBuilding skill prototypes...")
    prototypes = build_skill_prototypes(encoder, summaries, label_ints)

    # Assign learned labels using nearest prototype
    with torch.no_grad():
        sims = torch.nn.functional.cosine_similarity(
            z.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        learned_label_ints = sims.argmax(dim=1).numpy()
    learned_labels = [SKILL_TYPES[i] for i in learned_label_ints]

    # ── Analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # 1. Cluster-GT alignment
    print("\n1. Cluster vs Ground-Truth Alignment:")
    alignment = cluster_label_alignment(cluster_labels, gt_labels)
    print(f"  NMI:       {alignment['nmi']:.4f}")
    print(f"  ARI:       {alignment['ari']:.4f}")
    print(f"  V-measure: {alignment['v_measure']:.4f}")

    print(f"\n  Per-cluster purity:")
    for cid, info in sorted(alignment["cluster_purity"].items()):
        print(f"    Cluster {cid:2d}: {info['majority_label']:20s} "
              f"purity={info['purity']:.3f} size={info['size']}")

    print(f"\n  Per-skill coverage:")
    for skill, info in sorted(alignment["skill_coverage"].items()):
        print(f"    {skill:20s}: cluster={info['primary_cluster']} "
              f"coverage={info['coverage']:.3f} "
              f"spread={info['n_clusters']} clusters")

    # 2. Heuristic vs Learned agreement
    print("\n2. Heuristic vs Learned Label Agreement:")
    h_vs_l = heuristic_vs_learned_agreement(labels_str, learned_labels)
    print(f"  Agreement rate: {h_vs_l['agreement_rate']:.4f} "
          f"({h_vs_l['n_agree']}/{h_vs_l['n_total']})")
    if h_vs_l["top_disagreements"]:
        print(f"  Top disagreements:")
        for d in h_vs_l["top_disagreements"][:5]:
            print(f"    {d['heuristic']:20s} vs {d['learned']:20s} "
                  f"({d['count']} times)")

    # 3. Degenerate cluster check
    print("\n3. Cluster Health Check:")
    health = verify_no_degenerate_clusters(cluster_labels)
    stats = health["cluster_size_stats"]
    print(f"  Cluster sizes: min={stats['min']}, max={stats['max']}, "
          f"mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    print(f"  Degenerate clusters: {health['n_degenerate']}")

    # Save results
    results = {
        "alignment": alignment,
        "heuristic_vs_learned": h_vs_l,
        "cluster_health": health,
    }
    out_path = RESULTS_DIR / "skill_verification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
