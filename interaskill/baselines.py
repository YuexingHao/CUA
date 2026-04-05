"""
Baseline methods for skill composition comparison.

Implements three baselines:
1. SKILL.md (Fixed Lookup) — predicts next skill from a static transition table
2. AWM (Agent Workflow Memory) — mines workflow patterns from trajectories offline
3. Frequency — always predicts the most common next skill (random baseline)

All baselines operate on the same trajectory data and are evaluated
with the same metrics as our learned models.
"""

import json
import random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from .data import SKILL_TYPES, SKILL_TO_IDX, N_SKILLS


# ══════════════════════════════════════════════════════════════════════
# Baseline 1: SKILL.md (Fixed Transition Table)
# ══════════════════════════════════════════════════════════════════════
#
# Simulates a hand-coded skill specification where transitions between
# skills are defined in a static lookup table. This is the paradigm
# our paper argues against.

# Hard-coded "expert" transition table — what a human might write in SKILL.md
# Maps (current_skill) → most likely next skill
SKILLMD_TRANSITIONS = {
    "search_navigate":   "document_edit",
    "document_edit":     "review_content",
    "review_content":    "export_publish",
    "export_publish":    "send_message",
    "send_message":      "collaborate",
    "collaborate":       "document_edit",
    "schedule_meeting":  "send_message",
    "data_transfer":     "document_edit",
    "organize_files":    "document_edit",
    "presentation_edit": "review_content",
    "monitor_status":    "document_edit",
    "generic_action":    "document_edit",
}

# Richer transition table derived from structured SKILL.md files
# (skills/ directory, following anthropics/skills format).
# Each skill maps to a weighted distribution over likely next skills.
SKILLMD_RICH_TRANSITIONS = {
    "search_navigate":   {"document_edit": 0.30, "review_content": 0.30, "data_transfer": 0.15, "presentation_edit": 0.10, "organize_files": 0.10, "collaborate": 0.05},
    "document_edit":     {"review_content": 0.30, "export_publish": 0.25, "data_transfer": 0.15, "collaborate": 0.10, "send_message": 0.10, "presentation_edit": 0.10},
    "review_content":    {"document_edit": 0.30, "export_publish": 0.25, "collaborate": 0.20, "send_message": 0.15, "presentation_edit": 0.10},
    "export_publish":    {"send_message": 0.45, "organize_files": 0.25, "collaborate": 0.15, "review_content": 0.15},
    "send_message":      {"monitor_status": 0.25, "collaborate": 0.20, "schedule_meeting": 0.15, "review_content": 0.15, "search_navigate": 0.15, "document_edit": 0.10},
    "collaborate":       {"send_message": 0.25, "document_edit": 0.25, "review_content": 0.15, "schedule_meeting": 0.15, "search_navigate": 0.10, "monitor_status": 0.10},
    "schedule_meeting":  {"send_message": 0.40, "collaborate": 0.25, "document_edit": 0.15, "monitor_status": 0.10, "search_navigate": 0.10},
    "data_transfer":     {"document_edit": 0.40, "presentation_edit": 0.25, "review_content": 0.15, "export_publish": 0.10, "organize_files": 0.10},
    "organize_files":    {"document_edit": 0.25, "send_message": 0.20, "review_content": 0.20, "search_navigate": 0.15, "export_publish": 0.10, "collaborate": 0.10},
    "presentation_edit": {"review_content": 0.35, "export_publish": 0.30, "send_message": 0.15, "collaborate": 0.10, "data_transfer": 0.10},
    "monitor_status":    {"document_edit": 0.30, "search_navigate": 0.25, "collaborate": 0.20, "send_message": 0.15, "review_content": 0.10},
    "generic_action":    {"document_edit": 0.25, "search_navigate": 0.25, "review_content": 0.20, "collaborate": 0.15, "send_message": 0.15},
}


class SkillMdBaseline:
    """Fixed skill transition table — the SKILL.md paradigm."""

    def __init__(self, use_rich=False):
        self.use_rich = use_rich
        self.transitions = SKILLMD_TRANSITIONS
        self.rich_transitions = SKILLMD_RICH_TRANSITIONS

    def predict_sequence(self, first_skill: str, length: int) -> list[str]:
        """Predict a skill sequence starting from first_skill."""
        seq = [first_skill]
        current = first_skill
        for _ in range(length - 1):
            if self.use_rich:
                dist = self.rich_transitions.get(current, {"document_edit": 1.0})
                next_skill = max(dist, key=dist.get)
            else:
                next_skill = self.transitions.get(current, "document_edit")
            seq.append(next_skill)
            current = next_skill
        return seq


# ══════════════════════════════════════════════════════════════════════
# Baseline 2: AWM (Agent Workflow Memory) — Simplified
# ══════════════════════════════════════════════════════════════════════
#
# AWM (Wang et al., 2024) mines reusable workflow routines from agent
# trajectories. Our simplified implementation:
#   1. Extracts all bigram (skill_i → skill_j) transitions from training data
#   2. Builds a probability table P(next_skill | current_skill)
#   3. At inference: greedily picks the most probable next skill
#
# This captures the core AWM idea — learning workflows from data —
# without the LLM-based induction step.

class AWMBaseline:
    """Agent Workflow Memory — learns transition probabilities from data."""

    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.transition_probs = {}

    def fit(self, traj_data_list):
        """Learn transition probabilities from training trajectories."""
        for td in traj_data_list:
            seq = td.skill_sequence
            for i in range(len(seq) - 1):
                self.transition_counts[seq[i]][seq[i + 1]] += 1

        # Normalize to probabilities
        for skill, counts in self.transition_counts.items():
            total = sum(counts.values())
            self.transition_probs[skill] = {
                next_s: c / total for next_s, c in counts.items()
            }

    def predict_next(self, current_skill: str) -> str:
        """Predict the most probable next skill."""
        probs = self.transition_probs.get(current_skill, {})
        if not probs:
            return "document_edit"  # fallback
        return max(probs, key=probs.get)

    def predict_sequence(self, first_skill: str, length: int) -> list[str]:
        """Predict a full sequence greedily."""
        seq = [first_skill]
        current = first_skill
        for _ in range(length - 1):
            next_skill = self.predict_next(current)
            seq.append(next_skill)
            current = next_skill
        return seq

    def predict_next_top_k(self, current_skill: str, k: int = 3) -> list[tuple]:
        """Return top-k predictions with probabilities."""
        probs = self.transition_probs.get(current_skill, {})
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return sorted_probs[:k]


# ══════════════════════════════════════════════════════════════════════
# Baseline 3: Frequency Baseline
# ══════════════════════════════════════════════════════════════════════
#
# Always predicts the globally most common skill. This is the simplest
# baseline and provides a floor for comparison.

class FrequencyBaseline:
    """Always predict the most common skill."""

    def __init__(self):
        self.most_common = "document_edit"
        self.skill_freq = {}

    def fit(self, traj_data_list):
        """Count skill frequencies."""
        counts = Counter()
        for td in traj_data_list:
            for s in td.skill_sequence:
                counts[s] += 1
        self.most_common = counts.most_common(1)[0][0]
        total = sum(counts.values())
        self.skill_freq = {s: c / total for s, c in counts.items()}

    def predict_sequence(self, first_skill: str, length: int) -> list[str]:
        """Always predict the most common skill."""
        return [first_skill] + [self.most_common] * (length - 1)


# ══════════════════════════════════════════════════════════════════════
# Evaluation Runner
# ══════════════════════════════════════════════════════════════════════

def evaluate_baseline(baseline, traj_data_list, name: str) -> dict:
    """Evaluate a baseline on skill sequence prediction.

    For fair comparison with our models:
    - Position 0 is given (first skill)
    - We predict positions 1..N
    - Metrics: exact match, edit distance, per-position accuracy
    """
    from .evaluate import sequence_exact_match, normalized_edit_distance, per_position_accuracy

    pred_seqs = []
    gt_seqs = []

    for td in traj_data_list:
        gt = td.skill_sequence
        if len(gt) < 2:
            continue
        pred = baseline.predict_sequence(gt[0], len(gt))
        pred_seqs.append(pred)
        gt_seqs.append(gt)

    em = sequence_exact_match(pred_seqs, gt_seqs)
    ed = normalized_edit_distance(pred_seqs, gt_seqs)
    pa = per_position_accuracy(pred_seqs, gt_seqs)

    print(f"\n  [{name}]")
    print(f"    Exact match:     {em:.4f}")
    print(f"    Edit distance:   {ed:.4f}")
    print(f"    Per-position:    {pa}")
    print(f"    Mean pos acc:    {np.mean([v for k, v in pa.items() if k > 0]):.4f}"
          if len(pa) > 1 else "")

    return {
        "name": name,
        "exact_match": float(em),
        "edit_distance": float(ed),
        "per_position_accuracy": {str(k): v for k, v in pa.items()},
        "num_sequences": len(pred_seqs),
    }


def run_all_baselines(traj_data_list) -> dict:
    """Run all baselines and return results dict."""
    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    results = {}

    # 1. Frequency baseline
    freq = FrequencyBaseline()
    freq.fit(traj_data_list)
    results["frequency"] = evaluate_baseline(freq, traj_data_list, "Frequency (most common)")

    # 2. SKILL.md baseline (simple)
    skillmd = SkillMdBaseline(use_rich=False)
    results["skillmd"] = evaluate_baseline(skillmd, traj_data_list, "SKILL.md (fixed table)")

    # 2b. SKILL.md baseline (rich — using structured skill definitions)
    skillmd_rich = SkillMdBaseline(use_rich=True)
    results["skillmd_rich"] = evaluate_baseline(skillmd_rich, traj_data_list, "SKILL.md (rich transitions)")

    # 3. AWM baseline
    # Split: train on 80%, eval on 20%
    np.random.seed(42)
    idx = np.random.permutation(len(traj_data_list))
    split = int(0.8 * len(idx))
    train_data = [traj_data_list[i] for i in idx[:split]]
    eval_data = [traj_data_list[i] for i in idx[split:]]

    awm = AWMBaseline()
    awm.fit(train_data)
    results["awm"] = evaluate_baseline(awm, eval_data, "AWM (learned transitions)")

    # Also eval AWM on full data for comparison
    awm_full = AWMBaseline()
    awm_full.fit(traj_data_list)
    results["awm_full"] = evaluate_baseline(awm_full, traj_data_list, "AWM (full data, oracle)")

    # Print AWM transition table
    print(f"\n  AWM Learned Transitions (top-1):")
    for skill in sorted(awm_full.transition_probs.keys()):
        top = awm_full.predict_next_top_k(skill, k=3)
        top_str = ", ".join(f"{s}({p:.2f})" for s, p in top)
        print(f"    {skill:20s} → {top_str}")

    return results


if __name__ == "__main__":
    from .data import load_and_featurize
    traj_data = load_and_featurize("data/fabricated_trajectories.json")
    results = run_all_baselines(traj_data)

    with open("results/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/baseline_metrics.json")
