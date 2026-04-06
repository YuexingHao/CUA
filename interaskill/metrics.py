"""
Hierarchical success metrics for end-to-end task evaluation.

Level 1: Task Completion Rate (TCR)
Level 2: Partial Task Success (PTS)
Level 3: Task Efficiency (steps, skill switches, recovery)
Level 4: Robustness under distribution shift (perturbations)
"""

import random
import numpy as np
from dataclasses import dataclass, field


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single action step within a task."""
    step_idx: int
    predicted_skill: str
    ground_truth_skill: str
    predicted_action: str  # e.g. "click[Buy Now]" or "type[red jacket]"
    ground_truth_action: str
    skill_correct: bool
    action_correct: bool
    element_correct: bool = False  # Mind2Web: correct DOM element selected


@dataclass
class TaskResult:
    """Result of a full end-to-end task execution."""
    task_id: str
    benchmark: str  # "webshop" or "mind2web"
    task_description: str
    steps: list[StepResult] = field(default_factory=list)
    completed: bool = False
    reward: float = 0.0  # WebShop: partial credit [0,1]; Mind2Web: binary
    total_steps: int = 0
    max_steps: int = 15
    domain: str = ""  # Mind2Web domain (travel, shopping, etc.)
    perturbation: str = "none"  # "none", "layout_10", "layout_20", etc.


# ── Level 1: Task Completion Rate ────────────────────────────────────

def task_completion_rate(results: list[TaskResult]) -> dict:
    """L1: Fraction of tasks fully completed."""
    if not results:
        return {"tcr": 0.0, "n_completed": 0, "n_total": 0}
    completed = sum(1 for r in results if r.completed)
    return {
        "tcr": completed / len(results),
        "n_completed": completed,
        "n_total": len(results),
    }


def task_completion_rate_ci(results: list[TaskResult],
                            n_boot: int = 10000, ci: float = 0.95) -> dict:
    """L1 with bootstrap 95% CI."""
    if not results:
        return {"tcr": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    completions = np.array([int(r.completed) for r in results])
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(completions, size=len(completions), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return {
        "tcr": completions.mean(),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


# ── Level 2: Partial Task Success ────────────────────────────────────

def partial_task_success(results: list[TaskResult]) -> dict:
    """L2: Per-step success breakdown."""
    if not results:
        return {"pts_mean": 0.0, "per_step": [], "per_task": []}

    per_task_scores = []
    # Collect per-position stats
    max_pos = max((len(r.steps) for r in results), default=0)
    pos_correct = [0] * max_pos
    pos_total = [0] * max_pos

    for r in results:
        if not r.steps:
            per_task_scores.append(0.0)
            continue
        step_successes = [int(s.action_correct) for s in r.steps]
        per_task_scores.append(sum(step_successes) / len(step_successes))
        for i, s in enumerate(r.steps):
            if i < max_pos:
                pos_total[i] += 1
                pos_correct[i] += int(s.action_correct)

    per_step_acc = [
        pos_correct[i] / pos_total[i] if pos_total[i] > 0 else 0.0
        for i in range(max_pos)
    ]

    return {
        "pts_mean": float(np.mean(per_task_scores)) if per_task_scores else 0.0,
        "per_step": per_step_acc,
        "per_task": per_task_scores,
    }


def skill_stage_breakdown(results: list[TaskResult]) -> dict:
    """L2 diagnostic: which skill types fail most often?"""
    skill_correct = {}
    skill_total = {}
    for r in results:
        for s in r.steps:
            gt = s.ground_truth_skill
            skill_total[gt] = skill_total.get(gt, 0) + 1
            skill_correct[gt] = skill_correct.get(gt, 0) + int(s.action_correct)
    return {
        skill: {
            "accuracy": skill_correct.get(skill, 0) / skill_total[skill],
            "correct": skill_correct.get(skill, 0),
            "total": skill_total[skill],
        }
        for skill in sorted(skill_total)
    }


# ── Level 3: Task Efficiency ─────────────────────────────────────────

def task_efficiency(results: list[TaskResult]) -> dict:
    """L3: Steps-to-completion, skill switches, recovery rate."""
    completed = [r for r in results if r.completed]
    all_with_steps = [r for r in results if r.steps]

    # Steps to completion (only for completed tasks)
    steps_to_complete = [r.total_steps for r in completed] if completed else []

    # Skill switches: count transitions between different skills
    switch_counts = []
    for r in all_with_steps:
        switches = 0
        for i in range(1, len(r.steps)):
            if r.steps[i].predicted_skill != r.steps[i - 1].predicted_skill:
                switches += 1
        switch_counts.append(switches)

    # Recovery rate: operationalized as "agent took a DIFFERENT action after a
    # failed step, and the subsequent step succeeded". This measures adaptive
    # behavior, not just task-level luck.
    tasks_with_failures = 0
    recovered_tasks = 0
    recovery_attempts = 0  # steps where agent changed approach after failure
    recovery_successes = 0  # of those, how many succeeded

    for r in all_with_steps:
        has_failure = any(not s.action_correct for s in r.steps)
        if has_failure:
            tasks_with_failures += 1
            if r.completed:
                recovered_tasks += 1

        # Step-level recovery: after a failed step, did the agent adapt?
        for i in range(1, len(r.steps)):
            prev = r.steps[i - 1]
            curr = r.steps[i]
            if not prev.action_correct:
                # Agent had a failure — did it change strategy?
                changed = (curr.predicted_skill != prev.predicted_skill or
                           curr.predicted_action != prev.predicted_action)
                if changed:
                    recovery_attempts += 1
                    if curr.action_correct:
                        recovery_successes += 1

    return {
        "avg_steps_to_complete": float(np.mean(steps_to_complete)) if steps_to_complete else None,
        "median_steps_to_complete": float(np.median(steps_to_complete)) if steps_to_complete else None,
        "avg_skill_switches": float(np.mean(switch_counts)) if switch_counts else 0.0,
        "task_recovery_rate": (recovered_tasks / tasks_with_failures
                               if tasks_with_failures > 0 else None),
        "step_recovery_attempts": recovery_attempts,
        "step_recovery_successes": recovery_successes,
        "step_recovery_rate": (recovery_successes / recovery_attempts
                               if recovery_attempts > 0 else None),
        "tasks_with_failures": tasks_with_failures,
        "recovered_tasks": recovered_tasks,
    }


# ── Level 4: Robustness Under Distribution Shift ─────────────────────

def perturb_element_order(candidates: list[dict], seed: int = 42) -> list[dict]:
    """Shuffle the ordering of candidate DOM elements."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


def perturb_layout(observation: str, level: float = 0.1,
                   seed: int = 42) -> str:
    """Perturb DOM element attributes to simulate layout changes.

    Args:
        observation: Text/HTML observation string
        level: Perturbation intensity (0.1 = 10%, 0.5 = 50%)
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    lines = observation.split("\n")
    n_perturb = max(1, int(len(lines) * level))
    indices = rng.sample(range(len(lines)), min(n_perturb, len(lines)))

    perturbed = list(lines)
    for idx in indices:
        line = perturbed[idx]
        # Swap adjacent words with probability proportional to level
        words = line.split()
        if len(words) >= 2:
            swap_idx = rng.randint(0, len(words) - 2)
            words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]
            perturbed[idx] = " ".join(words)

    return "\n".join(perturbed)


def perturb_style(observation: str, level: float = 0.1,
                  seed: int = 42) -> str:
    """Rename button/link text to simulate style changes.

    Args:
        observation: Text observation
        level: Fraction of elements to rename
    """
    rng = random.Random(seed)
    renames = {
        "Buy Now": ["Purchase", "Add to Cart", "Get It"],
        "Add to Cart": ["Buy", "Add", "Get This"],
        "Search": ["Find", "Look up", "Go"],
        "Submit": ["Send", "Confirm", "Done"],
        "Back": ["Return", "Go Back", "Previous"],
        "Next": ["Continue", "Forward", "Proceed"],
    }
    result = observation
    for original, alternatives in renames.items():
        if original in result and rng.random() < level:
            result = result.replace(original, rng.choice(alternatives), 1)
    return result


def perturb_dynamic(candidates: list[dict], level: float = 0.1,
                    seed: int = 42) -> list[dict]:
    """Add/remove distractor elements to simulate dynamic page changes.

    Args:
        candidates: List of candidate DOM elements
        level: Fraction of elements to add/remove
    """
    rng = random.Random(seed)
    result = list(candidates)
    n_change = max(1, int(len(result) * level))

    # Remove some elements
    n_remove = n_change // 2
    if n_remove > 0 and len(result) > n_remove:
        remove_indices = rng.sample(range(len(result)), n_remove)
        result = [e for i, e in enumerate(result) if i not in remove_indices]

    # Add distractor elements
    n_add = n_change - n_remove
    distractors = [
        {"tag": "div", "text": "Advertisement", "id": f"ad_{i}"}
        for i in range(n_add)
    ]
    for d in distractors:
        insert_pos = rng.randint(0, len(result))
        result.insert(insert_pos, d)

    return result


PERTURBATION_LEVELS = {
    "none": 0.0,
    "layout_10": 0.1,
    "layout_20": 0.2,
    "layout_50": 0.5,
}


def robustness_summary(results_by_perturbation: dict[str, list[TaskResult]]) -> dict:
    """L4: Compare TCR across perturbation levels.

    Args:
        results_by_perturbation: {"none": [...], "layout_10": [...], ...}
    """
    summary = {}
    for level_name, results in results_by_perturbation.items():
        tcr = task_completion_rate_ci(results)
        summary[level_name] = {
            "tcr": tcr["tcr"],
            "ci_lo": tcr["ci_lo"],
            "ci_hi": tcr["ci_hi"],
            "n_tasks": len(results),
        }
    return summary


# ── Aggregate All Levels ─────────────────────────────────────────────

def compute_all_metrics(results: list[TaskResult]) -> dict:
    """Compute all 4 levels of hierarchical metrics."""
    return {
        "level1_tcr": task_completion_rate_ci(results),
        "level2_pts": partial_task_success(results),
        "level2_skill_breakdown": skill_stage_breakdown(results),
        "level3_efficiency": task_efficiency(results),
    }


def compute_all_metrics_with_perturbations(
    results_by_perturbation: dict[str, list[TaskResult]]
) -> dict:
    """Compute all metrics including L4 robustness."""
    base_results = results_by_perturbation.get("none", [])
    metrics = compute_all_metrics(base_results)
    metrics["level4_robustness"] = robustness_summary(results_by_perturbation)
    return metrics


def metrics_to_json(results: list[TaskResult], model: str,
                    benchmark: str) -> dict:
    """Format metrics for JSON output, matching existing result format."""
    metrics = compute_all_metrics(results)

    # Skill prediction accuracy (separate from action accuracy)
    all_steps = [s for r in results for s in r.steps]
    skill_acc = (sum(s.skill_correct for s in all_steps) / len(all_steps)
                 if all_steps else 0.0)
    action_acc = (sum(s.action_correct for s in all_steps) / len(all_steps)
                  if all_steps else 0.0)

    return {
        "model": model,
        "benchmark": benchmark,
        "level1_tcr": metrics["level1_tcr"]["tcr"],
        "level1_ci_lo": metrics["level1_tcr"]["ci_lo"],
        "level1_ci_hi": metrics["level1_tcr"]["ci_hi"],
        "level2_pts": {
            "mean": metrics["level2_pts"]["pts_mean"],
            "per_step": metrics["level2_pts"]["per_step"],
        },
        "level2_skill_breakdown": metrics["level2_skill_breakdown"],
        "level3_efficiency": metrics["level3_efficiency"],
        "skill_prediction_accuracy": skill_acc,
        "action_grounding_accuracy": action_acc,
        "n_tasks": len(results),
        "n_completed": sum(1 for r in results if r.completed),
        "n_steps_total": len(all_steps),
    }
