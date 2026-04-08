"""
Offline evaluation on Mind2Web benchmark.

Evaluates skill prediction + action grounding on real web task demonstrations.
Uses DOM snapshots (no live browser needed — reproducible for NeurIPS).

Standard Mind2Web protocol:
  Given task + action history + top-k candidate DOM elements
  → predict (element, action_type, value)

InteraSkill extension:
  Additionally predict which skill each step belongs to.

Usage:
    python -m interaskill.eval_mind2web \
        --model Qwen/Qwen3-8B \
        --adapter results/qwen3_lora/final_adapter \
        --split test_task --max-tasks 200

    python -m interaskill.eval_mind2web \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --split test_task --max-tasks 200
"""

import argparse
import json
import torch
from pathlib import Path
from collections import Counter

from .data import SKILL_TYPES, load_mind2web, mind2web_task_to_steps, _candidate_to_text
from .grounding import (
    SkillGrounder, classify_mind2web_step,
    MIND2WEB_ACTION_TYPES,
)
from .metrics import (
    TaskResult, StepResult, metrics_to_json,
    compute_all_metrics_with_perturbations,
    perturb_element_order, perturb_dynamic,
    PERTURBATION_LEVELS,
)
from .eval_model import (
    load_model_and_tokenizer, generate_response, model_short_name,
)

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
DATA_DIR = Path("data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate on Mind2Web benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--split", type=str, default="test_task",
                        choices=["test_task", "test_website", "test_domain"])
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--perturbations", action="store_true",
                        help="Run L4 robustness perturbation evaluation")
    parser.add_argument("--composition", action="store_true",
                        help="Use composition mode (autoregressive, not teacher forcing)")
    parser.add_argument("--skill-mode", type=str, default="llm",
                        choices=["llm", "heuristic", "learned", "oracle"],
                        help="Skill classification mode for ablation")
    parser.add_argument("--grounding-mode", type=str, default="llm",
                        choices=["llm", "heuristic", "direct"],
                        help="Action grounding mode for ablation")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    return parser.parse_args()


def evaluate_step(grounder: SkillGrounder, task_desc: str,
                  step: dict, history_str: str) -> StepResult:
    """Evaluate a single Mind2Web step.

    Returns StepResult with skill prediction and action grounding results.
    """
    candidates = step.get("candidates", [])
    gt_action_type = step.get("action_type", "click")
    gt_value = step.get("value", "")
    gt_element_text = step.get("element_text", "")
    gt_element_tag = step.get("element_tag", "")
    gt_idx = step.get("ground_truth_idx", 0)

    # Ground-truth skill classification (heuristic)
    gt_skill = classify_mind2web_step(
        gt_action_type, gt_element_text, gt_element_tag, task_desc)

    # Build observation from candidates
    obs_parts = []
    for i, c in enumerate(candidates[:20]):
        obs_parts.append(f"[{i}] {_candidate_to_text(c)}")
    observation = "\n".join(obs_parts) if obs_parts else "(empty page)"

    # Stage 1: Predict skill
    pred_skill = grounder.predict_skill(task_desc, observation, history_str)

    # Stage 2: Ground to action
    pred_elem_id, pred_action_type, pred_value = grounder.ground_action_mind2web(
        pred_skill, task_desc, history_str, candidates[:20])

    # Evaluate element selection
    try:
        pred_elem_idx = int(pred_elem_id)
    except (ValueError, TypeError):
        pred_elem_idx = -1
    element_correct = (pred_elem_idx == gt_idx)

    # Evaluate action type
    action_type_correct = (pred_action_type == gt_action_type)

    # Overall action correctness: element + action type must both be correct
    # For type/select, value must also match
    if gt_action_type in ("type", "select_option"):
        value_correct = (pred_value.strip().lower() == gt_value.strip().lower())
        action_correct = element_correct and action_type_correct and value_correct
    else:
        action_correct = element_correct and action_type_correct

    return StepResult(
        step_idx=step["step_idx"],
        predicted_skill=pred_skill,
        ground_truth_skill=gt_skill,
        predicted_action=f"{pred_action_type}[{pred_elem_id}]({pred_value})",
        ground_truth_action=f"{gt_action_type}[{gt_idx}]({gt_value})",
        skill_correct=(pred_skill == gt_skill),
        action_correct=action_correct,
        element_correct=element_correct,
    )


def evaluate_task(grounder: SkillGrounder, task: dict,
                  perturbation: str = "none",
                  composition: bool = False) -> TaskResult:
    """Evaluate all steps in a Mind2Web task.

    Args:
        composition: If True, use predicted actions in history (autoregressive).
                     If False, use ground-truth history (teacher forcing).
                     Composition mode tests full skill chaining — whether
                     errors in early predictions cascade to later steps.
    """
    task_desc = task["task"]
    steps = mind2web_task_to_steps(task)
    result = TaskResult(
        task_id=task["task_id"],
        benchmark="mind2web",
        task_description=task_desc,
        max_steps=len(steps),
        domain=task.get("domain", ""),
        perturbation=perturbation,
    )

    history_parts = []
    all_correct = True

    for step in steps:
        # Apply perturbations if requested
        if perturbation != "none" and step.get("candidates"):
            level = PERTURBATION_LEVELS.get(perturbation, 0.0)
            if "layout" in perturbation:
                step = dict(step)
                step["candidates"] = perturb_element_order(
                    step["candidates"], seed=hash(task["task_id"]) & 0xFFFFFFFF)
            elif "dynamic" in perturbation:
                step = dict(step)
                step["candidates"] = perturb_dynamic(
                    step["candidates"], level=level,
                    seed=hash(task["task_id"]) & 0xFFFFFFFF)

        history_str = " → ".join(history_parts[-5:]) if history_parts else ""
        step_result = evaluate_step(grounder, task_desc, step, history_str)
        result.steps.append(step_result)

        if not step_result.action_correct:
            all_correct = False

        if composition:
            # Composition mode: use model's own predictions in history
            # This tests whether the agent can recover from its own mistakes
            history_parts.append(
                f"[{step_result.predicted_skill}] {step_result.predicted_action}")
        else:
            # Teacher forcing: use ground-truth actions in history
            history_parts.append(
                step.get("action_repr", step_result.ground_truth_action))

    result.total_steps = len(steps)
    result.completed = all_correct
    result.reward = 1.0 if all_correct else (
        sum(s.action_correct for s in result.steps) / max(len(result.steps), 1)
    )

    return result


def _checkpoint_path(short_name: str, split: str) -> Path:
    return RESULTS_DIR / f"{short_name}_mind2web_{split}_checkpoint.json"


def save_checkpoint(path: Path, task_results: list[TaskResult], completed: int):
    """Save evaluation checkpoint."""
    data = {
        "completed_tasks": completed,
        "results": [
            {
                "task_id": r.task_id,
                "completed": r.completed,
                "reward": r.reward,
                "total_steps": r.total_steps,
                "domain": r.domain,
                "perturbation": r.perturbation,
                "steps": [
                    {
                        "step_idx": s.step_idx,
                        "predicted_skill": s.predicted_skill,
                        "ground_truth_skill": s.ground_truth_skill,
                        "predicted_action": s.predicted_action,
                        "ground_truth_action": s.ground_truth_action,
                        "skill_correct": s.skill_correct,
                        "action_correct": s.action_correct,
                        "element_correct": s.element_correct,
                    }
                    for s in r.steps
                ],
            }
            for r in task_results
        ],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)


def load_checkpoint(path: Path) -> tuple[list[TaskResult], int]:
    """Load checkpoint, returning (task_results, n_completed)."""
    if not path.exists():
        return [], 0
    with open(path) as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        tr = TaskResult(
            task_id=r["task_id"],
            benchmark="mind2web",
            task_description="",
            completed=r["completed"],
            reward=r["reward"],
            total_steps=r["total_steps"],
            domain=r.get("domain", ""),
            perturbation=r.get("perturbation", "none"),
        )
        for s in r["steps"]:
            tr.steps.append(StepResult(**s))
        results.append(tr)
    return results, data["completed_tasks"]


def main():
    args = parse_args()
    short_name = model_short_name(args.model)
    if args.adapter:
        short_name += "_lora"

    print("=" * 60)
    print(f"Mind2Web Evaluation")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Split: {args.split}")
    print(f"Max tasks: {args.max_tasks}")
    print(f"Skill mode: {args.skill_mode}")
    print(f"Grounding mode: {args.grounding_mode}")
    print(f"Composition: {args.composition}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    # Load learned classifier if requested
    learned_classifier = None
    if args.skill_mode == "learned":
        from .grounding import LearnedSkillClassifier
        learned_classifier = LearnedSkillClassifier()

    grounder = SkillGrounder(model, tokenizer, generate_response,
                             benchmark="mind2web",
                             skill_mode=args.skill_mode,
                             grounding_mode=args.grounding_mode,
                             learned_classifier=learned_classifier)

    # Load Mind2Web data
    data_path = DATA_DIR / f"mind2web_{args.split}.json"
    if data_path.exists():
        print(f"Loading from cached {data_path}...")
        with open(data_path) as f:
            tasks = json.load(f)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
    else:
        # Download first if not cached
        print(f"Data not found at {data_path}. Downloading Mind2Web...")
        import subprocess
        subprocess.run([
            "python", "data/download_mind2web.py",
            "--max-tasks", str(args.max_tasks or 500),
        ], check=True)
        if data_path.exists():
            with open(data_path) as f:
                tasks = json.load(f)
            if args.max_tasks:
                tasks = tasks[:args.max_tasks]
        else:
            # Fallback: load from HF directly (uses 'train' split)
            print(f"Loading Mind2Web from HuggingFace...")
            tasks = load_mind2web("train", args.max_tasks, args.cache_dir)

    print(f"Loaded {len(tasks)} tasks", flush=True)

    # Resume from checkpoint
    ckpt_path = _checkpoint_path(short_name, args.split)
    start_task = 0
    task_results = []
    if args.resume:
        task_results, start_task = load_checkpoint(ckpt_path)
        if start_task > 0:
            print(f"Resuming from task {start_task} "
                  f"({len(task_results)} results loaded)", flush=True)

    # Evaluate
    for ti in range(start_task, len(tasks)):
        task = tasks[ti]
        result = evaluate_task(grounder, task, composition=args.composition)
        task_results.append(result)

        if (ti + 1) % 10 == 0:
            completed = sum(1 for r in task_results if r.completed)
            total_steps = sum(len(r.steps) for r in task_results)
            correct_steps = sum(
                sum(s.action_correct for s in r.steps) for r in task_results)
            print(f"  {ti+1}/{len(tasks)} tasks: "
                  f"TCR={completed}/{len(task_results)} "
                  f"({completed/len(task_results):.3f}), "
                  f"step_acc={correct_steps}/{total_steps} "
                  f"({correct_steps/max(total_steps,1):.3f})",
                  flush=True)
            save_checkpoint(ckpt_path, task_results, ti + 1)

    # Run perturbations if requested
    results_by_perturbation = {"none": task_results}
    if args.perturbations:
        for level_name in ["layout_10", "layout_20", "layout_50"]:
            print(f"\nRunning perturbation: {level_name}...")
            perturbed_results = []
            for ti, task in enumerate(tasks[:len(task_results)]):
                result = evaluate_task(grounder, task, perturbation=level_name)
                perturbed_results.append(result)
                if (ti + 1) % 50 == 0:
                    completed = sum(1 for r in perturbed_results if r.completed)
                    print(f"  {ti+1}/{len(tasks)} [{level_name}]: "
                          f"TCR={completed/len(perturbed_results):.3f}",
                          flush=True)
            results_by_perturbation[level_name] = perturbed_results

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    metrics = metrics_to_json(task_results, args.model, "mind2web")
    metrics["split"] = args.split
    metrics["adapter"] = args.adapter
    metrics["skill_mode"] = args.skill_mode
    metrics["grounding_mode"] = args.grounding_mode
    metrics["composition"] = args.composition

    # Per-domain breakdown
    domain_results = {}
    for r in task_results:
        domain_results.setdefault(r.domain, []).append(r)
    metrics["per_domain"] = {
        domain: {
            "tcr": sum(1 for r in results if r.completed) / len(results),
            "n_tasks": len(results),
        }
        for domain, results in sorted(domain_results.items())
    }

    # Add perturbation results
    if args.perturbations:
        from .metrics import robustness_summary
        metrics["level4_robustness"] = robustness_summary(results_by_perturbation)

    print(f"\nTask Completion Rate: {metrics['level1_tcr']:.4f} "
          f"({metrics['n_completed']}/{metrics['n_tasks']})")
    print(f"Skill Prediction Accuracy: {metrics['skill_prediction_accuracy']:.4f}")
    print(f"Action Grounding Accuracy: {metrics['action_grounding_accuracy']:.4f}")
    print(f"Partial Task Success (mean): {metrics['level2_pts']['mean']:.4f}")

    eff = metrics["level3_efficiency"]
    if eff["avg_steps_to_complete"] is not None:
        print(f"Avg Steps to Completion: {eff['avg_steps_to_complete']:.1f}")
    if eff["recovery_rate"] is not None:
        print(f"Recovery Rate: {eff['recovery_rate']:.4f}")

    print(f"\nPer-domain TCR:")
    for domain, info in metrics["per_domain"].items():
        print(f"  {domain:20s}: {info['tcr']:.3f} ({info['n_tasks']} tasks)")

    if args.perturbations:
        print(f"\nRobustness (L4):")
        for level, info in metrics["level4_robustness"].items():
            print(f"  {level:15s}: TCR={info['tcr']:.3f} "
                  f"[{info['ci_lo']:.3f}, {info['ci_hi']:.3f}]")

    # Save results
    suffix = f"_{args.split}"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = METRICS_DIR / f"{short_name}_mind2web_metrics{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {results_path}")

    # Save detailed predictions
    preds = []
    for r in task_results:
        preds.append({
            "task_id": r.task_id,
            "domain": r.domain,
            "completed": r.completed,
            "reward": r.reward,
            "total_steps": r.total_steps,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "predicted_skill": s.predicted_skill,
                    "ground_truth_skill": s.ground_truth_skill,
                    "predicted_action": s.predicted_action,
                    "ground_truth_action": s.ground_truth_action,
                    "skill_correct": s.skill_correct,
                    "action_correct": s.action_correct,
                    "element_correct": s.element_correct,
                }
                for s in r.steps
            ],
        })

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = PREDICTIONS_DIR / f"{short_name}_mind2web_predictions{suffix}.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved predictions to {preds_path}")

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
