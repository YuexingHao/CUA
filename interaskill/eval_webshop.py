"""
End-to-end evaluation on WebShop benchmark.

Evaluates whether the agent can complete full e-commerce workflows:
  search → browse → select options → purchase

Two-stage prediction:
  1. Skill Predictor: predicts next InteraSkill skill
  2. Skill Grounder: maps skill + page observation → WebShop action

Usage:
    python -m interaskill.eval_webshop \
        --model Qwen/Qwen3-8B \
        --adapter results/qwen3_lora/final_adapter \
        --max-tasks 200

    python -m interaskill.eval_webshop \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --max-tasks 200
"""

import argparse
import json
import torch
from pathlib import Path

from .data import SKILL_TYPES, WEBSHOP_CANONICAL_SEQUENCE
from .grounding import SkillGrounder, HeuristicGrounder, classify_webshop_phase
from .envs.webshop_env import (
    WebShopSkillEnv, detect_page_type, page_type_to_skill,
    extract_available_actions,
)
from .metrics import (
    TaskResult, StepResult, metrics_to_json,
    compute_all_metrics_with_perturbations,
    perturb_style, PERTURBATION_LEVELS,
)
from .eval_model import (
    load_model_and_tokenizer, generate_response, model_short_name,
)

RESULTS_DIR = Path("results")
MAX_STEPS_PER_TASK = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation on WebShop")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TASK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--perturbations", action="store_true",
                        help="Run L4 robustness perturbation evaluation")
    parser.add_argument("--skill-mode", type=str, default="llm",
                        choices=["llm", "heuristic", "learned", "oracle"],
                        help="Skill classification mode for ablation")
    parser.add_argument("--grounding-mode", type=str, default="llm",
                        choices=["llm", "heuristic", "direct"],
                        help="Action grounding mode for ablation")
    parser.add_argument("--webshop-dir", type=str,
                        default="third_party/WebShop")
    return parser.parse_args()


def run_task(env: WebShopSkillEnv, grounder, task_idx: int,
             task_desc: str, max_steps: int,
             perturbation: str = "none") -> TaskResult:
    """Run a single WebShop task end-to-end.

    Returns TaskResult with all step-level details.
    """
    state = env.reset(task_idx)
    task = state["task"]
    if not task_desc:
        task_desc = task

    result = TaskResult(
        task_id=f"webshop_{task_idx}",
        benchmark="webshop",
        task_description=task_desc,
        max_steps=max_steps,
        perturbation=perturbation,
    )

    history_parts = []
    done = False

    for step_idx in range(max_steps):
        obs = state["observation"]
        available_actions = state["available_actions"]

        # Apply perturbations to observation
        if perturbation != "none":
            level = PERTURBATION_LEVELS.get(perturbation, 0.0)
            obs = perturb_style(obs, level=level,
                                seed=hash(f"{task_idx}_{step_idx}") & 0xFFFFFFFF)
            # Re-extract actions from perturbed observation
            available_actions = extract_available_actions(obs)

        # Ground-truth skill phase (from observation)
        gt_skill = state["skill"]

        # Stage 1: Predict skill
        history_str = " → ".join(history_parts[-5:]) if history_parts else ""

        if isinstance(grounder, SkillGrounder):
            pred_skill = grounder.predict_skill(task_desc, obs, history_str)
        else:
            pred_skill = classify_webshop_phase(obs)

        # Stage 2: Ground to action
        if isinstance(grounder, SkillGrounder):
            pred_action = grounder.ground_action_webshop(
                pred_skill, task_desc, obs, available_actions)
        else:
            pred_action = grounder.ground_action_webshop(
                pred_skill, obs, available_actions)

        # Execute action
        state, reward, done, info = env.step(pred_action)

        # Record step
        step_result = StepResult(
            step_idx=step_idx,
            predicted_skill=pred_skill,
            ground_truth_skill=gt_skill,
            predicted_action=pred_action,
            ground_truth_action="",  # WebShop doesn't provide GT actions
            skill_correct=(pred_skill == gt_skill),
            action_correct=done and reward > 0,  # only fully known at end
        )
        result.steps.append(step_result)
        history_parts.append(f"[{pred_skill}] {pred_action}")

        if done:
            break

    result.total_steps = len(result.steps)
    result.reward = reward if done else 0.0
    result.completed = (reward >= 1.0) if done else False

    # Update action_correct for all steps based on sub-goals
    subgoal_vector = env.subgoals.completion_vector
    for i, step in enumerate(result.steps):
        # Mark steps as correct if they contributed to sub-goal progress
        page_type = detect_page_type(env.current_obs)
        step.action_correct = (
            (i < len(subgoal_vector) and subgoal_vector[min(i, len(subgoal_vector) - 1)])
            or result.completed
        )

    return result


def _checkpoint_path(short_name: str) -> Path:
    return RESULTS_DIR / f"{short_name}_webshop_checkpoint.json"


def save_checkpoint(path: Path, task_results: list[TaskResult],
                    completed_tasks: int):
    """Save WebShop evaluation checkpoint."""
    data = {
        "completed_tasks": completed_tasks,
        "results": [
            {
                "task_id": r.task_id,
                "completed": r.completed,
                "reward": r.reward,
                "total_steps": r.total_steps,
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
    """Load checkpoint."""
    if not path.exists():
        return [], 0
    with open(path) as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        tr = TaskResult(
            task_id=r["task_id"],
            benchmark="webshop",
            task_description="",
            completed=r["completed"],
            reward=r["reward"],
            total_steps=r["total_steps"],
            perturbation=r.get("perturbation", "none"),
        )
        for s in r["steps"]:
            tr.steps.append(StepResult(
                step_idx=s["step_idx"],
                predicted_skill=s["predicted_skill"],
                ground_truth_skill=s["ground_truth_skill"],
                predicted_action=s["predicted_action"],
                ground_truth_action=s.get("ground_truth_action", ""),
                skill_correct=s["skill_correct"],
                action_correct=s["action_correct"],
            ))
        results.append(tr)
    return results, data["completed_tasks"]


def main():
    args = parse_args()
    short_name = model_short_name(args.model)
    if args.adapter:
        short_name += "_lora"

    print("=" * 60)
    print(f"WebShop End-to-End Evaluation")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Max tasks: {args.max_tasks}")
    print(f"Max steps/task: {args.max_steps}")
    print(f"Skill mode: {args.skill_mode}")
    print(f"Grounding mode: {args.grounding_mode}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    # Load learned classifier if requested
    learned_classifier = None
    if args.skill_mode == "learned":
        from .grounding import LearnedSkillClassifier
        learned_classifier = LearnedSkillClassifier()

    # Create grounder
    if args.grounding_mode == "heuristic" and args.skill_mode == "heuristic":
        grounder = HeuristicGrounder()
    else:
        grounder = SkillGrounder(model, tokenizer, generate_response,
                                 benchmark="webshop",
                                 skill_mode=args.skill_mode,
                                 grounding_mode=args.grounding_mode,
                                 learned_classifier=learned_classifier)

    # Initialize WebShop environment
    print(f"\nInitializing WebShop from {args.webshop_dir}...")
    env = WebShopSkillEnv(args.webshop_dir)
    n_tasks = min(args.max_tasks, env.n_tasks)
    print(f"Running {n_tasks} tasks", flush=True)

    # Resume from checkpoint
    ckpt_path = _checkpoint_path(short_name)
    start_task = 0
    task_results = []
    if args.resume:
        task_results, start_task = load_checkpoint(ckpt_path)
        if start_task > 0:
            print(f"Resuming from task {start_task} "
                  f"({len(task_results)} results loaded)", flush=True)

    # Evaluate
    for ti in range(start_task, n_tasks):
        result = run_task(env, grounder, ti, "", args.max_steps)
        task_results.append(result)

        if (ti + 1) % 10 == 0:
            completed = sum(1 for r in task_results if r.completed)
            avg_reward = sum(r.reward for r in task_results) / len(task_results)
            avg_steps = sum(r.total_steps for r in task_results) / len(task_results)
            print(f"  {ti+1}/{n_tasks}: "
                  f"TCR={completed}/{len(task_results)} "
                  f"({completed/len(task_results):.3f}), "
                  f"avg_reward={avg_reward:.3f}, "
                  f"avg_steps={avg_steps:.1f}",
                  flush=True)
            save_checkpoint(ckpt_path, task_results, ti + 1)

    # Run perturbations if requested
    results_by_perturbation = {"none": task_results}
    if args.perturbations:
        for level_name in ["layout_10", "layout_20", "layout_50"]:
            print(f"\nRunning perturbation: {level_name}...")
            perturbed_results = []
            for ti in range(len(task_results)):
                result = run_task(
                    env, grounder, ti, "", args.max_steps,
                    perturbation=level_name)
                perturbed_results.append(result)
                if (ti + 1) % 50 == 0:
                    completed = sum(1 for r in perturbed_results if r.completed)
                    print(f"  {ti+1} [{level_name}]: "
                          f"TCR={completed/len(perturbed_results):.3f}",
                          flush=True)
            results_by_perturbation[level_name] = perturbed_results

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    metrics = metrics_to_json(task_results, args.model, "webshop")
    metrics["adapter"] = args.adapter
    metrics["skill_mode"] = args.skill_mode
    metrics["grounding_mode"] = args.grounding_mode

    # WebShop-specific metrics
    metrics["average_reward"] = (
        sum(r.reward for r in task_results) / len(task_results)
        if task_results else 0.0)

    # Skill sequence edit distance vs canonical
    from .eval_model import normalized_edit_distance as ned_fn
    pred_seqs = [[s.predicted_skill for s in r.steps] for r in task_results]
    canon_seqs = [WEBSHOP_CANONICAL_SEQUENCE[:len(seq)] for seq in pred_seqs]
    metrics["skill_seq_edit_dist_vs_canonical"] = ned_fn(pred_seqs, canon_seqs)

    # Add perturbation results
    if args.perturbations:
        from .metrics import robustness_summary
        metrics["level4_robustness"] = robustness_summary(results_by_perturbation)

    print(f"\nTask Completion Rate: {metrics['level1_tcr']:.4f} "
          f"({metrics['n_completed']}/{metrics['n_tasks']})")
    print(f"Average Reward: {metrics['average_reward']:.4f}")
    print(f"Skill Prediction Accuracy: {metrics['skill_prediction_accuracy']:.4f}")
    print(f"Partial Task Success (mean): {metrics['level2_pts']['mean']:.4f}")

    eff = metrics["level3_efficiency"]
    if eff["avg_steps_to_complete"] is not None:
        print(f"Avg Steps to Completion: {eff['avg_steps_to_complete']:.1f}")
    if eff["recovery_rate"] is not None:
        print(f"Recovery Rate: {eff['recovery_rate']:.4f}")

    print(f"Skill Seq Edit Dist (vs canonical): "
          f"{metrics['skill_seq_edit_dist_vs_canonical']:.4f}")

    if args.perturbations:
        print(f"\nRobustness (L4):")
        for level, info in metrics["level4_robustness"].items():
            print(f"  {level:15s}: TCR={info['tcr']:.3f} "
                  f"[{info['ci_lo']:.3f}, {info['ci_hi']:.3f}]")

    # Save results
    results_path = RESULTS_DIR / f"{short_name}_webshop_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {results_path}")

    # Save predictions
    preds = []
    for r in task_results:
        preds.append({
            "task_id": r.task_id,
            "completed": r.completed,
            "reward": r.reward,
            "total_steps": r.total_steps,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "predicted_skill": s.predicted_skill,
                    "ground_truth_skill": s.ground_truth_skill,
                    "predicted_action": s.predicted_action,
                    "skill_correct": s.skill_correct,
                    "action_correct": s.action_correct,
                }
                for s in r.steps
            ],
        })

    preds_path = RESULTS_DIR / f"{short_name}_webshop_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved predictions to {preds_path}")

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
