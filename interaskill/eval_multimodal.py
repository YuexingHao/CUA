"""
Multimodal evaluation: screenshot-based skill prediction on Mind2Web.

Compares text-only (Qwen3-8B) vs screenshot-based (Qwen3-VL-8B) skill
prediction to validate the paper's multimodal grounding claims (Section 3.3).

Key experiments:
  1. VLM skill prediction: screenshot → skill (Qwen3-VL-8B)
  2. CLIP zero-shot: screenshot → CLIP embedding → nearest skill
  3. Text vs Vision ablation: same task, text-only vs screenshot input
  4. Cross-domain visual transfer: CLIP similarity across domains

Data: osunlp/Multimodal-Mind2Web (has screenshots + DOM + actions)

Usage:
    # VLM skill prediction
    python -m interaskill.eval_multimodal \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max-tasks 200

    # CLIP zero-shot baseline
    python -m interaskill.eval_multimodal \
        --mode clip-zeroshot --max-tasks 200

    # Cross-domain transfer analysis
    python -m interaskill.eval_multimodal \
        --mode cross-domain --max-tasks 500
"""

import argparse
import json
import torch
from pathlib import Path
from collections import defaultdict
from PIL import Image

from .data import SKILL_TYPES
from .grounding import classify_mind2web_step
from .metrics import (
    TaskResult, StepResult, metrics_to_json, task_completion_rate_ci,
)
from .multimodal import (
    VLMSkillPredictor, CLIPEncoder,
    compute_cross_domain_similarity,
    compute_skill_visual_invariance,
)

RESULTS_DIR = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal evaluation on Mind2Web with screenshots")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="VLM model name")
    parser.add_argument("--mode", type=str, default="vlm",
                        choices=["vlm", "clip-zeroshot", "cross-domain"],
                        help="Evaluation mode")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clip-model", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--composition", action="store_true",
                        help="Use model's own predictions in history")
    return parser.parse_args()


def load_multimodal_mind2web(max_tasks: int = 200,
                              cache_path: Path = None) -> list[dict]:
    """Load Multimodal-Mind2Web with screenshots.

    Each example has: screenshot (PIL Image), cleaned_html, action info.
    We group by annotation_id to reconstruct full tasks.
    """
    from datasets import load_dataset

    print("Loading Multimodal-Mind2Web from HuggingFace...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train",
                       streaming=True)

    # Group steps by annotation_id to form tasks
    tasks = {}
    n_loaded = 0
    for example in ds:
        ann_id = example.get("annotation_id", "")
        if ann_id not in tasks:
            if len(tasks) >= max_tasks:
                break
            tasks[ann_id] = {
                "task_id": f"mm2w_{ann_id[:8]}",
                "task": example.get("confirmed_task", ""),
                "website": example.get("website", ""),
                "domain": example.get("domain", ""),
                "subdomain": example.get("subdomain", ""),
                "steps": [],
            }

        # Parse action info
        op_str = example.get("operation", "{}")
        try:
            operation = json.loads(op_str) if isinstance(op_str, str) else op_str
        except json.JSONDecodeError:
            operation = {"op": "CLICK", "value": ""}

        step = {
            "step_idx": len(tasks[ann_id]["steps"]),
            "screenshot": example.get("screenshot"),  # PIL Image
            "action_type": operation.get("op", "CLICK").lower(),
            "value": operation.get("value", ""),
            "target_action": example.get("target_action_reprs", ""),
            "action_reprs": example.get("action_reprs", []),
            "pos_candidates": example.get("pos_candidates", []),
            "neg_candidates": example.get("neg_candidates", []),
        }
        tasks[ann_id]["steps"].append(step)
        n_loaded += 1

    task_list = list(tasks.values())
    print(f"  Loaded {len(task_list)} tasks ({n_loaded} steps total)")
    return task_list


def evaluate_vlm(args):
    """Experiment 1: VLM screenshot-based skill prediction."""
    print("=" * 60)
    print(f"Multimodal Skill Prediction (VLM)")
    print(f"Model: {args.model}")
    print(f"Max tasks: {args.max_tasks}")
    print("=" * 60)

    # Load VLM
    vlm = VLMSkillPredictor(args.model)

    # Load data
    tasks = load_multimodal_mind2web(args.max_tasks)

    # Evaluate
    task_results = []
    for ti, task in enumerate(tasks):
        task_desc = task["task"]
        history_parts = []
        all_correct = True

        result = TaskResult(
            task_id=task["task_id"],
            benchmark="mind2web_multimodal",
            task_description=task_desc,
            max_steps=len(task["steps"]),
            domain=task.get("domain", ""),
        )

        for step in task["steps"]:
            screenshot = step.get("screenshot")
            if screenshot is None:
                continue

            # Ground truth skill (heuristic from action)
            gt_skill = classify_mind2web_step(
                step["action_type"],
                step.get("target_action", ""),
                "",
                task_desc,
            )

            # VLM prediction from screenshot
            history_str = " → ".join(history_parts[-5:]) if history_parts else ""
            pred_skill = vlm.predict_skill(screenshot, task_desc, history_str)

            skill_correct = pred_skill == gt_skill

            # Action prediction from screenshot
            pred_action_str = vlm.predict_action(
                screenshot, task_desc, pred_skill, history_str)

            step_result = StepResult(
                step_idx=step["step_idx"],
                predicted_skill=pred_skill,
                ground_truth_skill=gt_skill,
                predicted_action=pred_action_str[:200],
                ground_truth_action=step.get("target_action", ""),
                skill_correct=skill_correct,
                action_correct=skill_correct,  # approximation
            )
            result.steps.append(step_result)

            if not skill_correct:
                all_correct = False

            if args.composition:
                history_parts.append(f"[{pred_skill}] {pred_action_str[:50]}")
            else:
                history_parts.append(step.get("target_action", ""))

        result.total_steps = len(result.steps)
        result.completed = all_correct
        result.reward = (
            sum(s.skill_correct for s in result.steps) / max(len(result.steps), 1)
        )
        task_results.append(result)

        if (ti + 1) % 10 == 0:
            skill_acc = sum(
                s.skill_correct for r in task_results for s in r.steps
            ) / max(sum(len(r.steps) for r in task_results), 1)
            print(f"  {ti+1}/{len(tasks)}: skill_acc={skill_acc:.3f}",
                  flush=True)

    # Save results
    _save_results(task_results, args.model, "vlm", args)


def evaluate_clip_zeroshot(args):
    """Experiment 2: CLIP zero-shot skill classification from screenshots."""
    print("=" * 60)
    print(f"CLIP Zero-Shot Skill Classification")
    print(f"CLIP model: {args.clip_model}")
    print("=" * 60)

    clip = CLIPEncoder(args.clip_model)
    tasks = load_multimodal_mind2web(args.max_tasks)

    task_results = []
    for ti, task in enumerate(tasks):
        result = TaskResult(
            task_id=task["task_id"],
            benchmark="mind2web_clip",
            task_description=task["task"],
            max_steps=len(task["steps"]),
            domain=task.get("domain", ""),
        )

        for step in task["steps"]:
            screenshot = step.get("screenshot")
            if screenshot is None:
                continue

            gt_skill = classify_mind2web_step(
                step["action_type"],
                step.get("target_action", ""),
                "",
                task["task"],
            )

            pred_skill, confidence = clip.zero_shot_classify(screenshot)

            result.steps.append(StepResult(
                step_idx=step["step_idx"],
                predicted_skill=pred_skill,
                ground_truth_skill=gt_skill,
                predicted_action=f"clip_zeroshot[{confidence:.3f}]",
                ground_truth_action=step.get("target_action", ""),
                skill_correct=(pred_skill == gt_skill),
                action_correct=False,  # CLIP doesn't predict actions
            ))

        result.total_steps = len(result.steps)
        result.completed = all(s.skill_correct for s in result.steps)
        result.reward = (
            sum(s.skill_correct for s in result.steps) / max(len(result.steps), 1)
        )
        task_results.append(result)

        if (ti + 1) % 20 == 0:
            skill_acc = sum(
                s.skill_correct for r in task_results for s in r.steps
            ) / max(sum(len(r.steps) for r in task_results), 1)
            print(f"  {ti+1}/{len(tasks)}: skill_acc={skill_acc:.3f}",
                  flush=True)

    _save_results(task_results, args.clip_model, "clip_zeroshot", args)


def evaluate_cross_domain(args):
    """Experiment 3: Cross-domain visual transfer analysis."""
    print("=" * 60)
    print(f"Cross-Domain Visual Transfer Analysis")
    print(f"CLIP model: {args.clip_model}")
    print("=" * 60)

    clip = CLIPEncoder(args.clip_model)
    tasks = load_multimodal_mind2web(args.max_tasks)

    # Collect screenshots by domain
    domain_screenshots = defaultdict(list)
    # Collect screenshots by (skill, domain)
    skill_domain_screenshots = defaultdict(lambda: defaultdict(list))

    max_per_domain = 50  # limit memory
    for task in tasks:
        domain = task.get("domain", "unknown")
        for step in task["steps"]:
            screenshot = step.get("screenshot")
            if screenshot is None:
                continue

            if len(domain_screenshots[domain]) < max_per_domain:
                domain_screenshots[domain].append(screenshot)

            skill = classify_mind2web_step(
                step["action_type"],
                step.get("target_action", ""),
                "",
                task["task"],
            )
            if len(skill_domain_screenshots[skill][domain]) < 20:
                skill_domain_screenshots[skill][domain].append(screenshot)

    print(f"\nDomains: {sorted(domain_screenshots.keys())}")
    for d, imgs in sorted(domain_screenshots.items()):
        print(f"  {d}: {len(imgs)} screenshots")

    # Cross-domain similarity
    print("\nComputing cross-domain visual similarity...")
    cross_domain = compute_cross_domain_similarity(clip, dict(domain_screenshots))

    print(f"\nCross-domain similarity (mean): {cross_domain['mean_cross_domain_sim']:.4f}")
    for pair, sim in sorted(cross_domain["pairwise_similarities"].items(),
                            key=lambda x: -x[1]):
        print(f"  {pair}: {sim:.4f}")

    # Per-skill invariance
    print("\nComputing per-skill visual invariance...")
    invariance = compute_skill_visual_invariance(
        clip, dict(skill_domain_screenshots))

    print(f"\nPer-skill invariance (lower = better transfer):")
    for skill, info in sorted(invariance.items(), key=lambda x: x[1]["invariance"]):
        print(f"  {skill:20s}: invariance={info['invariance']:.4f} "
              f"sim={info['mean_cross_domain_similarity']:.4f} "
              f"({info['n_domains']} domains, {info['n_screenshots']} imgs)")

    # Visual-semantic alignment (L_align)
    print("\nComputing visual-semantic alignment (L_align)...")
    alignment_losses = {}
    skill_embs = clip.skill_label_embeddings()
    for skill in sorted(skill_domain_screenshots.keys()):
        all_imgs = []
        for domain_imgs in skill_domain_screenshots[skill].values():
            all_imgs.extend(domain_imgs[:5])
        if all_imgs:
            img_embs = clip.batch_encode_images(all_imgs[:20])
            skill_emb = skill_embs[skill].to(img_embs.device)
            # L_align = mean ||φ_vision(v) - φ_text(l)||²
            l_align = (img_embs - skill_emb).pow(2).sum(dim=-1).mean().item()
            alignment_losses[skill] = l_align

    print(f"\nL_align per skill (lower = better vision-text alignment):")
    for skill, loss in sorted(alignment_losses.items(), key=lambda x: x[1]):
        print(f"  {skill:20s}: L_align = {loss:.4f}")

    # Save results
    results = {
        "cross_domain_similarity": cross_domain,
        "skill_invariance": invariance,
        "alignment_loss": alignment_losses,
        "mean_alignment_loss": float(np.mean(list(alignment_losses.values()))) if alignment_losses else 0.0,
        "n_tasks": len(tasks),
        "n_domains": len(domain_screenshots),
    }
    import numpy as np
    out_path = RESULTS_DIR / "multimodal_cross_domain.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def _save_results(task_results: list[TaskResult], model: str,
                  mode: str, args):
    """Save evaluation results."""
    metrics = metrics_to_json(task_results, model, f"mind2web_{mode}")
    metrics["mode"] = mode
    metrics["composition"] = getattr(args, "composition", False)

    # Per-domain breakdown
    domain_results = defaultdict(list)
    for r in task_results:
        domain_results[r.domain].append(r)
    metrics["per_domain"] = {
        domain: {
            "skill_accuracy": sum(
                s.skill_correct for r in results for s in r.steps
            ) / max(sum(len(r.steps) for r in results), 1),
            "n_tasks": len(results),
        }
        for domain, results in sorted(domain_results.items())
    }

    print(f"\n{'='*60}")
    print(f"RESULTS ({mode})")
    print(f"{'='*60}")
    print(f"Skill Prediction Accuracy: {metrics['skill_prediction_accuracy']:.4f}")
    print(f"Tasks: {metrics['n_tasks']}")

    print(f"\nPer-domain:")
    for domain, info in metrics["per_domain"].items():
        print(f"  {domain:20s}: {info['skill_accuracy']:.3f} ({info['n_tasks']} tasks)")

    # Short model name for file
    short = model.split("/")[-1].lower().replace("-instruct", "").replace("-", "_")
    out_path = RESULTS_DIR / f"{short}_{mode}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Predictions
    preds = [{
        "task_id": r.task_id,
        "domain": r.domain,
        "completed": r.completed,
        "steps": [{
            "predicted_skill": s.predicted_skill,
            "ground_truth_skill": s.ground_truth_skill,
            "skill_correct": s.skill_correct,
        } for s in r.steps],
    } for r in task_results]

    preds_path = RESULTS_DIR / f"{short}_{mode}_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved predictions to {preds_path}")


def main():
    args = parse_args()

    if args.mode == "vlm":
        evaluate_vlm(args)
    elif args.mode == "clip-zeroshot":
        evaluate_clip_zeroshot(args)
    elif args.mode == "cross-domain":
        evaluate_cross_domain(args)


if __name__ == "__main__":
    main()
