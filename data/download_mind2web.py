"""
Download and preprocess the Mind2Web dataset from HuggingFace.

Mind2Web (Deng et al., NeurIPS 2023): 2,350+ real web tasks across 137 websites.
Dataset: osunlp/Mind2Web

Usage:
    python data/download_mind2web.py [--cache-dir PATH] [--max-tasks N]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Mind2Web dataset")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to download per split")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    # Mind2Web has multiple test splits for generalization evaluation
    splits = {
        "test_task": "test_task",       # unseen tasks on seen websites
        "test_website": "test_website", # unseen websites in seen domains
        "test_domain": "test_domain",   # unseen domains entirely
    }

    for split_name, split_key in splits.items():
        print(f"Downloading {split_name}...")
        try:
            dataset = load_dataset("osunlp/Mind2Web", split=split_key, **kwargs)
        except Exception as e:
            print(f"  Warning: Could not load split '{split_key}': {e}")
            continue

        tasks = []
        for i, example in enumerate(dataset):
            if args.max_tasks and i >= args.max_tasks:
                break

            task = {
                "task_id": f"m2w_{split_name}_{i}",
                "task": example.get("confirmed_task", example.get("task", "")),
                "website": example.get("website", ""),
                "domain": example.get("domain", ""),
                "subdomain": example.get("subdomain", ""),
                "action_reprs": example.get("action_reprs", []),
                "num_actions": len(example.get("action_reprs", [])),
            }

            # Store structured actions
            actions = example.get("actions", [])
            task["actions"] = []
            for action in actions:
                if isinstance(action, dict):
                    task["actions"].append({
                        "operation": action.get("operation", {}),
                        "pos_candidates": action.get("pos_candidates", [])[:5],
                        "neg_candidates": action.get("neg_candidates", [])[:20],
                    })

            tasks.append(task)

        out_path = output_dir / f"mind2web_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"  Saved {len(tasks)} tasks to {out_path}")

    # Summary
    print("\nMind2Web dataset downloaded.")
    print("Splits for generalization evaluation:")
    print("  test_task:    Unseen tasks on seen websites")
    print("  test_website: Unseen websites in seen domains")
    print("  test_domain:  Unseen domains entirely")


if __name__ == "__main__":
    main()
