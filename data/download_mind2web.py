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

    # Mind2Web has a single 'train' split. We create cross-domain splits
    # by grouping by domain and holding out domains for generalization.
    print("Downloading Mind2Web (train split)...")
    dataset = load_dataset("osunlp/Mind2Web", split="train", **kwargs)

    # Group tasks by annotation_id (each annotation = one full task)
    task_map = {}
    for i, example in enumerate(dataset):
        ann_id = example.get("annotation_id", f"unknown_{i}")
        if ann_id not in task_map:
            task_map[ann_id] = {
                "task_id": f"m2w_{ann_id[:8]}",
                "task": example.get("confirmed_task", ""),
                "website": example.get("website", ""),
                "domain": example.get("domain", ""),
                "subdomain": example.get("subdomain", ""),
                "action_reprs": example.get("action_reprs", []),
                "num_actions": len(example.get("action_reprs", [])),
                "actions": [],
            }
        # Append action steps
        actions = example.get("actions", [])
        for action in actions:
            if isinstance(action, dict):
                task_map[ann_id]["actions"].append({
                    "operation": action.get("operation", {}),
                    "pos_candidates": action.get("pos_candidates", [])[:5],
                    "neg_candidates": action.get("neg_candidates", [])[:20],
                })

    all_tasks = list(task_map.values())
    if args.max_tasks:
        all_tasks = all_tasks[:args.max_tasks]
    print(f"  Total: {len(all_tasks)} unique tasks")

    # Save full dataset as test_task (primary eval)
    out_path = output_dir / "mind2web_test_task.json"
    with open(out_path, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"  Saved {len(all_tasks)} tasks to {out_path}")

    # Create domain-based splits for cross-domain generalization
    from collections import defaultdict
    by_domain = defaultdict(list)
    for t in all_tasks:
        by_domain[t["domain"]].append(t)

    domains = sorted(by_domain.keys())
    print(f"  Domains ({len(domains)}): {domains}")

    # Hold out one domain at a time for test_domain split
    # Use the largest domain as the held-out test domain
    if len(domains) >= 2:
        # Hold out the largest domain
        held_out = max(domains, key=lambda d: len(by_domain[d]))
        test_domain_tasks = by_domain[held_out]
        out_path = output_dir / "mind2web_test_domain.json"
        with open(out_path, "w") as f:
            json.dump(test_domain_tasks, f, indent=2)
        print(f"  test_domain ({held_out}): {len(test_domain_tasks)} tasks -> {out_path}")

    # Per-domain files for analysis
    for domain, tasks in by_domain.items():
        domain_safe = domain.lower().replace(" ", "_")
        out_path = output_dir / f"mind2web_domain_{domain_safe}.json"
        with open(out_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"  domain/{domain}: {len(tasks)} tasks")

    print("\nMind2Web dataset downloaded and split by domain.")


if __name__ == "__main__":
    main()
