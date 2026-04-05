#!/usr/bin/env python3
"""
Download and convert Go-Browse-WA (real WebArena trajectories)
into InteraSkill pipeline format.

Source: https://huggingface.co/datasets/apurvaga/go-browse-wa
Contains ~27K real agent trajectories on WebArena sites.

Each trajectory has steps with:
  - Observation: accessibility tree of the web page
  - Action: JSON with thought (reasoning) + action (click, fill, scroll, etc.)
  - Reward: 0 (fail) or 1 (success)

We convert these into our segment-based format for the InteraSkill pipeline.

Usage:
    python data/download_webarena.py [--max-trajs 1000] [--success-only]
"""

import argparse
import json
import re
import random
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "webarena_trajectories.json"

# ── Action type mapping ──────────────────────────────────────────────
# Map WebArena primitive actions to our skill vocabulary

ACTION_PATTERNS = {
    r"click\(": "click",
    r"fill\(": "type",
    r"type\(": "type",
    r"scroll\(": "scroll",
    r"goto\(": "navigate",
    r"go_back\(": "navigate",
    r"go_forward\(": "navigate",
    r"send_msg_to_user\(": "communicate",
    r"select_option\(": "select",
    r"hover\(": "click",
    r"press\(": "type",
    r"tab_focus\(": "navigate",
    r"new_tab\(": "navigate",
    r"tab_close\(": "navigate",
    r"close\(": "navigate",
}

# Map inferred high-level skills from action patterns + page context
SKILL_KEYWORDS = {
    "search_navigate": [
        "search", "find", "look for", "navigate", "go to", "open",
        "browse", "locate", "discover",
    ],
    "document_edit": [
        "edit", "modify", "update", "change", "write", "create document",
        "add text", "insert", "delete text", "rename",
    ],
    "send_message": [
        "send", "email", "message", "reply", "compose", "post comment",
        "submit comment", "write comment",
    ],
    "review_content": [
        "review", "check", "verify", "read", "view", "inspect",
        "look at", "examine", "compare",
    ],
    "data_transfer": [
        "copy", "paste", "transfer", "move data", "import", "export data",
        "download", "upload",
    ],
    "organize_files": [
        "organize", "sort", "filter", "arrange", "move file", "create folder",
        "archive", "categorize", "tag",
    ],
    "schedule_meeting": [
        "schedule", "calendar", "event", "appointment", "meeting",
        "set date", "set time",
    ],
    "collaborate": [
        "collaborate", "share", "assign", "team", "merge request",
        "pull request", "fork",
    ],
    "export_publish": [
        "export", "print", "publish", "save as", "convert",
    ],
    "monitor_status": [
        "status", "monitor", "dashboard", "notification", "activity",
        "check progress",
    ],
}

# WebArena site → domain mapping
SITE_DOMAINS = {
    "shopping": "e-commerce",
    "reddit": "social-forum",
    "gitlab": "dev-platform",
    "cms": "content-management",
    "map": "maps",
    "wikipedia": "encyclopedia",
}


def parse_action(completion: list) -> dict:
    """Parse the agent's action from the completion field."""
    if not completion or not isinstance(completion, list):
        return {"action_type": "unknown", "thought": "", "raw": ""}

    content = completion[0].get("content", "")

    try:
        parsed = json.loads(content)
        thought = parsed.get("thought", "")
        action_str = parsed.get("action", "")
    except (json.JSONDecodeError, TypeError):
        thought = ""
        action_str = content

    # Determine action type
    action_type = "unknown"
    for pattern, atype in ACTION_PATTERNS.items():
        if re.search(pattern, action_str):
            action_type = atype
            break

    # Extract target element ID if present
    target_match = re.search(r"['\"](\d+)['\"]", action_str)
    target_id = target_match.group(1) if target_match else ""

    # Extract text content if fill/type
    text_match = re.search(r",\s*['\"](.+?)['\"]", action_str)
    text_content = text_match.group(1) if text_match else ""

    return {
        "action_type": action_type,
        "thought": thought,
        "raw_action": action_str,
        "target_id": target_id,
        "text_content": text_content,
    }


def infer_skill_type(goal: str, actions: list[dict]) -> str:
    """Infer the high-level skill type from the goal and actions."""
    goal_lower = goal.lower()

    # Score each skill by keyword matches
    scores = {}
    for skill, keywords in SKILL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in goal_lower)
        # Also check action thoughts
        for a in actions:
            thought_lower = a.get("thought", "").lower()
            score += sum(0.5 for kw in keywords if kw in thought_lower)
        scores[skill] = score

    best_skill = max(scores, key=scores.get)
    if scores[best_skill] == 0:
        # Fallback: infer from dominant action types
        action_types = [a["action_type"] for a in actions]
        type_counts = Counter(action_types)
        dominant = type_counts.most_common(1)[0][0]
        fallback_map = {
            "click": "search_navigate",
            "type": "document_edit",
            "scroll": "review_content",
            "navigate": "search_navigate",
            "communicate": "send_message",
            "select": "search_navigate",
        }
        best_skill = fallback_map.get(dominant, "generic_action")

    return best_skill


def detect_site(goal: str, page_content: str) -> str:
    """Detect which WebArena site this trajectory is on."""
    combined = (goal + " " + page_content[:500]).lower()

    if "shopping" in combined or "product" in combined or "cart" in combined:
        return "shopping"
    elif "reddit" in combined or "subreddit" in combined or "upvote" in combined:
        return "reddit"
    elif "gitlab" in combined or "repository" in combined or "merge" in combined:
        return "gitlab"
    elif "cms" in combined or "admin" in combined or "wordpress" in combined:
        return "cms"
    elif "map" in combined or "openstreetmap" in combined or "direction" in combined:
        return "map"
    elif "wikipedia" in combined or "wiki" in combined or "article" in combined:
        return "wikipedia"
    return "unknown"


def group_steps_into_trajectory(steps: list[dict]) -> dict:
    """Group consecutive steps (same trajectory) into a single trajectory dict."""
    if not steps:
        return None

    first = steps[0]
    goal_text = first["step_data"]["prompt"][1]["content"]
    goal_line = goal_text.split("\n")[1] if "\n" in goal_text else goal_text[:200]

    # Parse all actions
    actions = []
    for step in steps:
        action = parse_action(step["step_data"]["completion"])
        action["step_idx"] = step["step_number"]
        action["timestamp_offset_ms"] = step["step_number"] * 2000  # synthetic timing
        actions.append(action)

    # Detect site and infer skill
    page_content = first["step_data"]["prompt"][1]["content"]
    site = detect_site(goal_line, page_content)
    skill_type = infer_skill_type(goal_line, actions)

    # Segment the trajectory by action type changes
    segments = []
    current_seg_actions = [actions[0]]

    for i in range(1, len(actions)):
        # New segment if action type changes significantly
        if actions[i]["action_type"] != actions[i-1]["action_type"]:
            segments.append({
                "segment_id": f"seg_{len(segments)}",
                "skill_type": infer_skill_type(goal_line, current_seg_actions),
                "position": len(segments) + 1,
                "num_actions": len(current_seg_actions),
                "actions": [{
                    "action_type": a["action_type"],
                    "target": a.get("target_id", ""),
                    "coordinates": {"x": random.uniform(0.1, 0.9),
                                    "y": random.uniform(0.1, 0.9)},
                    "description": a.get("thought", "")[:100],
                    "success": True,
                    "timestamp_offset_ms": a["timestamp_offset_ms"],
                    "text_length": len(a.get("text_content", "")),
                    "scroll_amount": 0,
                } for a in current_seg_actions],
            })
            current_seg_actions = [actions[i]]
        else:
            current_seg_actions.append(actions[i])

    # Don't forget the last segment
    if current_seg_actions:
        segments.append({
            "segment_id": f"seg_{len(segments)}",
            "skill_type": infer_skill_type(goal_line, current_seg_actions),
            "position": len(segments) + 1,
            "num_actions": len(current_seg_actions),
            "actions": [{
                "action_type": a["action_type"],
                "target": a.get("target_id", ""),
                "coordinates": {"x": random.uniform(0.1, 0.9),
                                "y": random.uniform(0.1, 0.9)},
                "description": a.get("thought", "")[:100],
                "success": True,
                "timestamp_offset_ms": a["timestamp_offset_ms"],
                "text_length": len(a.get("text_content", "")),
                "scroll_amount": 0,
            } for a in current_seg_actions],
        })

    return {
        "trajectory_id": f"wa_{site}_{len(segments):03d}",
        "source": "webarena",
        "seed_data": "go-browse-wa",
        "objective": goal_line.replace("# Goal\n", "").strip(),
        "complexity": "medium" if len(actions) <= 5 else "high",
        "site": site,
        "domain": SITE_DOMAINS.get(site, "unknown"),
        "apps_involved": [site],
        "total_duration_minutes": round(len(actions) * 0.5, 1),
        "num_segments": len(segments),
        "num_primitive_actions": len(actions),
        "reward": first["traj_reward"],
        "segments": segments,
        "skill_sequence": [s["skill_type"] for s in segments],
    }


def main():
    parser = argparse.ArgumentParser(description="Download WebArena trajectories")
    parser.add_argument("--max-trajs", type=int, default=1000,
                        help="Maximum trajectories to download")
    parser.add_argument("--success-only", action="store_true",
                        help="Only include successful trajectories")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading Go-Browse-WA dataset from HuggingFace...")
    print("(This streams data — no full download needed)")

    ds = load_dataset("apurvaga/go-browse-wa", split="train", streaming=True)

    # Group steps by trajectory (consecutive steps with same goal)
    trajectories = []
    current_traj_steps = []
    traj_count = 0

    for step in ds:
        step_num = step["step_number"]

        if step_num == 0 and current_traj_steps:
            # Process previous trajectory
            if not args.success_only or current_traj_steps[0]["traj_reward"] == 1.0:
                traj = group_steps_into_trajectory(current_traj_steps)
                if traj and traj["num_segments"] >= 2:
                    traj["trajectory_id"] = f"wa_{traj['site']}_{traj_count:04d}"
                    trajectories.append(traj)
                    traj_count += 1

            current_traj_steps = []

            if traj_count >= args.max_trajs:
                break

        current_traj_steps.append(step)

        if traj_count % 100 == 0 and step_num == 0 and traj_count > 0:
            print(f"  Processed {traj_count} trajectories...", flush=True)

    # Process last trajectory
    if current_traj_steps and traj_count < args.max_trajs:
        if not args.success_only or current_traj_steps[0]["traj_reward"] == 1.0:
            traj = group_steps_into_trajectory(current_traj_steps)
            if traj and traj["num_segments"] >= 2:
                traj["trajectory_id"] = f"wa_{traj['site']}_{traj_count:04d}"
                trajectories.append(traj)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nSaved {len(trajectories)} trajectories to {OUTPUT_FILE}")

    # Statistics
    print(f"\n{'='*60}")
    print("WEBARENA TRAJECTORY STATISTICS")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(trajectories)}")

    site_counts = Counter(t["site"] for t in trajectories)
    print(f"\nBy site:")
    for site, count in site_counts.most_common():
        print(f"  {site:15s}: {count}")

    reward_counts = Counter(t["reward"] for t in trajectories)
    print(f"\nBy reward:")
    for r, count in reward_counts.most_common():
        print(f"  {r}: {count}")

    n_actions = [t["num_primitive_actions"] for t in trajectories]
    n_segs = [t["num_segments"] for t in trajectories]
    print(f"\nActions/trajectory: min={min(n_actions)}, max={max(n_actions)}, "
          f"mean={sum(n_actions)/len(n_actions):.1f}")
    print(f"Segments/trajectory: min={min(n_segs)}, max={max(n_segs)}, "
          f"mean={sum(n_segs)/len(n_segs):.1f}")

    skill_counts = Counter()
    for t in trajectories:
        for s in t["skill_sequence"]:
            skill_counts[s] += 1
    print(f"\nSkill distribution:")
    for skill, count in skill_counts.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/sum(skill_counts.values()):.1f}%)")

    # Show example
    print(f"\n{'='*60}")
    print("EXAMPLE TRAJECTORY")
    print(f"{'='*60}")
    ex = trajectories[0]
    print(f"ID: {ex['trajectory_id']}")
    print(f"Site: {ex['site']}")
    print(f"Objective: {ex['objective'][:100]}")
    print(f"Skills: {' → '.join(ex['skill_sequence'])}")
    print(f"Actions: {ex['num_primitive_actions']}, Segments: {ex['num_segments']}")
    print(f"Reward: {ex['reward']}")


if __name__ == "__main__":
    main()
