#!/usr/bin/env python3
"""
Parse the IW Benchmark JSON into trajectory format for the InteraSkill
skill discovery pipeline.

Input:  data/iw-benchmark-examples.json
Output: data/parsed_trajectories.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_PATH = Path(__file__).parent / "iw-benchmark-examples.json"
OUTPUT_PATH = Path(__file__).parent / "parsed_trajectories.json"

APP_NAME_MAP = {
    "word": "word",
    "ppt": "powerpoint",
    "teams": "teams",
    "outlook": "outlook",
    "powerpoint": "powerpoint",
}

# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    """Remove all HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", html)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_objective(html: str) -> str:
    """Pull the objective text from a context HTML block."""
    m = re.search(
        r'<div class="context-objective">.*?<strong>Objective:</strong>'
        r"(?:<br>|<br/>|<br />)?\s*(.*?)</div>",
        html,
        re.DOTALL,
    )
    if m:
        return strip_html(m.group(1)).strip()
    return ""


# Pattern for sub-tasks:  bullet char, task name [duration], description (CATEGORY)
_SUBTASK_RE = re.compile(
    r"[•\u2022]\s*"                      # bullet
    r"(?P<name>[^\[]+?)"                 # task name (up to '[')
    r"\[(?P<dur>[\d.]+)\s*min\]"         # duration in brackets
    r"(?:<br>|<br/>|<br />)?\s*"         # optional <br>
    r"(?P<desc>.*?)"                     # description text
    r"\((?P<cat>[A-Z][A-Z &]+[A-Z])\)", # category in parens
    re.DOTALL,
)


def extract_subtasks(html: str) -> list[dict]:
    """Parse sub-task items from a context HTML block."""
    subtasks = []
    for m in _SUBTASK_RE.finditer(html):
        desc_raw = m.group("desc")
        # Strip any remaining HTML spans / tags from desc
        desc_clean = strip_html(desc_raw).strip()
        subtasks.append(
            {
                "name": m.group("name").strip(),
                "duration_minutes": float(m.group("dur")),
                "category": m.group("cat").strip(),
                "description": desc_clean,
            }
        )
    return subtasks


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------

# Complexity is always the last token after the final underscore
_COMPLEXITY_VALUES = {"low", "medium", "high"}


def parse_category_key(key: str) -> tuple[list[str], str]:
    """
    Parse a benchmark dict key like 'word_ppt_medium' into
    (apps_involved, complexity).
    """
    parts = key.split("_")
    complexity = parts[-1] if parts[-1] in _COMPLEXITY_VALUES else "unknown"
    app_tokens = parts[:-1] if complexity != "unknown" else parts

    # Handle 'all_four' as a special token
    if "all" in app_tokens and "four" in app_tokens:
        return ["word", "powerpoint", "teams", "outlook"], complexity

    apps = []
    for tok in app_tokens:
        mapped = APP_NAME_MAP.get(tok)
        if mapped and mapped not in apps:
            apps.append(mapped)
        elif tok == "only":
            continue  # skip filler word
        else:
            # Unknown token -- keep as-is if it looks like an app name
            if tok not in ("only",) and tok not in _COMPLEXITY_VALUES:
                # Could be a multi-word combo we haven't mapped; skip quietly
                pass
    return apps, complexity


# ---------------------------------------------------------------------------
# Activity -> Segment mapping
# ---------------------------------------------------------------------------

def apps_from_involvement(app_inv: dict) -> list[str]:
    """Return list of app names that are True in the involvement dict."""
    name_map = {
        "word": "word",
        "powerpoint": "powerpoint",
        "teams": "teams",
        "outlook": "outlook",
    }
    return [name_map[k] for k, v in app_inv.items() if v and k in name_map]


def activity_to_segment(activity: dict, traj_id: str) -> dict:
    """Convert a workflow_data activity to a trajectory segment."""
    pos = activity.get("position", 0)
    apps = apps_from_involvement(activity.get("app_involvement", {}))
    return {
        "segment_id": f"{traj_id}_seg{pos}",
        "name": activity.get("name", ""),
        "description": activity.get("description", ""),
        "duration_minutes": activity.get("duration_minutes", 0.0),
        "apps": apps,
        "position": pos,
        "actions": [],  # will be populated from HTML sub-tasks
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def build_trajectories(data: dict) -> list[dict]:
    """Convert the raw benchmark dict into a list of trajectory dicts."""
    trajectories = []
    global_idx = 0

    for key, entries in data.items():
        apps_involved, complexity = parse_category_key(key)

        for entry in entries:
            wf_idx = entry.get("workflow_index", global_idx)
            traj_id = f"{key}_wf{wf_idx}"

            # --- Extract objective + sub-tasks from without_context HTML ---
            without_ctx = entry.get("without_context", [])
            objective = ""
            all_subtasks: list[dict] = []
            parsed_actions_per_section: list[list[dict]] = []

            for html_block in without_ctx:
                if not objective:
                    obj = extract_objective(html_block)
                    if obj:
                        objective = obj
                subs = extract_subtasks(html_block)
                all_subtasks.extend(subs)
                parsed_actions_per_section.append(subs)

            # If no objective from without_context, try with_context
            if not objective:
                for html_block in entry.get("with_context", []):
                    obj = extract_objective(html_block)
                    if obj:
                        objective = obj
                        break

            # If still no sub-tasks from without_context, try with_context
            if not all_subtasks:
                for html_block in entry.get("with_context", []):
                    subs = extract_subtasks(html_block)
                    all_subtasks.extend(subs)
                    parsed_actions_per_section.append(subs)

            # --- Build segments from workflow_data activities ---
            wf_data = entry.get("workflow_data", {})
            activities = wf_data.get("activities", [])
            total_dur = wf_data.get("total_duration", 0.0)

            segments = []
            for act in activities:
                seg = activity_to_segment(act, traj_id)
                segments.append(seg)

            # Attach parsed sub-task actions to the first without_context
            # block's sub-tasks as the "actions" of the first segment, etc.
            # Strategy: distribute sub-tasks across segments proportionally
            # by matching position order.
            if segments and all_subtasks:
                # Simple heuristic: split sub-tasks evenly among segments
                n_seg = len(segments)
                n_sub = len(all_subtasks)
                chunk = max(1, n_sub // n_seg)
                for i, seg in enumerate(segments):
                    start = i * chunk
                    end = start + chunk if i < n_seg - 1 else n_sub
                    seg["actions"] = all_subtasks[start:end]

            trajectory = {
                "trajectory_id": traj_id,
                "task_description": objective,
                "complexity": complexity,
                "apps_involved": apps_involved,
                "total_duration_minutes": total_dur,
                "segments": segments,
                "sub_tasks": all_subtasks,
            }
            trajectories.append(trajectory)
            global_idx += 1

    return trajectories


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(trajectories: list[dict]) -> None:
    """Print human-readable summary of the parsed trajectories."""
    n = len(trajectories)
    print(f"Total trajectories: {n}")
    print()

    # By complexity
    complexity_counts = Counter(t["complexity"] for t in trajectories)
    print("By complexity:")
    for level in ("low", "medium", "high"):
        print(f"  {level:>8s}: {complexity_counts.get(level, 0)}")
    for level, cnt in complexity_counts.items():
        if level not in ("low", "medium", "high"):
            print(f"  {level:>8s}: {cnt}")
    print()

    # By app combination
    app_combo_counts = Counter(
        tuple(sorted(t["apps_involved"])) for t in trajectories
    )
    print("By app combination:")
    for combo, cnt in sorted(app_combo_counts.items(), key=lambda x: -x[1]):
        label = ", ".join(combo) if combo else "(none)"
        print(f"  {label}: {cnt}")
    print()

    # Sub-task category breakdown
    cat_counts: Counter = Counter()
    for t in trajectories:
        for st in t["sub_tasks"]:
            cat_counts[st.get("category", "UNKNOWN")] += 1
    print("Sub-task categories:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")
    print()

    # Duration stats
    durations = [t["total_duration_minutes"] for t in trajectories]
    if durations:
        print(
            f"Duration (min): min={min(durations):.1f}, "
            f"max={max(durations):.1f}, "
            f"mean={sum(durations)/len(durations):.1f}"
        )

    # Segments stats
    seg_counts = [len(t["segments"]) for t in trajectories]
    if seg_counts:
        print(
            f"Segments per trajectory: min={min(seg_counts)}, "
            f"max={max(seg_counts)}, "
            f"mean={sum(seg_counts)/len(seg_counts):.1f}"
        )

    subtask_counts = [len(t["sub_tasks"]) for t in trajectories]
    if subtask_counts:
        print(
            f"Sub-tasks per trajectory: min={min(subtask_counts)}, "
            f"max={max(subtask_counts)}, "
            f"mean={sum(subtask_counts)/len(subtask_counts):.1f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} benchmark categories from {INPUT_PATH}")

    trajectories = build_trajectories(raw_data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(trajectories)} trajectories to {OUTPUT_PATH}")
    print("=" * 60)
    print_summary(trajectories)


if __name__ == "__main__":
    main()
