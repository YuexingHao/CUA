"""Data loading and action featurization for InteraSkill pipeline."""

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

ACTION_TYPES = [
    "click", "copy", "format", "paste", "right_click",
    "save", "scroll", "select_text", "switch_app", "type",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
D_ACTION = len(ACTION_TYPES) + 5  # 10 one-hot + x + y + timestamp + text_len + scroll

SKILL_TYPES = [
    "collaborate", "data_transfer", "document_edit", "export_publish",
    "generic_action", "monitor_status", "organize_files",
    "presentation_edit", "review_content", "schedule_meeting",
    "search_navigate", "send_message",
]
SKILL_TO_IDX = {s: i for i, s in enumerate(SKILL_TYPES)}
N_SKILLS = len(SKILL_TYPES)

# ── Mind2Web Action Types ────────────────────────────────────────────
MIND2WEB_ACTION_TYPES = ["click", "type", "select_option"]

# ── WebShop Canonical Skill Sequences ────────────────────────────────
# Expected skill flow for typical WebShop tasks
WEBSHOP_CANONICAL_SEQUENCE = [
    "search_navigate",  # search for product
    "review_content",   # browse results
    "review_content",   # view product details
    "data_transfer",    # select options (size, color)
    "generic_action",   # add to cart / buy
]

TrajectoryData = namedtuple("TrajectoryData", [
    "traj_id",           # str
    "actions",           # Tensor (T, D_ACTION)
    "gt_skill_per_action",  # list[str] length T — skill label for each action
    "gt_boundaries",     # list[int] — action indices where new segments start
    "gt_segments",       # list[Tensor] — list of (seg_len, D_ACTION)
    "gt_segment_labels", # list[str] — one skill label per segment
    "skill_sequence",    # list[str] — ordered skill types
])


def encode_action(action: dict, max_ts: float) -> torch.Tensor:
    """Encode a single action as a D_ACTION-dim feature vector."""
    vec = torch.zeros(D_ACTION)
    # One-hot action type
    idx = ACTION_TO_IDX.get(action["action_type"], 0)
    vec[idx] = 1.0
    # Coordinates
    coords = action.get("coordinates", {})
    vec[10] = coords.get("x", 0.0)
    vec[11] = coords.get("y", 0.0)
    # Normalized timestamp
    ts = action.get("timestamp_offset_ms", 0)
    vec[12] = min(ts / max(max_ts, 1.0), 1.0)
    # Text length (normalized)
    vec[13] = min(action.get("text_length", 0) / 200.0, 1.0)
    # Scroll amount (normalized)
    vec[14] = min(action.get("scroll_amount", 0) / 500.0, 1.0)
    return vec


def load_trajectories(path: str) -> list[dict]:
    """Load raw trajectory JSON."""
    with open(path) as f:
        return json.load(f)


def featurize_trajectory(traj: dict) -> TrajectoryData:
    """Convert raw trajectory dict to featurized TrajectoryData."""
    # Find max timestamp for normalization
    all_ts = [a.get("timestamp_offset_ms", 0)
              for seg in traj["segments"] for a in seg["actions"]]
    max_ts = max(all_ts) if all_ts else 1.0

    all_actions = []
    gt_labels = []
    gt_boundaries = [0]
    gt_segments = []
    gt_segment_labels = []

    for seg in traj["segments"]:
        seg_actions = []
        for a in seg["actions"]:
            vec = encode_action(a, max_ts)
            all_actions.append(vec)
            gt_labels.append(seg["skill_type"])
            seg_actions.append(vec)
        if seg_actions:
            gt_segments.append(torch.stack(seg_actions))
            gt_segment_labels.append(seg["skill_type"])
            gt_boundaries.append(gt_boundaries[-1] + len(seg_actions))

    # Remove trailing boundary (it's just the total length)
    gt_boundaries = gt_boundaries[1:-1]  # internal boundaries only

    actions_tensor = torch.stack(all_actions) if all_actions else torch.zeros(0, D_ACTION)

    return TrajectoryData(
        traj_id=traj["trajectory_id"],
        actions=actions_tensor,
        gt_skill_per_action=gt_labels,
        gt_boundaries=gt_boundaries,
        gt_segments=gt_segments,
        gt_segment_labels=gt_segment_labels,
        skill_sequence=traj.get("skill_sequence", gt_segment_labels),
    )


def load_and_featurize(path: str) -> list[TrajectoryData]:
    """Load trajectories and featurize all."""
    trajs = load_trajectories(path)
    return [featurize_trajectory(t) for t in trajs]


def extract_all_segments(traj_data_list: list[TrajectoryData]):
    """Extract all ground-truth segments and labels across trajectories.
    Returns:
        segments: list of Tensor (each seg_len x D_ACTION)
        labels: list of str
        label_ints: Tensor (N,)
    """
    segments = []
    labels = []
    for td in traj_data_list:
        segments.extend(td.gt_segments)
        labels.extend(td.gt_segment_labels)
    label_ints = torch.tensor([SKILL_TO_IDX[l] for l in labels])
    return segments, labels, label_ints


# ── Mind2Web Helpers ─────────────────────────────────────────────────

def _extract_text_from_attrs(attrs: str) -> str:
    """Extract visible text from a Mind2Web attributes string.

    Attributes are typically a space-separated string of key=value pairs.
    We look for aria-label, title, placeholder, value, alt, or inner_text.
    """
    if not attrs:
        return ""
    import re
    # Try common text-bearing attributes
    for attr_name in ["aria-label", "title", "placeholder", "value",
                      "alt", "inner_text", "text"]:
        match = re.search(
            rf'{attr_name}\s*=\s*["\']([^"\']+)["\']', attrs)
        if match:
            return match.group(1)
    # Fallback: return first non-empty quoted string
    match = re.search(r'["\']([^"\']{2,50})["\']', attrs)
    return match.group(1) if match else ""


def _candidate_to_text(candidate: dict) -> str:
    """Convert a Mind2Web candidate element to display text."""
    if not isinstance(candidate, dict):
        return str(candidate)[:100]
    tag = candidate.get("tag", "div")
    attrs = candidate.get("attributes", "")
    text = _extract_text_from_attrs(attrs)
    node_id = candidate.get("backend_node_id", "")
    return f"<{tag}> {text}" if text else f"<{tag}> [node:{node_id}]"


# ── Mind2Web Data Loading ────────────────────────────────────────────

def load_mind2web(split: str = "test", max_tasks: int = None,
                  cache_dir: str = None) -> list[dict]:
    """Load Mind2Web dataset from HuggingFace.

    Args:
        split: "train", "test", or "test_task", "test_website", "test_domain"
        max_tasks: Maximum number of tasks to load
        cache_dir: HuggingFace cache directory

    Returns:
        List of task dicts with keys:
            task, website, domain, subdomain, action_reprs, actions
    """
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    dataset = load_dataset("osunlp/Mind2Web", split=split, **kwargs)

    tasks = []
    for i, example in enumerate(dataset):
        if max_tasks and i >= max_tasks:
            break
        tasks.append({
            "task_id": f"m2w_{split}_{i}",
            "task": example.get("confirmed_task", example.get("task", "")),
            "website": example.get("website", ""),
            "domain": example.get("domain", ""),
            "subdomain": example.get("subdomain", ""),
            "action_reprs": example.get("action_reprs", []),
            "actions": example.get("actions", []),
        })
    return tasks


def mind2web_task_to_steps(task: dict) -> list[dict]:
    """Convert a Mind2Web task to a list of evaluation steps.

    Each step has: action_type, element_text, element_tag, value,
                   candidates (top-k elements), ground_truth_element_id
    """
    steps = []
    actions = task.get("actions", [])
    action_reprs = task.get("action_reprs", [])

    for i, action in enumerate(actions):
        step = {
            "step_idx": i,
            "action_repr": action_reprs[i] if i < len(action_reprs) else "",
        }

        # Extract structured action info
        if isinstance(action, dict):
            step["action_type"] = action.get("operation", {}).get(
                "op", "click").lower()
            step["value"] = action.get("operation", {}).get("value", "")

            # Mind2Web candidates have: tag, backend_node_id, attributes
            # Extract text from attributes string or tag
            pos_cands = action.get("pos_candidates", [])
            if pos_cands:
                first = pos_cands[0] if isinstance(pos_cands[0], dict) else {}
                attrs = first.get("attributes", "")
                # Extract visible text from attributes (e.g. aria-label, title)
                step["element_text"] = _extract_text_from_attrs(attrs)
                step["element_tag"] = first.get("tag", "")
            else:
                step["element_text"] = ""
                step["element_tag"] = ""

            step["candidates"] = action.get("pos_candidates", []) + \
                                 action.get("neg_candidates", [])
            step["ground_truth_idx"] = 0  # pos_candidates[0] is the target
        else:
            # Fallback for string-format actions
            step["action_type"] = "click"
            step["value"] = ""
            step["element_text"] = str(action)
            step["element_tag"] = ""
            step["candidates"] = []
            step["ground_truth_idx"] = 0

        steps.append(step)
    return steps
