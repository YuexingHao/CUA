#!/usr/bin/env python3
"""Generate InteraSkill conversations from WorkArena-style trajectories.

This script expects a JSON file of WorkArena trajectories, already collected
from BrowserGym/WorkArena or another rollout logger. It converts each trajectory
into the same multi-turn conversation format used by IW/WebArena/BrowseComp
offline skill evaluation.

Expected input schema is intentionally permissive. Each item should contain a
task/objective field and either:
  - skill_sequence: list[str], or
  - segments: list[{"skill_type": str, ...}], or
  - steps/actions: list of primitive actions from which skills are inferred.

Usage:
    python scripts/data/generate_workarena_conversations.py \
        --input data/workarena/trajectories/workarena_trajectories.json \
        --output data/workarena/conversations/workarena_conversations.jsonl \
        --max-trajs 200
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from interaskill.paths import WORKARENA_CONVERSATIONS, WORKARENA_TRAJECTORIES, ensure_parent

DEFAULT_INPUT = WORKARENA_TRAJECTORIES
DEFAULT_OUTPUT = WORKARENA_CONVERSATIONS

SKILLS = [
    "document_edit",
    "send_message",
    "search_navigate",
    "review_content",
    "collaborate",
    "data_transfer",
    "export_publish",
    "organize_files",
    "monitor_status",
    "generic_action",
]

SYSTEM_PROMPT = """\
You are InteraSkill, an AI computer-using agent that helps users complete \
tasks in enterprise web applications such as ServiceNow.

You can execute the following skills:
- document_edit: Create, edit, or update records, forms, or text fields
- send_message: Compose and send messages, notes, or comments
- search_navigate: Search for records or navigate to pages
- review_content: Read, inspect, compare, or verify page content
- collaborate: Assign, share, comment, or coordinate with users
- data_transfer: Copy, move, import, export, or reuse data
- export_publish: Export, download, publish, or save outputs
- organize_files: Sort, filter, group, tag, or organize records
- monitor_status: Check dashboards, tickets, incidents, or status fields
- generic_action: Other UI interactions

When executing a task, choose the next skill and report the action."""

SKILL_KEYWORDS = {
    "search_navigate": [
        "search", "find", "open", "navigate", "go to", "look up", "lookup",
        "filter", "query",
    ],
    "review_content": [
        "review", "read", "check", "verify", "inspect", "compare", "view",
        "identify", "determine", "answer",
    ],
    "document_edit": [
        "create", "edit", "update", "change", "fill", "set", "modify",
        "submit", "add", "remove", "close", "resolve",
    ],
    "monitor_status": [
        "status", "incident", "ticket", "request", "approval", "dashboard",
        "priority", "state", "progress",
    ],
    "organize_files": [
        "sort", "group", "categorize", "tag", "list", "table", "column",
    ],
    "collaborate": [
        "assign", "comment", "share", "user", "group", "team", "watchlist",
    ],
    "send_message": [
        "message", "email", "notify", "note", "reply",
    ],
    "data_transfer": [
        "copy", "paste", "import", "transfer", "duplicate",
    ],
    "export_publish": [
        "export", "download", "publish", "save", "report",
    ],
}

ACTION_TO_SKILL = {
    "click": "search_navigate",
    "tap": "search_navigate",
    "goto": "search_navigate",
    "navigate": "search_navigate",
    "fill": "document_edit",
    "type": "document_edit",
    "select": "document_edit",
    "press": "generic_action",
    "scroll": "review_content",
    "read": "review_content",
    "wait": "monitor_status",
}

USER_TEMPLATES = {
    "search_navigate": [
        "Open the relevant record.",
        "Find the right page.",
        "Search for the item we need.",
        "Navigate to that section.",
    ],
    "review_content": [
        "Check the details.",
        "Read what is shown.",
        "Verify the information.",
        "Review the current record.",
    ],
    "document_edit": [
        "Update the field.",
        "Fill in the required value.",
        "Make the requested change.",
        "Submit the form.",
    ],
    "monitor_status": [
        "Check the current status.",
        "Look at the ticket state.",
        "Review the dashboard status.",
    ],
    "organize_files": [
        "Filter the list.",
        "Sort the records.",
        "Group the items.",
    ],
    "collaborate": [
        "Assign this to the right person.",
        "Add a comment for the team.",
        "Share this with the group.",
    ],
    "send_message": [
        "Send the note.",
        "Notify the user.",
    ],
    "data_transfer": [
        "Copy the needed value.",
        "Move this information over.",
    ],
    "export_publish": [
        "Export the result.",
        "Download the report.",
    ],
    "generic_action": [
        "Continue with the next step.",
        "Use the appropriate UI control.",
    ],
}

OBSERVATIONS = {
    "search_navigate": "The relevant page or record is now open.",
    "review_content": "The requested information is visible.",
    "document_edit": "The form or record has been updated.",
    "monitor_status": "The status information is visible.",
    "organize_files": "The list view has been updated.",
    "collaborate": "The collaboration field or comment has been updated.",
    "send_message": "The message or note has been sent.",
    "data_transfer": "The data has been copied or transferred.",
    "export_publish": "The output has been exported or saved.",
    "generic_action": "The interface has updated.",
}


def _text(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, sort_keys=True)


def _normalize_skill(skill: str) -> str:
    skill = (skill or "").lower().replace("-", "_").replace(" ", "_")
    return skill if skill in SKILLS else "generic_action"


def infer_skill(text: str, action_type: str = "") -> str:
    text_l = text.lower()
    scores = {}
    for skill, keywords in SKILL_KEYWORDS.items():
        scores[skill] = sum(1 for kw in keywords if kw in text_l)
    if scores and max(scores.values()) > 0:
        return max(scores, key=scores.get)
    action_l = (action_type or "").lower()
    for key, skill in ACTION_TO_SKILL.items():
        if key in action_l:
            return skill
    return "generic_action"


def extract_objective(traj: dict) -> str:
    for key in ("objective", "task", "goal", "instruction", "intent"):
        value = traj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Complete the WorkArena task."


def extract_steps(traj: dict) -> list[dict]:
    for key in ("steps", "actions", "trajectory", "events"):
        value = traj.get(key)
        if isinstance(value, list):
            return value
    return []


def extract_skill_flow(traj: dict) -> list[str]:
    if isinstance(traj.get("skill_sequence"), list):
        return [_normalize_skill(s) for s in traj["skill_sequence"] if s]

    if isinstance(traj.get("segments"), list):
        skills = []
        for seg in traj["segments"]:
            if isinstance(seg, dict):
                skills.append(_normalize_skill(seg.get("skill_type", "")))
        if skills:
            return skills

    objective = extract_objective(traj)
    skills = []
    for step in extract_steps(traj):
        if not isinstance(step, dict):
            skills.append(infer_skill(_text(step)))
            continue
        action_type = (
            step.get("action_type")
            or step.get("action")
            or step.get("type")
            or step.get("operation")
            or ""
        )
        text = " ".join(
            _text(step.get(k))
            for k in ("thought", "description", "observation", "target", "text")
        )
        skills.append(infer_skill(objective + " " + text, _text(action_type)))

    if not skills:
        skills = [infer_skill(objective)]

    # Merge exact adjacent duplicates. Skill composition cares about changes in
    # subtask type, not repeated clicks inside the same subtask.
    merged = []
    for skill in skills:
        skill = _normalize_skill(skill)
        if not merged or merged[-1] != skill:
            merged.append(skill)
    return merged or ["generic_action"]


def generate_agent_turn(skill: str, is_first: bool, objective: str,
                        skill_flow: list[str]) -> str:
    if is_first:
        plan = ", ".join(skill_flow[:5])
        thinking = f"Plan: {plan}."
    else:
        thinking = f"Next skill: {skill}."
    return (
        f"**[Thinking]** {thinking}\n\n"
        f"**[Action: {skill}]** Execute the {skill} step.\n\n"
        f"**[Observation]** {OBSERVATIONS.get(skill, OBSERVATIONS['generic_action'])}"
    )


def generate_conversation(traj: dict, conv_id: str) -> dict:
    objective = extract_objective(traj)
    skill_flow = extract_skill_flow(traj)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": objective})
    messages.append({
        "role": "assistant",
        "content": generate_agent_turn(skill_flow[0], True, objective, skill_flow),
    })

    for skill in skill_flow[1:]:
        user_msg = random.choice(USER_TEMPLATES.get(skill, USER_TEMPLATES["generic_action"]))
        messages.append({"role": "user", "content": user_msg})
        messages.append({
            "role": "assistant",
            "content": generate_agent_turn(skill, False, objective, skill_flow),
        })

    messages.append({"role": "user", "content": "That's all."})
    messages.append({
        "role": "assistant",
        "content": "Done. Skills used: " + ", ".join(skill_flow) + ".",
    })

    return {
        "conversation_id": conv_id,
        "task": objective,
        "complexity": traj.get("complexity", "medium"),
        "apps": traj.get("apps_involved", ["servicenow"]),
        "skill_flow": skill_flow,
        "num_turns": len(messages),
        "domain": traj.get("domain", "enterprise"),
        "source": "workarena",
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate WorkArena conversations")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-trajs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Collect WorkArena rollouts first, then rerun.\n"
            "Expected a JSON list of trajectories with task/objective and "
            "steps/actions or skill_sequence fields."
        )

    with open(args.input) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("trajectories", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of WorkArena trajectories")

    random.shuffle(data)
    data = data[:args.max_trajs]
    conversations = [
        generate_conversation(traj, f"workarena_conv_{i:05d}")
        for i, traj in enumerate(data)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(ensure_parent(args.output), "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    skill_counts = Counter()
    lengths = []
    for conv in conversations:
        lengths.append(len(conv["skill_flow"]))
        skill_counts.update(conv["skill_flow"])

    print(f"Saved {len(conversations)} conversations to {args.output}")
    if conversations:
        print(
            "Skill sequence length: "
            f"min={min(lengths)}, max={max(lengths)}, "
            f"mean={sum(lengths)/len(lengths):.1f}"
        )
        print("Skill distribution:")
        total = sum(skill_counts.values())
        for skill, count in skill_counts.most_common():
            print(f"  {skill:20s}: {count:5d} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()
