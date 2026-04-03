#!/usr/bin/env python3
"""
Summarize the IW Benchmark into a compact profile that can be used
to generate synthetic interaction trajectories for InteraSkill.

Produces:
  1. iw_summary.json — compact per-category summary with skill templates
  2. iw_skill_templates.json — reusable skill patterns extracted across workflows
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
INPUT = DATA_DIR / "parsed_trajectories.json"
SUMMARY_OUT = DATA_DIR / "iw_summary.json"
TEMPLATES_OUT = DATA_DIR / "iw_skill_templates.json"


def load_trajectories():
    with open(INPUT) as f:
        return json.load(f)


def categorize_action(name: str, desc: str) -> str:
    """Map a sub-task to a coarse action type for skill discovery."""
    text = (name + " " + desc).lower()
    if any(w in text for w in ["type", "write", "draft", "compose", "edit", "revise", "create", "design", "format"]):
        return "document_edit"
    if any(w in text for w in ["send", "email", "message", "reply", "forward", "notify"]):
        return "send_message"
    if any(w in text for w in ["schedule", "meeting", "calendar", "invite", "join"]):
        return "schedule_meeting"
    if any(w in text for w in ["review", "check", "verify", "approve", "feedback"]):
        return "review_content"
    if any(w in text for w in ["search", "find", "locate", "browse", "look"]):
        return "search_navigate"
    if any(w in text for w in ["copy", "paste", "move", "transfer", "attach", "upload", "download"]):
        return "data_transfer"
    if any(w in text for w in ["organize", "sort", "folder", "file", "archive", "tag", "label"]):
        return "organize_files"
    if any(w in text for w in ["slide", "presentation", "powerpoint"]):
        return "presentation_edit"
    if any(w in text for w in ["print", "export", "save", "publish"]):
        return "export_publish"
    if any(w in text for w in ["chat", "discuss", "collaborate", "brainstorm"]):
        return "collaborate"
    if any(w in text for w in ["monitor", "track", "update", "status", "follow"]):
        return "monitor_status"
    return "generic_action"


def summarize_trajectory(traj: dict) -> dict:
    """Condense a trajectory into a compact summary with ~3-7 high-level steps."""
    # Group sub-tasks by action type
    action_groups = defaultdict(list)
    for st in traj["sub_tasks"]:
        atype = categorize_action(st["name"], st["description"])
        action_groups[atype].append(st)

    # Create condensed steps (merge similar actions)
    steps = []
    for atype, tasks in sorted(action_groups.items(), key=lambda x: -sum(t["duration_minutes"] for t in x[1])):
        total_dur = sum(t["duration_minutes"] for t in tasks)
        categories = list(set(t["category"] for t in tasks))
        representative = max(tasks, key=lambda t: t["duration_minutes"])
        steps.append({
            "action_type": atype,
            "representative_task": representative["name"],
            "description": representative["description"],
            "count": len(tasks),
            "total_duration_minutes": round(total_dur, 1),
            "categories": categories,
            "apps": list(set(
                app for seg in traj["segments"]
                for app in seg["apps"]
            )),
        })

    return {
        "trajectory_id": traj["trajectory_id"],
        "objective": traj["task_description"],
        "complexity": traj["complexity"],
        "apps_involved": traj["apps_involved"],
        "total_duration_minutes": traj["total_duration_minutes"],
        "num_original_subtasks": len(traj["sub_tasks"]),
        "num_segments": len(traj["segments"]),
        "condensed_steps": steps[:7],  # cap at 7 high-level steps
        "num_condensed_steps": min(len(steps), 7),
    }


def extract_skill_templates(trajectories: list) -> list:
    """
    Extract reusable skill templates from across all trajectories.
    A skill template is a recurring action pattern that appears in multiple workflows.
    """
    # Collect all action types with their instances
    action_instances = defaultdict(list)
    for traj in trajectories:
        for st in traj["sub_tasks"]:
            atype = categorize_action(st["name"], st["description"])
            action_instances[atype].append({
                "name": st["name"],
                "description": st["description"],
                "category": st["category"],
                "duration": st["duration_minutes"],
                "apps": traj["apps_involved"],
                "complexity": traj["complexity"],
                "trajectory_id": traj["trajectory_id"],
            })

    templates = []
    for atype, instances in sorted(action_instances.items(), key=lambda x: -len(x[1])):
        durations = [inst["duration"] for inst in instances]
        categories = Counter(inst["category"] for inst in instances)
        apps = Counter(app for inst in instances for app in inst["apps"])
        complexities = Counter(inst["complexity"] for inst in instances)

        # Pick representative examples (diverse by category)
        seen_cats = set()
        examples = []
        for inst in instances:
            if inst["category"] not in seen_cats and len(examples) < 3:
                examples.append({
                    "name": inst["name"],
                    "description": inst["description"],
                    "category": inst["category"],
                })
                seen_cats.add(inst["category"])

        templates.append({
            "skill_type": atype,
            "frequency": len(instances),
            "pct_of_all_actions": round(100 * len(instances) / sum(len(v) for v in action_instances.values()), 1),
            "avg_duration_minutes": round(statistics.mean(durations), 1),
            "std_duration_minutes": round(statistics.stdev(durations), 1) if len(durations) > 1 else 0,
            "top_categories": dict(categories.most_common(3)),
            "apps_distribution": dict(apps.most_common()),
            "complexity_distribution": dict(complexities),
            "examples": examples,
            "simulated_action_sequence": _generate_action_template(atype),
        })

    return templates


def _generate_action_template(atype: str) -> list:
    """Generate a template action sequence for synthetic trajectory fabrication."""
    templates = {
        "document_edit": [
            {"action": "click", "target": "document_area", "desc": "Focus on document"},
            {"action": "select_text", "target": "content_region", "desc": "Select text to edit"},
            {"action": "type", "target": "content_region", "desc": "Type/replace content"},
            {"action": "format", "target": "toolbar", "desc": "Apply formatting"},
            {"action": "save", "target": "save_button", "desc": "Save changes"},
        ],
        "send_message": [
            {"action": "click", "target": "compose_button", "desc": "Open compose window"},
            {"action": "type", "target": "recipient_field", "desc": "Enter recipient"},
            {"action": "type", "target": "message_body", "desc": "Write message content"},
            {"action": "click", "target": "send_button", "desc": "Send message"},
        ],
        "schedule_meeting": [
            {"action": "click", "target": "calendar_button", "desc": "Open calendar"},
            {"action": "click", "target": "new_meeting", "desc": "Create new meeting"},
            {"action": "type", "target": "title_field", "desc": "Enter meeting title"},
            {"action": "type", "target": "attendee_field", "desc": "Add attendees"},
            {"action": "click", "target": "time_picker", "desc": "Set date/time"},
            {"action": "click", "target": "send_invite", "desc": "Send invitation"},
        ],
        "review_content": [
            {"action": "click", "target": "document_link", "desc": "Open document to review"},
            {"action": "scroll", "target": "document_body", "desc": "Read through content"},
            {"action": "click", "target": "comment_button", "desc": "Add review comment"},
            {"action": "type", "target": "comment_box", "desc": "Write feedback"},
            {"action": "click", "target": "submit_review", "desc": "Submit review"},
        ],
        "search_navigate": [
            {"action": "click", "target": "search_bar", "desc": "Focus search field"},
            {"action": "type", "target": "search_bar", "desc": "Enter search query"},
            {"action": "click", "target": "search_submit", "desc": "Execute search"},
            {"action": "click", "target": "result_item", "desc": "Select from results"},
        ],
        "data_transfer": [
            {"action": "click", "target": "source_element", "desc": "Select source data"},
            {"action": "copy", "target": "source_element", "desc": "Copy data"},
            {"action": "switch_app", "target": "target_app", "desc": "Switch to target"},
            {"action": "click", "target": "target_location", "desc": "Position cursor"},
            {"action": "paste", "target": "target_location", "desc": "Paste data"},
        ],
        "organize_files": [
            {"action": "click", "target": "file_item", "desc": "Select file"},
            {"action": "right_click", "target": "file_item", "desc": "Open context menu"},
            {"action": "click", "target": "move_option", "desc": "Choose move/organize"},
            {"action": "click", "target": "destination_folder", "desc": "Select destination"},
        ],
        "presentation_edit": [
            {"action": "click", "target": "slide_thumbnail", "desc": "Select slide"},
            {"action": "click", "target": "text_box", "desc": "Click text element"},
            {"action": "type", "target": "text_box", "desc": "Edit slide content"},
            {"action": "click", "target": "design_tab", "desc": "Adjust design/layout"},
            {"action": "save", "target": "save_button", "desc": "Save presentation"},
        ],
        "export_publish": [
            {"action": "click", "target": "file_menu", "desc": "Open file menu"},
            {"action": "click", "target": "export_option", "desc": "Select export/print"},
            {"action": "click", "target": "format_selector", "desc": "Choose output format"},
            {"action": "click", "target": "confirm_button", "desc": "Execute export"},
        ],
        "collaborate": [
            {"action": "click", "target": "chat_window", "desc": "Open collaboration space"},
            {"action": "type", "target": "message_input", "desc": "Send message to team"},
            {"action": "click", "target": "share_button", "desc": "Share content/file"},
            {"action": "click", "target": "reaction_button", "desc": "React/acknowledge"},
        ],
        "monitor_status": [
            {"action": "click", "target": "dashboard_tab", "desc": "Open status view"},
            {"action": "scroll", "target": "status_list", "desc": "Review items"},
            {"action": "click", "target": "status_item", "desc": "Check specific item"},
            {"action": "click", "target": "update_button", "desc": "Update status if needed"},
        ],
        "generic_action": [
            {"action": "click", "target": "ui_element", "desc": "Interact with element"},
            {"action": "type", "target": "input_field", "desc": "Enter information"},
            {"action": "click", "target": "confirm_button", "desc": "Confirm action"},
        ],
    }
    return templates.get(atype, templates["generic_action"])


def print_summary(summaries, templates):
    """Print a human-readable summary."""
    print("=" * 70)
    print("IW BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\nTotal trajectories: {len(summaries)}")
    print(f"Condensed to 3-7 steps each (from 20-60 original sub-tasks)")

    # Steps distribution
    step_counts = [s["num_condensed_steps"] for s in summaries]
    print(f"Condensed steps per trajectory: min={min(step_counts)}, max={max(step_counts)}, "
          f"mean={statistics.mean(step_counts):.1f}")

    # By complexity
    print("\nBy complexity:")
    for comp in ["low", "medium", "high"]:
        subset = [s for s in summaries if s["complexity"] == comp]
        sc = [s["num_condensed_steps"] for s in subset]
        dur = [s["total_duration_minutes"] for s in subset]
        print(f"  {comp:8s}: {len(subset)} traj, steps: {statistics.mean(sc):.1f} avg, "
              f"duration: {statistics.mean(dur):.1f} min avg")

    # Skill templates
    print(f"\n{'=' * 70}")
    print(f"SKILL TEMPLATES ({len(templates)} discovered)")
    print(f"{'=' * 70}")
    for t in templates:
        print(f"\n  {t['skill_type']}")
        print(f"    Frequency: {t['frequency']} instances ({t['pct_of_all_actions']}% of all actions)")
        print(f"    Avg duration: {t['avg_duration_minutes']} +/- {t['std_duration_minutes']} min")
        print(f"    Top categories: {t['top_categories']}")
        print(f"    Action template: {len(t['simulated_action_sequence'])} steps")
        for step in t["simulated_action_sequence"]:
            print(f"      {step['action']:12s} -> {step['target']:20s} // {step['desc']}")


def main():
    trajectories = load_trajectories()
    print(f"Loaded {len(trajectories)} trajectories")

    # 1. Summarize each trajectory
    summaries = [summarize_trajectory(t) for t in trajectories]

    # 2. Extract skill templates
    templates = extract_skill_templates(trajectories)

    # 3. Save
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} summaries to {SUMMARY_OUT}")

    with open(TEMPLATES_OUT, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"Saved {len(templates)} skill templates to {TEMPLATES_OUT}")

    # 4. Print
    print_summary(summaries, templates)


if __name__ == "__main__":
    main()
