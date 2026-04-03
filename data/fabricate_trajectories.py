#!/usr/bin/env python3
"""
Generate synthetic interaction trajectories based on IW Benchmark patterns.

Uses the skill templates and summary profiles from summarize_iw_benchmark.py
to fabricate realistic CUA trajectories for training the InteraSkill pipeline.

Usage:
    python fabricate_trajectories.py [--num 500] [--seed 42]

Produces:
    fabricated_trajectories.json — synthetic trajectories in InteraSkill format
"""

import argparse
import json
import math
import random
import uuid
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent
SUMMARY_FILE = DATA_DIR / "iw_summary.json"
TEMPLATES_FILE = DATA_DIR / "iw_skill_templates.json"
OUTPUT_FILE = DATA_DIR / "fabricated_trajectories.json"

# ── Vocabulary for realistic variation ──────────────────────────────────

OBJECTIVES = {
    "word": [
        "Draft quarterly report for leadership review",
        "Create employee onboarding documentation",
        "Write product specification document",
        "Prepare meeting minutes and action items",
        "Compose client proposal with pricing details",
        "Draft internal policy update memorandum",
        "Write technical documentation for new feature",
        "Create training manual for new software system",
    ],
    "powerpoint": [
        "Design pitch deck for investor meeting",
        "Create training presentation for new hires",
        "Build project status update slides",
        "Design product launch presentation",
        "Prepare conference talk slides",
        "Create department strategy overview",
        "Build data visualization dashboard deck",
        "Design customer case study presentation",
    ],
    "teams": [
        "Coordinate sprint planning with engineering team",
        "Run weekly standup and capture action items",
        "Organize project kickoff meeting with stakeholders",
        "Facilitate design review session",
        "Conduct interview debrief with hiring panel",
        "Coordinate cross-team feature handoff",
        "Run retrospective and document improvements",
        "Host vendor evaluation discussion",
    ],
    "outlook": [
        "Process and respond to client emails",
        "Organize inbox and set up email filters",
        "Send project status updates to stakeholders",
        "Schedule follow-up meetings from email threads",
        "Forward and categorize vendor proposals",
        "Manage subscription and notification emails",
        "Draft and send approval request chain",
        "Coordinate travel arrangements via email",
    ],
}

MULTI_APP_OBJECTIVES = [
    "Prepare and distribute quarterly business review materials",
    "Coordinate product launch across marketing and sales teams",
    "Organize company all-hands meeting and distribute materials",
    "Execute end-of-quarter reporting and team communication",
    "Plan and execute client onboarding workflow",
    "Coordinate cross-department budget review process",
    "Manage project milestone review and stakeholder updates",
    "Execute hiring pipeline: schedule, document, communicate",
    "Prepare board meeting: slides, documents, invitations",
    "Run customer feedback analysis and team action planning",
    "Coordinate office relocation planning and communication",
    "Execute compliance audit preparation workflow",
    "Organize team offsite: logistics, agenda, follow-ups",
    "Manage vendor selection: proposals, reviews, decisions",
    "Plan product roadmap review with cross-functional input",
]

# Target element names for realistic actions
TARGETS = {
    "document_area": ["main_content", "body_text", "editor_canvas", "text_area"],
    "content_region": ["paragraph_2", "section_header", "table_cell_B3", "bullet_list"],
    "toolbar": ["bold_btn", "font_size", "style_dropdown", "format_painter"],
    "save_button": ["save_btn", "ctrl_s", "auto_save_indicator"],
    "compose_button": ["new_email_btn", "compose_icon", "new_message"],
    "recipient_field": ["to_field", "cc_field", "search_contacts"],
    "message_body": ["email_body", "rich_text_editor", "reply_area"],
    "send_button": ["send_btn", "send_and_archive", "schedule_send"],
    "calendar_button": ["calendar_icon", "schedule_tab", "planner_view"],
    "new_meeting": ["new_event_btn", "quick_meeting", "schedule_meeting"],
    "title_field": ["event_title", "subject_line", "meeting_name"],
    "attendee_field": ["add_attendees", "invite_people", "required_field"],
    "time_picker": ["start_time", "date_picker", "duration_dropdown"],
    "send_invite": ["send_invite_btn", "save_event", "schedule_btn"],
    "document_link": ["file_link", "shared_doc", "recent_file"],
    "comment_button": ["add_comment", "review_tab", "track_changes"],
    "comment_box": ["comment_input", "review_note", "annotation_box"],
    "submit_review": ["post_comment", "resolve_thread", "approve_btn"],
    "search_bar": ["search_input", "global_search", "find_box"],
    "search_submit": ["search_icon", "enter_key", "go_btn"],
    "result_item": ["result_1", "top_match", "file_result"],
    "source_element": ["selected_range", "highlighted_text", "data_table"],
    "target_app": ["word_window", "ppt_window", "outlook_window"],
    "target_location": ["cursor_position", "insert_point", "paste_target"],
    "file_item": ["document_file", "attachment", "folder_item"],
    "move_option": ["move_to", "copy_to", "organize_menu"],
    "destination_folder": ["project_folder", "archive", "shared_drive"],
    "slide_thumbnail": ["slide_3", "slide_panel", "slide_sorter"],
    "text_box": ["title_placeholder", "content_box", "notes_area"],
    "design_tab": ["design_menu", "theme_selector", "layout_option"],
    "file_menu": ["file_tab", "hamburger_menu", "more_options"],
    "export_option": ["export_pdf", "print_option", "save_as"],
    "format_selector": ["pdf_format", "docx_format", "pptx_format"],
    "confirm_button": ["ok_btn", "confirm_dialog", "apply_btn"],
    "chat_window": ["teams_chat", "channel_pane", "group_chat"],
    "message_input": ["chat_input", "reply_box", "thread_reply"],
    "share_button": ["share_icon", "share_link", "share_file"],
    "reaction_button": ["like_btn", "emoji_react", "thumbs_up"],
    "dashboard_tab": ["overview_tab", "status_board", "kanban_view"],
    "status_list": ["task_list", "activity_feed", "notification_panel"],
    "status_item": ["task_card", "alert_item", "update_entry"],
    "update_button": ["mark_complete", "update_status", "edit_task"],
    "ui_element": ["menu_item", "nav_link", "action_button"],
    "input_field": ["text_input", "dropdown_select", "checkbox"],
}


def random_coord() -> tuple:
    """Generate realistic normalized screen coordinates."""
    return (round(random.uniform(0.05, 0.95), 3),
            round(random.uniform(0.05, 0.95), 3))


def add_noise_duration(base_dur: float, std: float) -> float:
    """Add realistic noise to a duration value."""
    return max(0.1, round(random.gauss(base_dur, std), 1))


def pick_target(target_key: str) -> str:
    """Pick a specific target element name."""
    options = TARGETS.get(target_key, [target_key])
    return random.choice(options)


def generate_action(template_action: dict, noise_factor: float = 0.3) -> dict:
    """Generate a concrete action from a template action."""
    x, y = random_coord()
    target = pick_target(template_action["target"])
    action_type = template_action["action"]

    action = {
        "action_type": action_type,
        "target": target,
        "coordinates": {"x": x, "y": y},
        "description": template_action["desc"],
        "success": random.random() > 0.05,  # 5% failure rate
    }

    if action_type == "type":
        action["text_length"] = random.randint(5, 200)
    elif action_type == "scroll":
        action["scroll_amount"] = random.randint(50, 500)
    elif action_type == "switch_app":
        action["target_app"] = random.choice(["word", "powerpoint", "teams", "outlook"])

    return action


def fabricate_trajectory(
    traj_id: str,
    templates: list,
    complexity: str = "medium",
    apps: Optional[list] = None,
) -> dict:
    """Generate a single synthetic trajectory."""

    if apps is None:
        n_apps = {"low": 1, "medium": random.choice([1, 2]), "high": random.choice([2, 3, 4])}[complexity]
        apps = random.sample(["word", "powerpoint", "teams", "outlook"], n_apps)

    # Pick objective
    if len(apps) == 1:
        objective = random.choice(OBJECTIVES[apps[0]])
    else:
        objective = random.choice(MULTI_APP_OBJECTIVES)

    # Determine number of high-level steps based on complexity
    n_steps = {"low": random.randint(2, 4), "medium": random.randint(3, 5), "high": random.randint(4, 7)}[complexity]

    # Weight templates by frequency for realistic distribution
    template_weights = [t["frequency"] for t in templates]
    total_weight = sum(template_weights)
    template_probs = [w / total_weight for w in template_weights]

    # Sample skill sequence
    selected_skills = random.choices(templates, weights=template_probs, k=n_steps)

    # Build segments and actions
    segments = []
    all_actions = []
    total_duration = 0

    for i, skill in enumerate(selected_skills):
        seg_duration = add_noise_duration(skill["avg_duration_minutes"], skill["std_duration_minutes"])
        total_duration += seg_duration

        # Generate concrete actions from template
        template_actions = skill["simulated_action_sequence"]
        actions = []
        for ta in template_actions:
            action = generate_action(ta)
            action["timestamp_offset_ms"] = len(all_actions) * random.randint(500, 3000)
            actions.append(action)

        # Occasionally add extra actions for variation
        if random.random() < 0.3:
            extra = generate_action({"action": "scroll", "target": "document_body", "desc": "Scroll to see more"})
            extra["timestamp_offset_ms"] = len(all_actions) * random.randint(500, 3000)
            actions.append(extra)

        segment = {
            "segment_id": f"{traj_id}_seg{i+1}",
            "skill_type": skill["skill_type"],
            "position": i + 1,
            "duration_minutes": seg_duration,
            "apps": [random.choice(apps)] if len(apps) > 1 else apps,
            "num_actions": len(actions),
            "actions": actions,
        }
        segments.append(segment)
        all_actions.extend(actions)

    return {
        "trajectory_id": traj_id,
        "source": "fabricated",
        "seed_data": "iw_benchmark",
        "objective": objective,
        "complexity": complexity,
        "apps_involved": sorted(apps),
        "total_duration_minutes": round(total_duration, 1),
        "num_segments": len(segments),
        "num_primitive_actions": len(all_actions),
        "segments": segments,
        "skill_sequence": [s["skill_type"] for s in segments],
    }


def main():
    parser = argparse.ArgumentParser(description="Fabricate synthetic CUA trajectories")
    parser.add_argument("--num", type=int, default=500, help="Number of trajectories to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load templates
    with open(TEMPLATES_FILE) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} skill templates")

    # Generate trajectories with balanced complexity
    trajectories = []
    complexities = ["low", "medium", "high"]
    per_complexity = args.num // 3
    remainder = args.num % 3

    for ci, comp in enumerate(complexities):
        count = per_complexity + (1 if ci < remainder else 0)
        for i in range(count):
            traj_id = f"fab_{comp}_{i:04d}"
            traj = fabricate_trajectory(traj_id, templates, complexity=comp)
            trajectories.append(traj)

    # Shuffle
    random.shuffle(trajectories)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Saved {len(trajectories)} fabricated trajectories to {OUTPUT_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print("FABRICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(trajectories)}")

    from collections import Counter
    import statistics

    comp_counts = Counter(t["complexity"] for t in trajectories)
    for c in complexities:
        print(f"  {c:8s}: {comp_counts[c]}")

    n_actions = [t["num_primitive_actions"] for t in trajectories]
    n_segs = [t["num_segments"] for t in trajectories]
    durs = [t["total_duration_minutes"] for t in trajectories]

    print(f"\nActions per trajectory: min={min(n_actions)}, max={max(n_actions)}, "
          f"mean={statistics.mean(n_actions):.1f}")
    print(f"Segments per trajectory: min={min(n_segs)}, max={max(n_segs)}, "
          f"mean={statistics.mean(n_segs):.1f}")
    print(f"Duration (min): min={min(durs):.1f}, max={max(durs):.1f}, "
          f"mean={statistics.mean(durs):.1f}")

    # Skill usage
    skill_usage = Counter()
    for t in trajectories:
        for s in t["skill_sequence"]:
            skill_usage[s] += 1
    print(f"\nSkill usage across all trajectories:")
    for skill, count in skill_usage.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/sum(skill_usage.values()):.1f}%)")

    # App distribution
    app_counts = Counter()
    for t in trajectories:
        for a in t["apps_involved"]:
            app_counts[a] += 1
    print(f"\nApp involvement:")
    for app, count in app_counts.most_common():
        print(f"  {app:12s}: {count}")


if __name__ == "__main__":
    main()
