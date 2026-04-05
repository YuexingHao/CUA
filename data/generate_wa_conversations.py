#!/usr/bin/env python3
"""
Generate multi-turn user-agent conversation data from WebArena trajectories
for evaluating Qwen3-8B on WebArena skill prediction.

Unlike generate_conversations.py (which uses IW fabricated scenarios),
this script builds conversations directly from real WebArena trajectory
data, preserving the actual skill sequences and objectives.

Usage:
    python data/generate_wa_conversations.py [--max-trajs 200] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent
WA_TRAJS_FILE = DATA_DIR / "webarena_trajectories.json"
OUTPUT_FILE = DATA_DIR / "wa_conversations.jsonl"

# ── System prompt (same as IW but adapted for WebArena domain) ──────

SYSTEM_PROMPT = """\
You are InteraSkill, an AI computer-using agent that helps users complete \
tasks in web applications including maps, e-commerce, forums, and content \
management systems.

You can execute the following skills:
- document_edit: Edit text content on web pages
- send_message: Compose and send messages or posts
- search_navigate: Search for information or navigate to pages
- review_content: Read, inspect, or verify web content
- collaborate: Interact with other users (comments, replies)
- data_transfer: Copy or move data between pages/apps
- export_publish: Export, download, or publish content
- organize_files: Manage files, bookmarks, or collections
- monitor_status: Check notifications, dashboards, or status
- generic_action: Other UI interactions (clicks, form fills)

When executing a task:
1. Think step-by-step about what skills are needed
2. Explain what you're doing and why
3. Report what you observe on screen
4. Ask for clarification when the user's intent is ambiguous
5. Adapt when the user gives corrections or changes direction"""

# ── Domain-specific user message templates ──────────────────────────

# Templates keyed by (site/domain, skill) for realistic user messages
USER_MESSAGES_BY_DOMAIN = {
    "maps": {
        "search_navigate": [
            "Search for {location} on the map.",
            "Navigate to {location}.",
            "Find {location} and zoom in.",
            "Look up directions to {location}.",
            "Can you find {location} on the map?",
            "Type in {location} in the search bar.",
            "Go to the search box and enter {location}.",
            "Switch to the {layer} map view.",
        ],
        "review_content": [
            "What information is shown about this location?",
            "Check the details panel for this place.",
            "Read the route summary.",
            "What does the map show in this area?",
            "Look at the current view — what do you see?",
            "Check the distance and estimated time.",
            "Review the directions it's showing.",
        ],
        "document_edit": [
            "Add a marker at this location.",
            "Edit the route to avoid highways.",
            "Modify the waypoint.",
            "Change the destination.",
            "Update the route preferences.",
            "Adjust the map settings.",
        ],
        "collaborate": [
            "Share these directions with the group.",
            "Post this map link in the shared space.",
            "Send the route to my colleague.",
        ],
        "send_message": [
            "Share the directions via message.",
            "Send the route details to the team.",
            "Email these directions.",
        ],
        "export_publish": [
            "Save the map view as an image.",
            "Export the route details.",
            "Download the directions.",
        ],
        "data_transfer": [
            "Copy the coordinates to the clipboard.",
            "Transfer the route data.",
            "Copy the address information.",
        ],
        "generic_action": [
            "Click on that option.",
            "Select the appropriate setting.",
            "Proceed with the next step.",
        ],
    },
}

# Fallback templates for any domain
FALLBACK_USER_MESSAGES = {
    "search_navigate": [
        "Search for what we need.",
        "Navigate to the right page.",
        "Find the relevant information.",
        "Look it up.",
    ],
    "review_content": [
        "What do you see?",
        "Check the content on this page.",
        "Review what's shown.",
        "Look at the results.",
    ],
    "document_edit": [
        "Make the necessary changes.",
        "Edit the content.",
        "Update the information.",
    ],
    "collaborate": [
        "Share this with the team.",
        "Post it for others to see.",
    ],
    "send_message": [
        "Send the information.",
        "Share this via message.",
    ],
    "export_publish": [
        "Export the results.",
        "Save the output.",
    ],
    "data_transfer": [
        "Copy the data over.",
        "Transfer the information.",
    ],
    "monitor_status": [
        "Check the current status.",
        "What's the status now?",
    ],
    "organize_files": [
        "Organize the files.",
        "Sort the items.",
    ],
    "generic_action": [
        "Go ahead with the next step.",
        "Continue.",
        "Proceed.",
    ],
}

# ── Map locations for template filling ──────────────────────────────

MAP_LOCATIONS = [
    "Central Park, New York", "Times Square", "Empire State Building",
    "Grand Central Station", "Brooklyn Bridge", "Statue of Liberty",
    "Wall Street", "Broadway", "Fifth Avenue", "Rockefeller Center",
    "Madison Square Garden", "Carnegie Hall", "Lincoln Center",
    "Metropolitan Museum", "One World Trade Center", "Battery Park",
    "Union Square", "Washington Square Park", "SoHo", "Chinatown",
]

MAP_LAYERS = ["satellite", "terrain", "cycling", "transit", "standard"]

# ── Screen observation templates ────────────────────────────────────

WA_OBSERVATIONS = {
    "search_navigate": [
        "I can see the search results on the map. The location is marked with a pin.",
        "The map has centered on the searched location. I can see nearby landmarks.",
        "The search box shows the result. The map is zooming to the location.",
        "I've navigated to the page. The content is loading.",
        "The route planner has opened with the destination filled in.",
    ],
    "review_content": [
        "I can see the location details in the side panel.",
        "The route summary shows distance and estimated travel time.",
        "The map view displays the area with relevant markers.",
        "I'm reading through the content on the page.",
        "The information panel shows the details for this item.",
    ],
    "document_edit": [
        "I've made the change. The content has been updated.",
        "The edit is applied. The map/page reflects the modification.",
        "I've adjusted the settings as requested.",
    ],
    "collaborate": [
        "The content has been shared. Others can now see it.",
        "I've posted the information in the shared space.",
    ],
    "send_message": [
        "The message has been composed and sent.",
        "The directions have been shared via message.",
    ],
    "export_publish": [
        "The export is complete. The file has been saved.",
        "The content has been downloaded successfully.",
    ],
    "data_transfer": [
        "The data has been copied successfully.",
        "The information has been transferred.",
    ],
    "generic_action": [
        "The action was completed. The interface has updated.",
        "Done. The page is showing the result.",
    ],
    "monitor_status": [
        "The status panel shows the current state.",
        "I can see the notifications and updates.",
    ],
    "organize_files": [
        "The items have been organized.",
        "The files are sorted as requested.",
    ],
}

# ── Skill-specific reasoning ────────────────────────────────────────

WA_SKILL_REASONS = {
    "search_navigate": "I need to find or navigate to the relevant location/page",
    "review_content": "I should examine what's currently displayed",
    "document_edit": "the content needs to be modified",
    "collaborate": "this needs to be shared with others",
    "send_message": "we need to send this information",
    "export_publish": "the content needs to be exported or saved",
    "data_transfer": "data needs to be moved between sources",
    "generic_action": "a UI interaction is needed here",
    "monitor_status": "I should check the current status",
    "organize_files": "the items need to be organized",
}

# ── Reasoning trace templates ──────────────────────────────────────

REASONING_TEMPLATES = {
    "planning": [
        "Let me plan the approach. To {task}, I'll need to: {steps}.",
        "I'll break this down. First {first_step}, then continue from there.",
        "For this task, the workflow is: {steps}.",
    ],
    "skill_selection": [
        "I'll use **{skill}** here because {reason}.",
        "The next step requires **{skill}** — {reason}.",
        "This calls for **{skill}**. {reason}.",
    ],
    "completion": [
        "Done! I've {action}.",
        "That's complete. What would you like me to do next?",
        "Finished {action}.",
    ],
    "adaptation": [
        "Got it, I'll adjust. Let me {adjustment} instead.",
        "Understood — changing course. I'll {adjustment}.",
    ],
}

# ── Correction templates ────────────────────────────────────────────

CORRECTION_TEMPLATES = [
    "Wait, use a different route option.",
    "Actually, can you try a different approach?",
    "Hold on, zoom in more before continuing.",
    "Let me change that — use the other option instead.",
    "Actually, search for a different location first.",
    "Can you redo that? I want the walking directions instead.",
    "Switch to the satellite view first.",
]


def get_user_message(domain: str, skill: str, objective: str,
                     skill_idx: int, total_skills: int) -> str:
    """Generate a contextual user message for a given skill step."""
    # For the first message, use the objective directly
    if skill_idx == 0:
        return objective

    # Try domain-specific templates first
    domain_msgs = USER_MESSAGES_BY_DOMAIN.get(domain, {})
    templates = domain_msgs.get(skill, FALLBACK_USER_MESSAGES.get(skill, ["Continue."]))

    msg = random.choice(templates)

    # Fill placeholders
    msg = msg.replace("{location}", random.choice(MAP_LOCATIONS))
    msg = msg.replace("{layer}", random.choice(MAP_LAYERS))

    return msg


def generate_agent_turn(skill: str, is_first: bool = False,
                        task_desc: str = "", skill_flow: list = None) -> str:
    """Generate agent response with reasoning + action + observation."""
    parts = []

    # Reasoning
    if is_first and skill_flow:
        steps = ", ".join(skill_flow[:4])
        reasoning = random.choice(REASONING_TEMPLATES["planning"])
        reasoning = reasoning.replace("{task}", task_desc[:80])
        reasoning = reasoning.replace("{steps}", steps)
        reasoning = reasoning.replace("{first_step}", f"use {skill}")
        parts.append(f"**[Thinking]** {reasoning}")
    else:
        reason = WA_SKILL_REASONS.get(skill, "this is the next step")
        reasoning = random.choice(REASONING_TEMPLATES["skill_selection"])
        reasoning = reasoning.replace("{skill}", skill)
        reasoning = reasoning.replace("{reason}", reason)
        parts.append(f"**[Thinking]** {reasoning}")

    # Action
    parts.append(f"**[Action: {skill}]** Executing {skill} step...")

    # Observation
    observations = WA_OBSERVATIONS.get(skill, WA_OBSERVATIONS["generic_action"])
    parts.append(f"**[Observation]** {random.choice(observations)}")

    # Completion
    completion = random.choice(REASONING_TEMPLATES["completion"])
    completion = completion.replace("{action}", f"completed the {skill} step")
    parts.append(completion)

    return "\n\n".join(parts)


def generate_correction_response(skill: str) -> str:
    """Generate agent response to user correction."""
    adaptation = random.choice(REASONING_TEMPLATES["adaptation"])
    adaptation = adaptation.replace("{adjustment}",
                                    f"redo the {skill} step with your updated instructions")
    obs = random.choice(WA_OBSERVATIONS.get(skill, WA_OBSERVATIONS["generic_action"]))

    return (
        f"**[Adapting]** {adaptation}\n\n"
        f"**[Action: {skill}]** Redoing the step with updated approach...\n\n"
        f"**[Observation]** {obs}\n\n"
        f"I've incorporated your feedback."
    )


def generate_conversation(traj: dict, conv_id: str) -> dict:
    """Generate a multi-turn conversation from a WebArena trajectory."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    skill_flow = traj["skill_sequence"]
    objective = traj["objective"]
    domain = traj.get("domain", "maps")

    # Decide whether to add a correction (30% chance, only for longer sequences)
    correction_pos = None
    if len(skill_flow) >= 3 and random.random() < 0.3:
        correction_pos = random.randint(1, len(skill_flow) - 2)

    # First user message: the objective
    messages.append({"role": "user", "content": objective})

    # First agent turn with planning
    agent_intro = generate_agent_turn(
        skill_flow[0], is_first=True, task_desc=objective, skill_flow=skill_flow)
    messages.append({"role": "assistant", "content": agent_intro})

    # Remaining skill steps
    for i, skill in enumerate(skill_flow[1:], start=1):
        # User message
        user_msg = get_user_message(domain, skill, objective, i, len(skill_flow))
        messages.append({"role": "user", "content": user_msg})

        # Agent response
        agent_response = generate_agent_turn(skill)
        messages.append({"role": "assistant", "content": agent_response})

        # Optional correction
        if i == correction_pos:
            correction_text = random.choice(CORRECTION_TEMPLATES)
            messages.append({"role": "user", "content": correction_text})
            correction_response = generate_correction_response(skill)
            messages.append({"role": "assistant", "content": correction_response})

    # Closing
    messages.append({"role": "user", "content": "That's all, thanks!"})
    messages.append({"role": "assistant", "content":
        "You're welcome! Here's a summary of what we did:\n\n"
        + "\n".join(f"- **{skill}**: completed" for skill in skill_flow)
        + "\n\nLet me know if you need anything else!"
    })

    return {
        "conversation_id": conv_id,
        "task": objective,
        "complexity": traj.get("complexity", "medium"),
        "apps": traj.get("apps_involved", []),
        "skill_flow": skill_flow,
        "num_turns": len(messages),
        "domain": domain,
        "source": "webarena",
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate WebArena conversations for Qwen3 evaluation")
    parser.add_argument("--max-trajs", type=int, default=200,
                        help="Max trajectories to convert (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading WebArena trajectories from {WA_TRAJS_FILE}...")
    with open(WA_TRAJS_FILE) as f:
        trajs = json.load(f)
    print(f"  Loaded {len(trajs)} trajectories")

    # Sample up to max_trajs with diverse skill sequences
    if len(trajs) > args.max_trajs:
        random.shuffle(trajs)
        trajs = trajs[:args.max_trajs]

    print(f"  Using {len(trajs)} trajectories")

    # Generate conversations
    conversations = []
    for i, traj in enumerate(trajs):
        conv_id = f"wa_conv_{i:05d}"
        conv = generate_conversation(traj, conv_id)
        conversations.append(conv)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"\nSaved {len(conversations)} conversations to {OUTPUT_FILE}")

    # Statistics
    print(f"\n{'='*60}")
    print("WEBARENA CONVERSATION STATISTICS")
    print(f"{'='*60}")

    turns = [c["num_turns"] for c in conversations]
    print(f"Turns per conversation: min={min(turns)}, max={max(turns)}, "
          f"mean={sum(turns)/len(turns):.1f}")

    skill_counts = Counter()
    seq_lens = []
    for c in conversations:
        seq_lens.append(len(c["skill_flow"]))
        for s in c["skill_flow"]:
            skill_counts[s] += 1
    print(f"Skill sequence length: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")

    print(f"\nSkill distribution:")
    for skill, count in skill_counts.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/sum(skill_counts.values()):.1f}%)")


if __name__ == "__main__":
    main()
