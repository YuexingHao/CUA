"""
Prepare trajectory data for LLM/VLM fine-tuning.

Converts fabricated_trajectories.json into prompt-completion pairs
for training a model to predict skill sequences.

Format per sample:
  - Input:  trajectory context (objective, apps, action history so far)
  - Output: next skill type

This creates train/val JSONL files ready for fine-tuning.
"""

import json
import random
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
TRAJ_FILE = DATA_DIR / "fabricated_trajectories.json"

# ── System prompt (shared across all samples) ────────────────────────
SYSTEM_PROMPT = (
    "You are InteraSkill, a computer-using agent that predicts the next "
    "skill to execute in a workflow trajectory.\n\n"
    "Available skills: collaborate, data_transfer, document_edit, "
    "export_publish, generic_action, monitor_status, organize_files, "
    "presentation_edit, review_content, schedule_meeting, "
    "search_navigate, send_message.\n\n"
    "Given a trajectory context (objective, apps, and skills executed so far), "
    "predict the NEXT skill. Reply with only the skill name, nothing else."
)


def trajectory_to_samples(traj: dict) -> list[dict]:
    """Convert one trajectory into multiple training samples.

    For a trajectory with skills [A, B, C, D], we create:
      - Input: "Objective: ... | Skills so far: A"        → Output: "B"
      - Input: "Objective: ... | Skills so far: A, B"     → Output: "C"
      - Input: "Objective: ... | Skills so far: A, B, C"  → Output: "D"

    This teaches the model to predict the next skill at every position.
    """
    skills = traj["skill_sequence"]
    if len(skills) < 2:
        return []

    samples = []
    for i in range(1, len(skills)):
        # Build context: what happened so far
        skills_so_far = skills[:i]
        next_skill = skills[i]

        # Summarize actions in the most recent segment
        recent_seg = traj["segments"][i - 1]
        action_summary = ", ".join(
            a["action_type"] for a in recent_seg["actions"][:5]  # first 5 actions
        )

        user_prompt = (
            f"Objective: {traj['objective']}\n"
            f"Apps: {', '.join(traj['apps_involved'])}\n"
            f"Complexity: {traj['complexity']}\n"
            f"Skills completed ({len(skills_so_far)}/{len(skills)}): "
            f"{' → '.join(skills_so_far)}\n"
            f"Recent actions: {action_summary}\n"
            f"What is the next skill?"
        )

        # Chat format for fine-tuning
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": next_skill},
            ]
        }
        samples.append(sample)

    return samples


def main():
    # Load trajectories
    with open(TRAJ_FILE) as f:
        trajectories = json.load(f)
    print(f"Loaded {len(trajectories)} trajectories")

    # Convert to samples
    all_samples = []
    for traj in trajectories:
        all_samples.extend(trajectory_to_samples(traj))
    print(f"Generated {len(all_samples)} training samples")

    # Shuffle and split: 85% train, 15% val
    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.85 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    # Save as JSONL (one JSON object per line)
    train_path = DATA_DIR / "train_llm.jsonl"
    val_path = DATA_DIR / "val_llm.jsonl"

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(samples)} samples to {path}")

    # Print example
    print(f"\n{'='*60}")
    print("EXAMPLE TRAINING SAMPLE")
    print(f"{'='*60}")
    ex = train_samples[0]
    print(f"\n[System]\n{ex['messages'][0]['content'][:200]}...")
    print(f"\n[User]\n{ex['messages'][1]['content']}")
    print(f"\n[Assistant]\n{ex['messages'][2]['content']}")

    # Print label distribution
    from collections import Counter
    label_dist = Counter(s["messages"][2]["content"] for s in all_samples)
    print(f"\n{'='*60}")
    print("LABEL DISTRIBUTION")
    print(f"{'='*60}")
    for skill, count in label_dist.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/len(all_samples):.1f}%)")


if __name__ == "__main__":
    main()
