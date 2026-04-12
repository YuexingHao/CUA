"""
Online CUA agent using BrowserGym + InteraSkill skill composition.

Bridges the InteraSkill skill prediction pipeline with a live browser
environment via BrowserGym. The agent:
  1. Observes the page (accessibility tree + screenshot)
  2. Predicts the next skill using the SkillGrounder (Stage 1)
  3. Grounds the skill to a concrete BrowserGym action (Stage 2)
  4. Executes the action in the live browser
  5. Records the trajectory for offline skill learning

Usage:
    # Single task
    python -m interaskill.agent_online \
        --model Qwen/Qwen3-8B --adapter results/qwen3_lora/final_adapter \
        --task-id 0 --benchmark webarena

    # Batch evaluation
    python -m interaskill.agent_online \
        --model Qwen/Qwen3-8B --adapter results/qwen3_lora/final_adapter \
        --benchmark webarena --max-tasks 50

    # Open-ended (custom URL)
    python -m interaskill.agent_online \
        --model Qwen/Qwen3-8B --adapter results/qwen3_lora/final_adapter \
        --url "https://www.google.com" --goal "Search for BrowserGym"

Requires:
    pip install browsergym-core browsergym-webarena
    playwright install chromium
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    from browsergym.utils.obs import flatten_axtree_to_str
    HAS_FLATTEN = True
except ImportError:
    HAS_FLATTEN = False

from .data import SKILL_TYPES
from .grounding import (
    SkillGrounder,
    SKILL_PREDICTION_SYSTEM_PROMPT,
)
from .metrics import TaskResult, StepResult
from .eval_model import load_model_and_tokenizer, generate_response, model_short_name

RESULTS_DIR = Path("results")
TRAJ_DIR = RESULTS_DIR / "trajectories"


# ── BrowserGym Action Mapping ──────────────────────────────────────

# BrowserGym uses bid-based actions: click(bid), fill(bid, "text"), etc.
# We need to convert InteraSkill skill + LLM output into these formats.

BROWSERGYM_GROUNDING_TEMPLATE = """You are a computer-using agent executing the skill "{skill}" on a live web page.

Task: {task}

Current page (accessibility tree):
{observation}

Action history:
{history}

Select the next action. You MUST respond with EXACTLY ONE action in one of these formats:
  click(bid)           - click an element by its BrowserGym ID (bid)
  fill(bid, "text")    - type text into an input field
  select_option(bid, "option") - select a dropdown option
  scroll(x, y)         - scroll the page (use 0, 300 for down, 0, -300 for up)
  press("Enter")       - press a key
  goto("url")          - navigate to a URL
  noop()               - do nothing (wait/observe)

The bid values are shown in square brackets like [bid] in the accessibility tree above.
Respond with ONLY the action, nothing else."""

BROWSERGYM_DIRECT_TEMPLATE = """You are a computer-using agent completing a web task.

Task: {task}

Current page (accessibility tree):
{observation}

Action history:
{history}

Select the next action. Respond with EXACTLY ONE action:
  click(bid)           - click element
  fill(bid, "text")    - type into input
  select_option(bid, "option") - select dropdown
  scroll(x, y)         - scroll page
  press("Enter")       - press key
  goto("url")          - navigate
  noop()               - wait

Respond with ONLY the action."""

BROWSERGYM_ACTIONS = [
    "click", "fill", "select_option", "scroll", "press",
    "goto", "noop", "new_tab", "tab_close", "tab_focus",
]


def extract_browsergym_action(response: str) -> str:
    """Extract a valid BrowserGym action from LLM response."""
    # Strip think blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "<think>" in response:
        response = response.split("<think>")[0].strip() or response.replace("<think>", "")

    # Try to extract function-call-style action
    for action_name in BROWSERGYM_ACTIONS:
        pattern = rf'{action_name}\s*\(([^)]*)\)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            args = match.group(1).strip()
            return f"{action_name}({args})"

    # Fallback: try to find any parenthesized call
    match = re.search(r'(\w+)\(([^)]*)\)', response)
    if match:
        fn = match.group(1).lower()
        args = match.group(2)
        if fn in BROWSERGYM_ACTIONS:
            return f"{fn}({args})"

    return "noop()"


class InteraSkillAgent:
    """Online agent bridging InteraSkill with BrowserGym.

    Two-stage prediction:
      Stage 1: LLM predicts next skill from task + observation + history
      Stage 2: LLM grounds skill to a BrowserGym action (click/fill/etc.)

    Optionally records trajectories for offline skill discovery.
    """

    def __init__(self, model, tokenizer, generate_fn,
                 skill_mode="llm", grounding_mode="llm",
                 record_trajectories=True,
                 max_obs_length=4000):
        self.model = model
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.skill_mode = skill_mode
        self.grounding_mode = grounding_mode
        self.record_trajectories = record_trajectories
        self.max_obs_length = max_obs_length
        self.skill_list = ", ".join(SKILL_TYPES)

        # Episode state
        self.history = []
        self.trajectory = []
        self.task_desc = ""
        self.step_count = 0

    def reset(self, task_description: str = ""):
        """Reset agent state for a new episode."""
        self.history = []
        self.trajectory = []
        self.task_desc = task_description
        self.step_count = 0

    def _truncate_obs(self, obs_text: str) -> str:
        """Truncate observation to fit context window."""
        if len(obs_text) > self.max_obs_length:
            return obs_text[:self.max_obs_length] + "\n... (truncated)"
        return obs_text

    def _build_history_str(self) -> str:
        """Build compact history string from past actions."""
        if not self.history:
            return "None (first step)"
        recent = self.history[-5:]  # last 5 steps
        parts = []
        for h in recent:
            parts.append(f"[{h['skill']}] {h['action']}")
        return " -> ".join(parts)

    def predict_skill(self, observation: str) -> str:
        """Stage 1: Predict next skill given observation."""
        if self.skill_mode == "heuristic":
            return self._heuristic_skill(observation)

        messages = [
            {"role": "system", "content": SKILL_PREDICTION_SYSTEM_PROMPT.format(
                skills=self.skill_list)},
            {"role": "user", "content": f"""Task: {self.task_desc}

Current page:
{self._truncate_obs(observation)}

Action history:
{self._build_history_str()}

What skill should be executed next?"""},
        ]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=100)
        return self._extract_skill(response)

    def ground_action(self, skill: str, observation: str) -> str:
        """Stage 2: Ground skill to BrowserGym action."""
        if self.grounding_mode == "direct":
            template = BROWSERGYM_DIRECT_TEMPLATE
        else:
            template = BROWSERGYM_GROUNDING_TEMPLATE

        prompt = template.format(
            skill=skill,
            task=self.task_desc,
            observation=self._truncate_obs(observation),
            history=self._build_history_str(),
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=150)
        return extract_browsergym_action(response)

    def act(self, obs: dict) -> str:
        """Full agent step: observe -> predict skill -> ground action.

        Args:
            obs: BrowserGym observation dict with keys like
                 'axtree_txt', 'goal', 'url', etc.

        Returns:
            action: BrowserGym action string (e.g., "click(bid_123)")
        """
        # Extract text observation (prefer accessibility tree)
        # BrowserGym returns raw axtree_object; flatten to text with bid markers
        obs_text = ""
        if HAS_FLATTEN and "axtree_object" in obs:
            try:
                obs_text = flatten_axtree_to_str(
                    obs["axtree_object"],
                    extra_properties=obs.get("extra_element_properties", {}),
                )
            except Exception:
                pass
        if not obs_text:
            obs_text = obs.get("axtree_txt", obs.get("text", ""))
        if not obs_text:
            obs_text = obs.get("dom_txt", "(empty page)")

        # Get task from observation if not set
        if not self.task_desc and obs.get("goal"):
            self.task_desc = obs["goal"]

        # Stage 1: Predict skill
        skill = self.predict_skill(obs_text)

        # Stage 2: Ground to action
        if self.grounding_mode == "direct":
            action = self.ground_action(skill, obs_text)
        else:
            action = self.ground_action(skill, obs_text)

        # Record step
        step_record = {
            "step": self.step_count,
            "skill": skill,
            "action": action,
            "url": obs.get("url", ""),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(step_record)
        if self.record_trajectories:
            self.trajectory.append({
                **step_record,
                "observation_length": len(obs_text),
                "goal": self.task_desc,
            })

        self.step_count += 1
        return action

    def save_trajectory(self, path: Path, task_id: str = "",
                        reward: float = 0.0, completed: bool = False):
        """Save recorded trajectory to JSON."""
        data = {
            "task_id": task_id,
            "task_description": self.task_desc,
            "completed": completed,
            "reward": reward,
            "total_steps": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "steps": self.trajectory,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _extract_skill(self, response: str) -> str:
        """Extract skill from LLM response."""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if "<think>" in response:
            response = response.split("<think>")[0].strip() or response.replace("<think>", "")

        match = re.search(r"\[Action:\s*(\w+)\]", response, re.IGNORECASE)
        if match:
            skill = match.group(1).lower()
            if skill in set(SKILL_TYPES):
                return skill
        response_lower = response.lower()
        for skill in SKILL_TYPES:
            if skill in response_lower:
                return skill
        return "generic_action"

    def _heuristic_skill(self, observation: str) -> str:
        """Heuristic skill classification from page content."""
        obs_lower = observation.lower()
        if any(kw in obs_lower for kw in ["search", "query", "find"]):
            return "search_navigate"
        if any(kw in obs_lower for kw in ["form", "input", "textarea", "edit"]):
            return "document_edit"
        if any(kw in obs_lower for kw in ["select", "option", "dropdown"]):
            return "data_transfer"
        if any(kw in obs_lower for kw in ["submit", "confirm", "buy", "checkout"]):
            return "generic_action"
        return "search_navigate"


# ── Episode Runner ──────────────────────────────────────────────────

def run_episode(agent: InteraSkillAgent, env, task_id: str = "",
                max_steps: int = 15, verbose: bool = True) -> dict:
    """Run a single episode: agent interacts with live browser until done.

    Args:
        agent: InteraSkillAgent instance
        env: BrowserGym environment (gymnasium interface)
        task_id: Task identifier for logging
        max_steps: Maximum steps before truncation
        verbose: Print step-by-step output

    Returns:
        dict with task_id, reward, completed, steps, trajectory
    """
    obs, info = env.reset()
    goal = obs.get("goal", "")
    agent.reset(goal)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {goal[:100]}")
        print(f"{'='*60}")

    total_reward = 0.0
    terminated = False
    truncated = False

    for step in range(max_steps):
        # Agent decides action
        action = agent.act(obs)

        if verbose:
            skill = agent.history[-1]["skill"]
            print(f"  Step {step}: [{skill}] {action}")

        # Execute in browser
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward = reward  # BrowserGym rewards are cumulative
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            # On error, try noop to recover
            obs, reward, terminated, truncated, info = env.step("noop()")
            total_reward = reward

        if terminated or truncated:
            break

    completed = terminated and total_reward > 0.5
    if verbose:
        status = "COMPLETED" if completed else ("TRUNCATED" if truncated else "FAILED")
        print(f"\n  Result: {status} | Reward: {total_reward:.3f} | Steps: {step+1}")

    # Save trajectory
    if agent.record_trajectories:
        traj_path = TRAJ_DIR / f"{task_id or 'task'}_{datetime.now():%Y%m%d_%H%M%S}.json"
        agent.save_trajectory(traj_path, task_id, total_reward, completed)

    return {
        "task_id": task_id,
        "goal": goal,
        "reward": total_reward,
        "completed": completed,
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "trajectory": agent.trajectory,
    }


# ── Batch Evaluation ────────────────────────────────────────────────

def run_batch(agent: InteraSkillAgent, benchmark: str = "webarena",
              max_tasks: int = 50, max_steps: int = 15,
              start_task: int = 0, verbose: bool = True) -> list[dict]:
    """Run batch evaluation on a BrowserGym benchmark.

    Args:
        agent: InteraSkillAgent instance
        benchmark: "webarena", "miniwob", or "workarena"
        max_tasks: Number of tasks to evaluate
        max_steps: Maximum steps per task
        start_task: Starting task index (for resume)
        verbose: Print progress

    Returns:
        List of episode result dicts
    """
    results = []

    for task_idx in range(start_task, start_task + max_tasks):
        env_id = f"browsergym/{benchmark}.{task_idx}"
        try:
            env = gym.make(env_id, headless=True)
        except Exception as e:
            if verbose:
                print(f"  Skipping task {task_idx}: {e}")
            continue

        try:
            result = run_episode(agent, env, task_id=str(task_idx),
                                 max_steps=max_steps, verbose=verbose)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Task {task_idx} failed: {e}")
            results.append({
                "task_id": str(task_idx), "reward": 0.0,
                "completed": False, "steps": 0, "error": str(e),
            })
        finally:
            env.close()

        # Print running stats
        if verbose and len(results) % 10 == 0:
            completed = sum(1 for r in results if r.get("completed"))
            avg_reward = sum(r.get("reward", 0) for r in results) / len(results)
            print(f"\n--- Progress: {len(results)}/{max_tasks} tasks, "
                  f"TCR={completed}/{len(results)} ({completed/len(results):.1%}), "
                  f"avg_reward={avg_reward:.3f} ---\n")

        # Checkpoint every 10 tasks
        if len(results) % 10 == 0:
            ckpt_path = RESULTS_DIR / f"online_{benchmark}_checkpoint.json"
            with open(ckpt_path, "w") as f:
                json.dump({"results": results, "n_tasks": len(results)}, f)

    return results


# ── Open-Ended Mode ─────────────────────────────────────────────────

def run_openended(agent: InteraSkillAgent, url: str, goal: str,
                  max_steps: int = 20, headless: bool = False,
                  verbose: bool = True) -> dict:
    """Run agent on an arbitrary URL with a custom goal.

    Useful for demos, debugging, and exploratory testing.
    """
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": url, "goal": goal},
        wait_for_user_message=False,
        headless=headless,
    )

    try:
        result = run_episode(agent, env, task_id="openended",
                             max_steps=max_steps, verbose=verbose)
    finally:
        env.close()

    return result


# ── Main ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="InteraSkill online CUA agent with BrowserGym")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["webarena", "miniwob", "workarena"])
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run a single task by ID")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--url", type=str, default=None,
                        help="Open-ended: starting URL")
    parser.add_argument("--goal", type=str, default=None,
                        help="Open-ended: task goal")
    parser.add_argument("--skill-mode", type=str, default="llm",
                        choices=["llm", "heuristic"])
    parser.add_argument("--grounding-mode", type=str, default="llm",
                        choices=["llm", "direct"])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        default=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main():
    if not HAS_GYM:
        print("ERROR: gymnasium not installed. Run: pip install browsergym-core")
        return

    args = parse_args()

    print("=" * 60)
    print("InteraSkill Online Agent")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Skill mode: {args.skill_mode}")
    print(f"Grounding mode: {args.grounding_mode}")
    print(f"Headless: {args.headless}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    # Create agent
    agent = InteraSkillAgent(
        model, tokenizer, generate_response,
        skill_mode=args.skill_mode,
        grounding_mode=args.grounding_mode,
        record_trajectories=args.record,
    )

    # Run mode
    if args.url and args.goal:
        # Open-ended mode
        print(f"\nOpen-ended: {args.url}")
        print(f"Goal: {args.goal}")
        result = run_openended(
            agent, args.url, args.goal,
            max_steps=args.max_steps,
            headless=args.headless,
            verbose=args.verbose,
        )

    elif args.benchmark and args.task_id is not None:
        # Single task
        env_id = f"browsergym/{args.benchmark}.{args.task_id}"
        print(f"\nSingle task: {env_id}")
        env = gym.make(env_id, headless=args.headless)
        try:
            result = run_episode(
                agent, env, task_id=str(args.task_id),
                max_steps=args.max_steps, verbose=args.verbose,
            )
        finally:
            env.close()

    elif args.benchmark:
        # Batch evaluation
        print(f"\nBatch: {args.benchmark} ({args.max_tasks} tasks)")
        results = run_batch(
            agent, benchmark=args.benchmark,
            max_tasks=args.max_tasks,
            max_steps=args.max_steps,
            start_task=args.start_task,
            verbose=args.verbose,
        )

        # Summary
        completed = sum(1 for r in results if r.get("completed"))
        avg_reward = sum(r.get("reward", 0) for r in results) / max(len(results), 1)
        avg_steps = sum(r.get("steps", 0) for r in results) / max(len(results), 1)
        print(f"\n{'='*60}")
        print(f"RESULTS: {args.benchmark}")
        print(f"  Tasks: {len(results)}")
        print(f"  TCR: {completed}/{len(results)} ({completed/max(len(results),1):.1%})")
        print(f"  Avg reward: {avg_reward:.3f}")
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"{'='*60}")

        # Save final results
        short = model_short_name(args.model)
        if args.adapter:
            short += "_grpo" if "grpo" in args.adapter.lower() else "_lora"
        out_path = RESULTS_DIR / "metrics" / f"{short}_online_{args.benchmark}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model, "adapter": args.adapter,
                "benchmark": args.benchmark,
                "tcr": completed / max(len(results), 1),
                "n_completed": completed, "n_tasks": len(results),
                "avg_reward": avg_reward, "avg_steps": avg_steps,
                "results": results,
            }, f, indent=2)
        print(f"Saved to {out_path}")
    else:
        print("ERROR: Specify --benchmark (batch), --benchmark + --task-id (single),")
        print("       or --url + --goal (open-ended)")


if __name__ == "__main__":
    main()
