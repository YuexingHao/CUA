"""
WebShop environment wrapper with InteraSkill skill-phase classification.

Wraps WebShop's text environment to provide:
  - Skill-phase classification for each observation
  - Available actions extraction from page text
  - Reward tracking with sub-goal decomposition

WebShop action space: search[query] and click[element]
"""

import re
import sys
from pathlib import Path


# ── WebShop Phase Detection ──────────────────────────────────────────

def detect_page_type(observation: str) -> str:
    """Detect the current WebShop page type from observation text.

    Returns: "search", "results", "product", "options", "done"
    """
    obs_lower = observation.lower()

    if "your score" in obs_lower or "thank you" in obs_lower:
        return "done"

    if any(kw in obs_lower for kw in [
        "instruction:", "enter your search", "search query"
    ]):
        return "search"

    if any(kw in obs_lower for kw in [
        "buy now", "add to cart", "options", "size:", "color:"
    ]):
        if any(kw in obs_lower for kw in ["size:", "color:", "options"]):
            return "options"
        return "product"

    if "results for" in obs_lower or "page " in obs_lower:
        return "results"

    # Default: if we see product listings, it's results
    if obs_lower.count("[button]") > 3:
        return "results"

    return "search"


def page_type_to_skill(page_type: str) -> str:
    """Map WebShop page type to InteraSkill skill."""
    return {
        "search": "search_navigate",
        "results": "review_content",
        "product": "review_content",
        "options": "data_transfer",
        "done": "generic_action",
    }.get(page_type, "generic_action")


def extract_available_actions(observation: str) -> list[str]:
    """Extract clickable elements from WebShop observation text.

    WebShop observations contain [button] and [link] markers.
    Returns list of action strings like 'click[Buy Now]' or 'search[...]'.
    """
    actions = []
    page_type = detect_page_type(observation)

    # On search page, search is available
    if page_type == "search":
        actions.append("search[<query>]")

    # Extract clickable elements
    # WebShop format: "[button] Element Text [button]" or similar
    button_pattern = re.compile(r'\[(?:button|link)\]\s*([^\[\]]+?)(?:\s*\[|$)')
    for match in button_pattern.finditer(observation):
        text = match.group(1).strip()
        if text:
            actions.append(f"click[{text}]")

    # Also look for bare bracketed items
    bracket_pattern = re.compile(r'(?:^|\n)\s*([A-Z][^[\n]{3,50})\s*(?:\[|$)')
    for match in bracket_pattern.finditer(observation):
        text = match.group(1).strip()
        if text and f"click[{text}]" not in actions:
            actions.append(f"click[{text}]")

    if not actions:
        actions.append("click[back to search]")

    return actions


# ── Sub-goal Tracking ────────────────────────────────────────────────

class WebShopSubGoals:
    """Track WebShop sub-goals for partial task success (L2 metric).

    Sub-goals:
      1. search: Found relevant products (reached results page)
      2. select: Selected a product and options (reached product/options page)
      3. purchase: Completed purchase (clicked Buy Now)
    """

    def __init__(self):
        self.searched = False
        self.selected = False
        self.purchased = False
        self._page_history = []

    def update(self, observation: str, action: str):
        """Update sub-goal completion based on observation and action."""
        page_type = detect_page_type(observation)
        self._page_history.append(page_type)

        if page_type == "results":
            self.searched = True
        if page_type in ("product", "options"):
            self.selected = True
        if "buy" in action.lower() or page_type == "done":
            self.purchased = True

    @property
    def completion_vector(self) -> list[bool]:
        """Return [searched, selected, purchased]."""
        return [self.searched, self.selected, self.purchased]

    @property
    def partial_score(self) -> float:
        """Fraction of sub-goals completed."""
        return sum(self.completion_vector) / 3.0


# ── Environment Wrapper ──────────────────────────────────────────────

class WebShopSkillEnv:
    """Wraps WebShop's text environment with InteraSkill skill tracking.

    This wrapper adds:
      - Skill phase classification per step
      - Available actions extraction
      - Sub-goal tracking for partial task success
      - Skill transition counting for efficiency metrics
    """

    def __init__(self, webshop_dir: str = "third_party/WebShop"):
        """Initialize WebShop environment.

        Args:
            webshop_dir: Path to cloned WebShop repository
        """
        self.webshop_dir = Path(webshop_dir)
        self.env = None
        self.subgoals = None
        self.skill_history = []
        self.action_history = []
        self.current_obs = ""
        self._setup_env()

    def _setup_env(self):
        """Import and initialize the WebShop environment."""
        ws_path = str(self.webshop_dir)
        if ws_path not in sys.path:
            sys.path.insert(0, ws_path)

        try:
            from web_agent_site.envs import WebAgentTextEnv
            self.env = WebAgentTextEnv(
                observation_mode="text",
                human_goals=True,
            )
        except ImportError:
            print("Warning: WebShop not installed. "
                  "Run `python data/setup_webshop.py` first.")
            self.env = None

    def reset(self, task_idx: int = None) -> dict:
        """Reset environment for a new task.

        Returns:
            dict with keys: observation, task, skill, available_actions, subgoals
        """
        if self.env is None:
            raise RuntimeError("WebShop environment not initialized")

        if task_idx is not None:
            obs, info = self.env.reset(task_idx)
        else:
            obs, info = self.env.reset()

        self.current_obs = obs
        self.subgoals = WebShopSubGoals()
        self.skill_history = []
        self.action_history = []

        page_type = detect_page_type(obs)
        skill = page_type_to_skill(page_type)
        self.skill_history.append(skill)

        return {
            "observation": obs,
            "task": info.get("goal", obs.split("\n")[0]),
            "skill": skill,
            "page_type": page_type,
            "available_actions": extract_available_actions(obs),
            "subgoals": self.subgoals.completion_vector,
        }

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """Execute an action in WebShop.

        Args:
            action: WebShop action string, e.g. "search[red jacket]"
                    or "click[Buy Now]"

        Returns:
            (state_dict, reward, done, info)
        """
        if self.env is None:
            raise RuntimeError("WebShop environment not initialized")

        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        self.action_history.append(action)

        # Update sub-goals
        self.subgoals.update(obs, action)

        # Classify skill phase
        page_type = detect_page_type(obs)
        skill = page_type_to_skill(page_type)
        self.skill_history.append(skill)

        # Count skill switches
        n_switches = sum(
            1 for i in range(1, len(self.skill_history))
            if self.skill_history[i] != self.skill_history[i - 1]
        )

        state = {
            "observation": obs,
            "skill": skill,
            "page_type": page_type,
            "available_actions": extract_available_actions(obs),
            "subgoals": self.subgoals.completion_vector,
        }

        info.update({
            "skill_history": list(self.skill_history),
            "action_history": list(self.action_history),
            "n_skill_switches": n_switches,
            "partial_score": self.subgoals.partial_score,
        })

        return state, reward, done, info

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        if self.env and hasattr(self.env, "num_goals"):
            return self.env.num_goals
        return 500  # Default WebShop test set size
