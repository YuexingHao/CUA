"""
Skill Grounding: Bridge high-level InteraSkill skills to low-level actions.

Two-stage prediction:
  1. Skill Predictor (LLM) → predicts next skill given task + observation + history
  2. Skill Grounder → maps skill + observation → concrete action (click, type, etc.)

Skill classification modes:
  - "heuristic": keyword-based (fast, but replicates SKILL.md's problem)
  - "learned": uses InfoNCE encoder from Phase 2 (Section 4.2)
  - "oracle": uses ground-truth skill labels (upper bound)

This module implements stage 2 and the prompt construction for both stages.
"""

import re
import torch
import numpy as np
from pathlib import Path
from .data import SKILL_TYPES, SKILL_TO_IDX, _candidate_to_text

# ── Skill-to-Action Mapping ─────────────────────────────────────────

# WebShop action types: search[query] and click[element]
WEBSHOP_SKILL_ACTIONS = {
    "search_navigate": ["search", "click"],
    "review_content": ["click"],
    "data_transfer": ["click"],  # selecting options (size, color)
    "generic_action": ["click"],  # buy, back, etc.
    "document_edit": ["search"],  # refining search query
    "monitor_status": ["click"],
    "organize_files": ["click"],
    "collaborate": ["click"],
    "export_publish": ["click"],
    "presentation_edit": ["click"],
    "schedule_meeting": ["click"],
    "send_message": ["click"],
}

# Mind2Web action types
MIND2WEB_ACTION_TYPES = ["click", "type", "select_option"]

MIND2WEB_SKILL_ACTIONS = {
    "search_navigate": ["click", "type"],
    "review_content": ["click"],
    "data_transfer": ["click", "type", "select_option"],
    "generic_action": ["click", "select_option"],
    "document_edit": ["type", "click"],
    "monitor_status": ["click"],
    "organize_files": ["click"],
    "collaborate": ["click", "type"],
    "export_publish": ["click"],
    "presentation_edit": ["click", "type"],
    "schedule_meeting": ["click", "type", "select_option"],
    "send_message": ["type", "click"],
}

# ── WebShop Phase Classification ─────────────────────────────────────

WEBSHOP_PHASE_KEYWORDS = {
    "search_navigate": [
        "search", "instruction:", "find", "looking for",
        "enter your search", "query",
    ],
    "review_content": [
        "results for", "product", "description", "features",
        "details", "specifications", "review",
    ],
    "data_transfer": [
        "options", "size", "color", "choose", "select",
        "variant", "quantity",
    ],
    "generic_action": [
        "buy now", "add to cart", "checkout", "purchase",
        "back to", "prev", "next",
    ],
}


def classify_webshop_phase(observation: str) -> str:
    """Classify the current WebShop page into an InteraSkill skill phase."""
    obs_lower = observation.lower()

    # Check option selection first (most specific product page state)
    for keyword in WEBSHOP_PHASE_KEYWORDS["data_transfer"]:
        if keyword in obs_lower:
            return "data_transfer"

    # Check results/product page
    for keyword in WEBSHOP_PHASE_KEYWORDS["review_content"]:
        if keyword in obs_lower:
            return "review_content"

    # Check for buy/cart signals (generic actions)
    for keyword in WEBSHOP_PHASE_KEYWORDS["generic_action"]:
        if keyword in obs_lower:
            return "generic_action"

    # Default to search
    return "search_navigate"


# ── Mind2Web Skill Classification ────────────────────────────────────

MIND2WEB_CONTEXT_TO_SKILL = {
    # (action_type, element_context_keywords) → skill
    "search": "search_navigate",
    "nav": "search_navigate",
    "menu": "search_navigate",
    "filter": "search_navigate",
    "sort": "search_navigate",
    "form": "document_edit",
    "input": "document_edit",
    "textarea": "document_edit",
    "edit": "document_edit",
    "fill": "document_edit",
    "submit": "generic_action",
    "confirm": "generic_action",
    "accept": "generic_action",
    "ok": "generic_action",
    "cancel": "generic_action",
    "upload": "data_transfer",
    "download": "data_transfer",
    "attach": "data_transfer",
    "file": "organize_files",
    "folder": "organize_files",
    "share": "collaborate",
    "send": "send_message",
    "email": "send_message",
    "message": "send_message",
    "export": "export_publish",
    "print": "export_publish",
    "schedule": "schedule_meeting",
    "calendar": "schedule_meeting",
    "date": "schedule_meeting",
    "review": "review_content",
    "view": "review_content",
    "detail": "review_content",
    "status": "monitor_status",
    "progress": "monitor_status",
    "dashboard": "monitor_status",
}


def classify_mind2web_step(action_type: str, element_text: str,
                           element_tag: str = "", task: str = "") -> str:
    """Classify a Mind2Web action step into an InteraSkill skill.

    Uses heuristics based on action type, element text, and task context.
    """
    context = f"{element_text} {element_tag} {task}".lower()

    # Check context keywords
    for keyword, skill in MIND2WEB_CONTEXT_TO_SKILL.items():
        if keyword in context:
            return skill

    # Fallback based on action type
    if action_type == "type":
        return "document_edit"
    elif action_type == "select_option":
        return "generic_action"
    else:
        return "generic_action"


# ── Learned Skill Classifier (Phase 2 Encoder) ───────────────────────

class LearnedSkillClassifier:
    """Classify observations into skills using the InfoNCE encoder from Phase 2.

    Instead of keyword heuristics, this uses the learned embedding space
    to find the nearest skill prototype.
    """

    def __init__(self, encoder_path: str = "results/encoder.pt",
                 prototypes_path: str = "results/skill_prototypes.pt"):
        """Load trained encoder and skill prototypes.

        Args:
            encoder_path: Path to saved SegmentEncoder state dict
            prototypes_path: Path to skill prototype embeddings (n_skills, d_latent)
        """
        from .discover import SegmentEncoder, segments_to_summaries

        self.encoder = SegmentEncoder()
        self.prototypes = None
        self._loaded = False

        encoder_p = Path(encoder_path)
        proto_p = Path(prototypes_path)

        if encoder_p.exists() and proto_p.exists():
            self.encoder.load_state_dict(torch.load(encoder_p, weights_only=True))
            self.encoder.eval()
            self.prototypes = torch.load(proto_p, weights_only=True)
            self._loaded = True
        else:
            print(f"Warning: Learned classifier not available "
                  f"(missing {encoder_p} or {proto_p}). "
                  f"Falling back to heuristic.")

    @property
    def available(self) -> bool:
        return self._loaded

    def classify(self, observation_embedding: torch.Tensor) -> str:
        """Classify a segment embedding to the nearest skill prototype.

        Args:
            observation_embedding: (d_in,) raw segment summary or (d_latent,) encoded
        """
        if not self._loaded:
            return "generic_action"

        with torch.no_grad():
            if observation_embedding.shape[0] != self.prototypes.shape[1]:
                # Raw summary → encode first
                z = self.encoder(observation_embedding.unsqueeze(0))
            else:
                z = observation_embedding.unsqueeze(0)

            # Cosine similarity to prototypes
            sims = torch.nn.functional.cosine_similarity(
                z, self.prototypes, dim=1)
            idx = sims.argmax().item()

        return SKILL_TYPES[idx] if idx < len(SKILL_TYPES) else "generic_action"

    def classify_from_text(self, observation: str, action_type: str = "",
                           element_text: str = "") -> str:
        """Fallback text-based classification when no embedding available.

        Uses the heuristic classifier as fallback since we can't embed
        raw text through the segment encoder (it expects action features).
        """
        if action_type:
            return classify_mind2web_step(action_type, element_text)
        return classify_webshop_phase(observation)


def build_skill_prototypes(encoder, summaries: torch.Tensor,
                           labels: torch.Tensor,
                           save_path: str = "results/skill_prototypes.pt"):
    """Build skill prototype embeddings by averaging encoded segments per skill.

    Call this after training the InfoNCE encoder in Phase 2.
    """
    encoder.eval()
    with torch.no_grad():
        z = encoder(summaries)  # (N, d_latent)

    prototypes = torch.zeros(len(SKILL_TYPES), z.shape[1])
    for i, skill in enumerate(SKILL_TYPES):
        mask = labels == i
        if mask.any():
            prototypes[i] = z[mask].mean(dim=0)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)

    torch.save(prototypes, save_path)
    print(f"Saved skill prototypes to {save_path}")
    return prototypes


# ── Prompt Templates ─────────────────────────────────────────────────

SKILL_PREDICTION_SYSTEM_PROMPT = """You are a computer-using agent that completes web tasks by executing skills.
Available skills: {skills}

Given the task description, current page observation, and action history,
predict the NEXT skill to execute. Respond with: [Action: skill_name]"""

SKILL_PREDICTION_TEMPLATE = """Task: {task}

Current page:
{observation}

Action history:
{history}

What skill should be executed next?"""

WEBSHOP_GROUNDING_TEMPLATE = """You are executing the skill "{skill}" on a web page.

Task: {task}
Current page:
{observation}

Available actions:
{available_actions}

Select the best action. Respond with exactly one action in the format:
search[query text] OR click[element text]"""

MIND2WEB_GROUNDING_TEMPLATE = """You are executing the skill "{skill}" on a web page.

Task: {task}
Action history: {history}

Candidate elements:
{candidates}

Select the element to interact with and the action type.
Respond in format: element_id | action_type | value
Where action_type is one of: click, type, select_option
And value is the text to type/select (empty for click)."""


# ── Grounding Classes ────────────────────────────────────────────────

SKILL_MODES = ["learned", "heuristic", "oracle"]
GROUNDING_MODES = ["llm", "heuristic"]
ABLATION_MODES = [
    "skill_llm",           # Stage 1: LLM skill → Stage 2: LLM action (full pipeline)
    "skill_heuristic",     # Stage 1: LLM skill → Stage 2: heuristic action
    "direct_action",       # No skill prediction — LLM directly predicts action
    "oracle_skill_llm",    # Stage 1: GT skill → Stage 2: LLM action (skill value upper bound)
    "heuristic_skill_llm", # Stage 1: heuristic skill → Stage 2: LLM action
    "learned_skill_llm",   # Stage 1: encoder skill → Stage 2: LLM action
]

# Prompt for direct action prediction (no skill stage — ablation baseline)
DIRECT_ACTION_TEMPLATE = """You are a web agent completing tasks.

Task: {task}

Current page:
{observation}

Action history:
{history}

Select the best next action. Respond with exactly one action:
{action_format}"""


class SkillGrounder:
    """Two-stage skill prediction + action grounding using an LLM.

    Stage 1: Predict the next skill given task + observation + history
    Stage 2: Ground the predicted skill to a concrete action

    Supports multiple skill classification modes for ablation:
      - "llm": LLM predicts skill (default)
      - "heuristic": keyword-based classification
      - "learned": InfoNCE encoder from Phase 2
      - "oracle": ground-truth skill labels (upper bound)

    And grounding modes:
      - "llm": LLM selects action given skill + observation
      - "heuristic": rule-based action selection
      - "direct": LLM directly selects action (no skill stage)
    """

    def __init__(self, model, tokenizer, generate_fn, benchmark="webshop",
                 skill_mode="llm", grounding_mode="llm",
                 learned_classifier=None):
        """
        Args:
            model: HF model (loaded with quantization)
            tokenizer: HF tokenizer
            generate_fn: Function(model, tokenizer, messages, max_new_tokens) -> str
            benchmark: "webshop" or "mind2web"
            skill_mode: "llm", "heuristic", "learned", or "oracle"
            grounding_mode: "llm", "heuristic", or "direct"
            learned_classifier: LearnedSkillClassifier instance (for skill_mode="learned")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.benchmark = benchmark
        self.skill_mode = skill_mode
        self.grounding_mode = grounding_mode
        self.learned_classifier = learned_classifier
        self.skill_list = ", ".join(SKILL_TYPES)

    def predict_skill(self, task: str, observation: str,
                      history: str = "",
                      gt_skill: str = None) -> str:
        """Stage 1: Predict next skill from context.

        Args:
            gt_skill: Ground-truth skill (used when skill_mode="oracle")
        """
        if self.skill_mode == "oracle" and gt_skill:
            return gt_skill

        if self.skill_mode == "heuristic":
            if self.benchmark == "webshop":
                return classify_webshop_phase(observation)
            else:
                return classify_mind2web_step("click", observation)

        if self.skill_mode == "learned" and self.learned_classifier:
            if self.learned_classifier.available:
                return self.learned_classifier.classify_from_text(
                    observation)

        # Default: LLM prediction
        messages = [
            {"role": "system", "content": SKILL_PREDICTION_SYSTEM_PROMPT.format(
                skills=self.skill_list)},
            {"role": "user", "content": SKILL_PREDICTION_TEMPLATE.format(
                task=task, observation=observation[:2000],
                history=history[-500:] if history else "None")},
        ]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=100)
        return self._extract_skill(response)

    def predict_action_direct(self, task: str, observation: str,
                              history: str = "",
                              available_actions: list[str] = None) -> str:
        """Direct action prediction — no skill stage (ablation baseline).

        The LLM directly predicts the next action without intermediate
        skill classification. This ablation tests whether the skill
        abstraction layer adds value.
        """
        if self.benchmark == "webshop":
            action_format = "search[query text] OR click[element text]"
        else:
            action_format = "element_id | action_type | value"

        messages = [
            {"role": "user", "content": DIRECT_ACTION_TEMPLATE.format(
                task=task, observation=observation[:2000],
                history=history[-500:] if history else "None",
                action_format=action_format)},
        ]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=100)

        if self.benchmark == "webshop":
            return self._extract_webshop_action(response)
        else:
            return self._extract_mind2web_action(response)

    def ground_action_webshop(self, skill: str, task: str,
                              observation: str,
                              available_actions: list[str]) -> str:
        """Stage 2 (WebShop): Map skill + observation to WebShop action."""
        messages = [
            {"role": "user", "content": WEBSHOP_GROUNDING_TEMPLATE.format(
                skill=skill, task=task,
                observation=observation[:2000],
                available_actions="\n".join(available_actions[:20]))},
        ]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=100)
        return self._extract_webshop_action(response)

    def ground_action_mind2web(self, skill: str, task: str,
                               history: str,
                               candidates: list[dict]) -> tuple[str, str, str]:
        """Stage 2 (Mind2Web): Map skill + DOM candidates to action.

        Returns: (element_id, action_type, value)
        """
        cand_text = "\n".join(
            f"[{i}] {_candidate_to_text(c)}"
            for i, c in enumerate(candidates[:20])
        )
        messages = [
            {"role": "user", "content": MIND2WEB_GROUNDING_TEMPLATE.format(
                skill=skill, task=task,
                history=history[-500:] if history else "None",
                candidates=cand_text)},
        ]
        response = self.generate_fn(
            self.model, self.tokenizer, messages, max_new_tokens=100)
        return self._extract_mind2web_action(response)

    def _extract_skill(self, response: str) -> str:
        """Extract skill name from model response."""
        # Strip <think>...</think> blocks (Qwen3)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if "<think>" in response:
            response = response.split("<think>")[0].strip() or response.replace("<think>", "")

        match = re.search(r"\[Action:\s*(\w+)\]", response, re.IGNORECASE)
        if match:
            skill = match.group(1).lower()
            if skill in set(SKILL_TYPES):
                return skill

        # Fallback: find any valid skill name in response
        response_lower = response.lower()
        for skill in SKILL_TYPES:
            if skill in response_lower:
                return skill
        return "generic_action"

    def _extract_webshop_action(self, response: str) -> str:
        """Extract WebShop action from model response."""
        # Match search[...] or click[...]
        match = re.search(r"(search|click)\[([^\]]+)\]", response, re.IGNORECASE)
        if match:
            action_type = match.group(1).lower()
            target = match.group(2).strip()
            return f"{action_type}[{target}]"
        # Fallback
        return "click[buy now]"

    def _extract_mind2web_action(self, response: str) -> tuple[str, str, str]:
        """Extract Mind2Web action from model response.

        Expected format: element_id | action_type | value
        """
        parts = response.strip().split("|")
        if len(parts) >= 2:
            elem_id = parts[0].strip()
            action_type = parts[1].strip().lower()
            value = parts[2].strip() if len(parts) > 2 else ""
            if action_type in MIND2WEB_ACTION_TYPES:
                return elem_id, action_type, value
        # Fallback
        return "0", "click", ""


class HeuristicGrounder:
    """Rule-based grounder for ablation (no LLM for stage 2).

    Uses the skill phase + simple heuristics to select actions.
    Useful as a baseline to measure the value of LLM-based grounding.
    """

    def ground_action_webshop(self, skill: str, observation: str,
                              available_actions: list[str]) -> str:
        """Select action using skill-based heuristics."""
        obs_lower = observation.lower()

        if skill == "search_navigate":
            # If on search page, search; otherwise click first result
            for action in available_actions:
                if action.startswith("search["):
                    return action
            for action in available_actions:
                if action.startswith("click["):
                    return action

        elif skill == "review_content":
            # Click the first product or detail link
            for action in available_actions:
                if "click[" in action:
                    return action

        elif skill == "data_transfer":
            # Click first option
            for action in available_actions:
                if "click[" in action:
                    return action

        elif skill == "generic_action":
            # Look for buy/cart buttons
            for action in available_actions:
                a_lower = action.lower()
                if any(kw in a_lower for kw in ["buy", "cart", "purchase"]):
                    return action

        # Fallback: first available action
        return available_actions[0] if available_actions else "click[back]"
