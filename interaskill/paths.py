"""Repository paths used by InteraSkill scripts.

Keep dataset locations centralized so command-line tools work from a fresh
clone regardless of the user's current working directory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

IW_DIR = DATA_DIR / "interaskill"
IW_RAW_DIR = IW_DIR / "raw"
IW_PROCESSED_DIR = IW_DIR / "processed"
IW_CONVERSATIONS_DIR = IW_DIR / "conversations"

IW_BENCHMARK_EXAMPLES = IW_RAW_DIR / "iw-benchmark-examples.json"
IW_PARSED_TRAJECTORIES = IW_PROCESSED_DIR / "parsed_trajectories.json"
IW_SUMMARY = IW_PROCESSED_DIR / "iw_summary.json"
IW_SKILL_TEMPLATES = IW_PROCESSED_DIR / "iw_skill_templates.json"
IW_FABRICATED_TRAJECTORIES = IW_PROCESSED_DIR / "fabricated_trajectories.json"
IW_TRAIN_CONVERSATIONS = IW_CONVERSATIONS_DIR / "train_conversations.jsonl"
IW_VAL_CONVERSATIONS = IW_CONVERSATIONS_DIR / "val_conversations.jsonl"
IW_TRAIN_LLM = IW_CONVERSATIONS_DIR / "train_llm.jsonl"
IW_VAL_LLM = IW_CONVERSATIONS_DIR / "val_llm.jsonl"

WEBARENA_DIR = DATA_DIR / "webarena"
WEBARENA_TRAJECTORIES_DIR = WEBARENA_DIR / "trajectories"
WEBARENA_CONVERSATIONS_DIR = WEBARENA_DIR / "conversations"
WEBARENA_TRAJECTORIES = WEBARENA_TRAJECTORIES_DIR / "webarena_trajectories.json"
WEBARENA_EXAMPLE_RAW = WEBARENA_TRAJECTORIES_DIR / "webarena_example_centralpark_timessquare_raw.json"
WEBARENA_CONVERSATIONS = WEBARENA_CONVERSATIONS_DIR / "wa_conversations.jsonl"

BROWSECOMP_DIR = DATA_DIR / "browsecomp_plus"
BROWSECOMP_REPO_DIR = BROWSECOMP_DIR / "repo"
BROWSECOMP_CONVERSATIONS = BROWSECOMP_DIR / "conversations" / "bc_conversations.jsonl"

MIND2WEB_DIR = DATA_DIR / "mind2web"
MIND2WEB_CACHE_DIR = MIND2WEB_DIR / "cache"

WORKARENA_DIR = DATA_DIR / "workarena"
WORKARENA_TRAJECTORIES = WORKARENA_DIR / "trajectories" / "workarena_trajectories.json"
WORKARENA_CONVERSATIONS = WORKARENA_DIR / "conversations" / "workarena_conversations.jsonl"
WORKARENA_NLP = WORKARENA_DIR / "workarena_nlp.json"


def ensure_parent(path: Path) -> Path:
    """Create a path's parent directory and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
