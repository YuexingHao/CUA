"""
GRPO (Group Relative Policy Optimization) fine-tuning for skill prediction.

Unlike SFT which learns from demonstrations via teacher forcing, GRPO:
  1. Generates multiple candidate responses for each prompt
  2. Scores each response with a reward function
  3. Optimizes the policy to prefer high-reward responses

This is the same RL approach used in DeepSeek-R1. For our skill prediction
task, the reward function checks:
  - Did the model predict the correct skill? (+1.0)
  - Did it use the correct [Action: ...] format? (+0.1)
  - Did it include reasoning before the action? (+0.1)
  - Penalty for invalid/unknown skills (-0.3)

Usage:
    python -m interaskill.finetune_grpo
    python -m interaskill.finetune_grpo \\
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \\
        --output-dir results/llama31_70b_grpo \\
        --num-generations 2 --max-completion-length 128

    sbatch scripts/train/run_grpo.sh

Llama-3.1-70B GRPO: weights in 4-bit still need large aggregate VRAM during
rollouts (several generations per step). Prefer multiple 80GB GPUs
(device_map=\"auto\" uses all visible CUDA devices) or lower --num-generations
/--max-completion-length. Gated Llama weights require HF_TOKEN.
"""

import argparse
import re
import json
import torch
from pathlib import Path

from .data import SKILL_TYPES
from .eval_model import ACTION_PATTERN
from .paths import IW_TRAIN_CONVERSATIONS, IW_VAL_CONVERSATIONS, RESULTS_DIR

# ── Defaults (overridden by CLI) ─────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-8B"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "qwen3_grpo"


def _default_output_dir(model_name: str) -> Path:
    """results/<slug>_grpo from the last path segment of the HF model id."""
    slug = model_name.rstrip("/").split("/")[-1].lower().replace(".", "_").replace("-", "_")
    return RESULTS_DIR / f"{slug}_grpo"

VALID_SKILLS = set(SKILL_TYPES)

# ── Reward Function ──────────────────────────────────────────────────

def _extract_skill_robust(text: str) -> str | None:
    """Robustly extract a skill name from model output.

    Handles multiple formats to avoid parsing errors deflating scores:
      - [Action: skill_name]
      - **[Action: skill_name]**
      - Action: skill_name
      - skill_name (bare mention)
      - Variations with underscores, hyphens, spaces
    """
    # Strip thinking blocks
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in clean:
        clean = clean.split("<think>")[0].strip() or clean.replace("<think>", "")

    # Try strict format: [Action: skill_name]
    match = ACTION_PATTERN.search(clean)
    if match:
        skill = match.group(1).lower().replace("-", "_").replace(" ", "_")
        if skill in VALID_SKILLS:
            return skill

    # Try looser format: Action: skill_name (without brackets)
    match = re.search(r"action\s*:\s*(\w+)", clean, re.IGNORECASE)
    if match:
        skill = match.group(1).lower().replace("-", "_").replace(" ", "_")
        if skill in VALID_SKILLS:
            return skill

    # Try bare skill name mention (last resort)
    clean_lower = clean.lower()
    for skill in VALID_SKILLS:
        if skill in clean_lower:
            return skill
        # Also try with spaces/hyphens: "search navigate" or "search-navigate"
        if skill.replace("_", " ") in clean_lower:
            return skill
        if skill.replace("_", "-") in clean_lower:
            return skill

    return None


def skill_reward_fn(completions: list[str], ground_truths: list[str],
                    **kwargs) -> list[float]:
    """Compute rewards for a batch of generated skill predictions.

    Reward components:
      +1.0  — correct skill prediction
      +0.1  — used [Action: skill] format (even if wrong skill)
      +0.1  — included reasoning/thinking before action
      -0.3  — no recognizable skill in output
      +0.0  — wrong skill but valid

    Args:
        completions: List of model-generated responses
        ground_truths: List of ground-truth skill names

    Returns:
        List of reward scores
    """
    rewards = []
    for completion, gt_skill in zip(completions, ground_truths):
        reward = 0.0

        pred_skill = _extract_skill_robust(completion)

        # Format bonus: used [Action: ...] pattern
        clean = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
        if ACTION_PATTERN.search(clean):
            reward += 0.1

        # Reasoning bonus
        if any(kw in clean.lower() for kw in ["thinking", "step", "because", "need to"]):
            reward += 0.1

        # Correctness
        if pred_skill == gt_skill:
            reward += 1.0
        elif pred_skill is not None:
            reward += 0.0  # wrong but recognized
        else:
            reward -= 0.3  # couldn't parse any skill

        rewards.append(reward)

    return rewards


# ── Data Preparation ─────────────────────────────────────────────────

def prepare_grpo_dataset(tokenizer, data_path: str, max_samples: int = None):
    """Convert conversation data to prompt/ground-truth pairs for GRPO.

    For each assistant turn that contains a skill action:
      - prompt = all messages up to that turn
      - ground_truth = the skill in [Action: skill_name]
    """
    conversations = []
    with open(data_path) as f:
        for line in f:
            conversations.append(json.loads(line))

    prompts = []
    gt_skills = []

    for conv in conversations:
        msgs = conv["messages"]
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue

            # Extract ground-truth skill from this assistant turn
            match = ACTION_PATTERN.search(msg["content"])
            if not match:
                continue
            gt_skill = match.group(1).lower()
            if gt_skill not in VALID_SKILLS:
                continue

            # Build prompt from all messages up to this turn
            context = msgs[:i]
            try:
                prompt = tokenizer.apply_chat_template(
                    context,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    context,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompts.append(prompt)
            gt_skills.append(gt_skill)

            if max_samples and len(prompts) >= max_samples:
                break
        if max_samples and len(prompts) >= max_samples:
            break

    return prompts, gt_skills


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOTrainer, GRPOConfig

    parser = argparse.ArgumentParser(description="GRPO fine-tuning for skill prediction.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Training output directory (default: results/<model_slug>_grpo).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoints in output-dir.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO rollouts per prompt (lower for 70B / tight VRAM).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=200,
        help="Max new tokens per rollout completion.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="GRPO learning rate.",
    )
    args = parser.parse_args()

    model_name = args.model
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(model_name)
    # Keep historical default path when using the default Qwen id and no explicit --output-dir.
    if model_name == DEFAULT_MODEL and args.output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    print("=" * 60)
    print(f"GRPO fine-tuning: {model_name}")
    print(f"  output_dir={output_dir}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Data ────────────────────────────────────────
    print("\n[1/5] Loading and preparing data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    train_path = str(IW_TRAIN_CONVERSATIONS)
    val_path = str(IW_VAL_CONVERSATIONS)

    train_prompts, train_gt = prepare_grpo_dataset(tokenizer, train_path)
    val_prompts, val_gt = prepare_grpo_dataset(tokenizer, val_path, max_samples=200)

    print(f"  Train: {len(train_prompts)} skill prediction prompts")
    print(f"  Val:   {len(val_prompts)} prompts")

    # Create HF dataset
    from datasets import Dataset
    train_dataset = Dataset.from_dict({
        "prompt": train_prompts,
        "ground_truth": train_gt,
    })
    val_dataset = Dataset.from_dict({
        "prompt": val_prompts,
        "ground_truth": val_gt,
    })

    # ── Step 2: Load Model ───────────────────────────────────────
    print("\n[2/5] Loading model with 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 3: Apply LoRA ───────────────────────────────────────
    print("\n[3/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Total: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # ── Step 4: Configure GRPO ───────────────────────────────────
    print("\n[4/5] Configuring GRPO trainer...")

    # Reward wrapper that extracts ground_truth from the dataset
    def _completion_text(completion) -> str:
        """Normalize TRL completion payloads across string/chat formats."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", completion))
        if isinstance(completion, list):
            return "\n".join(_completion_text(item) for item in completion)
        return str(completion)

    def _expand_ground_truths(raw_ground_truths, n_completions: int) -> list[str] | None:
        if raw_ground_truths is None:
            return None
        if isinstance(raw_ground_truths, str):
            raw_ground_truths = [raw_ground_truths]
        ground_truths = list(raw_ground_truths)
        if not ground_truths:
            return None
        if len(ground_truths) == n_completions:
            return ground_truths
        if n_completions % len(ground_truths) == 0:
            repeats = n_completions // len(ground_truths)
            return [gt for gt in ground_truths for _ in range(repeats)]
        return None

    def reward_fn(completions, **kwargs):
        """Reward function called by GRPOTrainer.

        completions: list of generated text strings
        kwargs may contain prompts and other metadata
        """
        ground_truths = (
            kwargs.get("ground_truth")
            or kwargs.get("ground_truths")
            or kwargs.get("gt_skill")
        )

        # Newer TRL passes dataset columns through kwargs. Use those labels
        # directly; prompt-prefix lookup is only a fallback for older versions.
        if completions and all(isinstance(c, list) for c in completions):
            expanded = _expand_ground_truths(ground_truths, len(completions))
            rewards = []
            for i, completion_group in enumerate(completions):
                gt = expanded[i] if expanded else None
                if gt is None:
                    prompts = kwargs.get("prompts", [])
                    prompt_key = prompts[i][:200] if i < len(prompts) else ""
                    gt_lookup = {p[:200]: gt for p, gt in zip(train_prompts, train_gt)}
                    gt = gt_lookup.get(prompt_key, "generic_action")
                texts = [_completion_text(c) for c in completion_group]
                rewards.append(skill_reward_fn(texts, [gt] * len(texts)))
            return rewards

        texts = [_completion_text(c) for c in completions]
        expanded = _expand_ground_truths(ground_truths, len(texts))
        if expanded is not None:
            return skill_reward_fn(texts, expanded)

        prompts = kwargs.get("prompts", [])

        # Build a lookup from prompt to ground truth
        gt_lookup = {}
        for p, gt in zip(train_prompts, train_gt):
            gt_lookup[p[:200]] = gt  # use prefix as key

        rewards = []
        for i, completion in enumerate(texts):
            prompt_key = prompts[i][:200] if i < len(prompts) else ""
            gt = gt_lookup.get(prompt_key, "generic_action")
            rewards.append(skill_reward_fn([completion], [gt])[0])

        return rewards

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
    )

    # ── Step 5: Train ────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    print(f"  Epochs: {grpo_config.num_train_epochs}")
    print(f"  Batch: {grpo_config.per_device_train_batch_size} "
          f"x {grpo_config.gradient_accumulation_steps} accum")
    print(f"  Generations per prompt: {grpo_config.num_generations}")
    print(f"  Max completion length: {grpo_config.max_completion_length}")
    print(f"  Gradient accumulation: {grpo_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print()

    # Patch for trl/peft compatibility: GRPOTrainer accesses
    # model.warnings_issued which doesn't exist on PEFT-wrapped models
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    last_ckpt = None
    if not args.no_resume and output_dir.exists():
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"  Resuming from {last_ckpt}")
    elif args.no_resume:
        print("  --no-resume: starting without loading a checkpoint")

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GRPO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training time: {train_result.metrics['train_runtime']:.0f}s")
    print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  GRPO adapter saved to: {adapter_path}")

    metrics = {
        **train_result.metrics,
        "method": "grpo",
        "base_model": model_name,
        "num_generations": grpo_config.num_generations,
        "reward_components": ["correct_skill(+1.0)", "format(+0.1)",
                              "reasoning(+0.1)", "invalid(-0.3)"],
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {output_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
