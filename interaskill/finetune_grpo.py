"""
GRPO (Group Relative Policy Optimization) Fine-Tuning of Qwen3-8B.

Unlike SFT which learns from demonstrations via teacher forcing, GRPO:
  1. Generates multiple candidate responses for each prompt
  2. Scores each response with a reward function
  3. Optimizes the policy to prefer high-reward responses

This is the same RL approach used in DeepSeek-R1. For our skill prediction
task, the reward function checks:
  - Did the model predict the correct skill? (+1.0)
  - Did it use the correct [Action: ...] format? (+0.2)
  - Did it include reasoning before the action? (+0.1)
  - Penalty for invalid/unknown skills (-0.5)

Usage:
    python -m interaskill.finetune_grpo
    sbatch scripts/train/run_grpo.sh
"""

import os
import re
import json
import torch
from pathlib import Path
from datasets import load_dataset

from .data import SKILL_TYPES
from .eval_model import ACTION_PATTERN

# ── Configuration ────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "qwen3_grpo"

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
      +0.3  — used [Action: skill] format (even if wrong skill)
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
            reward += 0.3

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

    print("=" * 60)
    print("GRPO Fine-Tuning: Qwen3-8B on Skill Prediction")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Data ────────────────────────────────────────
    print("\n[1/5] Loading and preparing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    train_path = str(DATA_DIR / "train_conversations.jsonl")
    val_path = str(DATA_DIR / "val_conversations.jsonl")

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
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    def reward_fn(completions, **kwargs):
        """Reward function called by GRPOTrainer.

        completions: list of generated text strings
        kwargs may contain prompts and other metadata
        """
        # GRPOTrainer passes completions as list of lists (one per prompt)
        # We need to flatten and match with ground truths
        prompts = kwargs.get("prompts", [])

        # Build a lookup from prompt to ground truth
        gt_lookup = {}
        for p, gt in zip(train_prompts, train_gt):
            gt_lookup[p[:200]] = gt  # use prefix as key

        rewards = []
        for i, completion_group in enumerate(completions):
            if isinstance(completion_group, list):
                # Multiple completions per prompt
                prompt_key = prompts[i][:200] if i < len(prompts) else ""
                gt = gt_lookup.get(prompt_key, "generic_action")
                group_rewards = skill_reward_fn(completion_group, [gt] * len(completion_group))
                rewards.append(group_rewards)
            else:
                # Single completion
                prompt_key = prompts[i][:200] if i < len(prompts) else ""
                gt = gt_lookup.get(prompt_key, "generic_action")
                r = skill_reward_fn([completion_group], [gt])[0]
                rewards.append(r)

        return rewards

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
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
        # GRPO-specific
        num_generations=4,          # Generate 4 candidates per prompt
        max_completion_length=200,  # Max tokens per generation
        # Temperature for generation diversity
        temperature=0.7,
    )

    # ── Step 5: Train ────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    print(f"  Epochs: {grpo_config.num_train_epochs}")
    print(f"  Batch: {grpo_config.per_device_train_batch_size} "
          f"x {grpo_config.gradient_accumulation_steps} accum")
    print(f"  Generations per prompt: {grpo_config.num_generations}")
    print(f"  Max completion length: {grpo_config.max_completion_length}")
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

    # Resume from checkpoint if available (for jobs that timed out)
    last_ckpt = None
    if OUTPUT_DIR.exists():
        ckpts = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"  Resuming from {last_ckpt}")

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GRPO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training time: {train_result.metrics['train_runtime']:.0f}s")
    print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # Save adapter
    adapter_path = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  GRPO adapter saved to: {adapter_path}")

    # Save metrics
    metrics = {
        **train_result.metrics,
        "method": "grpo",
        "num_generations": grpo_config.num_generations,
        "reward_components": ["correct_skill(+1.0)", "format(+0.2)",
                              "reasoning(+0.1)", "invalid(-0.5)"],
    }
    with open(OUTPUT_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {OUTPUT_DIR / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
