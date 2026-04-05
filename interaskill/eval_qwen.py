"""
Evaluate the LoRA fine-tuned Qwen3-8B on multi-turn skill prediction.

For each conversation in the validation set, we replay the conversation
up to each assistant turn and check if the model's generated response
contains the correct skill action.

Usage:
    python -m interaskill.eval_qwen
"""

import json
import re
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .data import SKILL_TYPES

# ── Configuration ────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_PATH = Path("results/qwen3_lora/final_adapter")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

VALID_SKILLS = set(SKILL_TYPES)

# Regex to extract skill from agent response: [Action: skill_name]
ACTION_PATTERN = re.compile(r"\[Action:\s*(\w+)\]", re.IGNORECASE)


def load_model_and_tokenizer():
    """Load base model with LoRA adapter."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()

    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return model, tokenizer


def extract_skill_from_response(response: str) -> str:
    """Extract the skill name from an agent response.

    Looks for patterns like: [Action: document_edit]
    Falls back to matching any valid skill name in the text.
    """
    # Try regex match first
    match = ACTION_PATTERN.search(response)
    if match:
        skill = match.group(1).lower()
        if skill in VALID_SKILLS:
            return skill

    # Fallback: find any valid skill name in the response
    response_lower = response.lower()
    for skill in VALID_SKILLS:
        if skill in response_lower:
            return skill

    return response_lower.split()[0] if response_lower.strip() else "unknown"


def generate_response(model, tokenizer, messages: list[dict],
                      max_new_tokens: int = 100) -> str:
    """Generate a response given conversation history."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    print("=" * 60)
    print("Evaluating LoRA Fine-Tuned Qwen3-8B (Multi-Turn)")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load validation conversations (cap at 50 for tractable eval time)
    MAX_EVAL_CONVS = 50
    val_path = DATA_DIR / "val_conversations.jsonl"
    conversations = []
    with open(val_path) as f:
        for line in f:
            conversations.append(json.loads(line))
            if len(conversations) >= MAX_EVAL_CONVS:
                break
    print(f"\nLoaded {len(conversations)} validation conversations (max {MAX_EVAL_CONVS})", flush=True)

    # Evaluate: for each conversation, replay up to each assistant turn
    # and check if the model predicts the correct skill
    total = 0
    correct = 0
    pos_correct = {}
    pos_total = {}
    predictions = []
    skill_step = 0  # which skill in the flow

    for ci, conv in enumerate(conversations):
        msgs = conv["messages"]
        skill_flow = conv["skill_flow"]
        skill_idx = 0  # tracks which skill we're on

        for mi in range(len(msgs)):
            if msgs[mi]["role"] != "assistant":
                continue

            # The ground truth is the skill used in this assistant turn
            gt_response = msgs[mi]["content"]
            gt_skill = extract_skill_from_response(gt_response)

            if gt_skill not in VALID_SKILLS:
                continue

            # Build context: all messages up to (but not including) this assistant turn
            context = msgs[:mi]

            # Generate model response
            pred_response = generate_response(model, tokenizer, context)
            pred_skill = extract_skill_from_response(pred_response)

            is_correct = pred_skill == gt_skill
            correct += int(is_correct)
            total += 1

            # Track per-position accuracy
            pos_total[skill_idx] = pos_total.get(skill_idx, 0) + 1
            pos_correct[skill_idx] = pos_correct.get(skill_idx, 0) + int(is_correct)

            predictions.append({
                "conversation_id": conv["conversation_id"],
                "skill_position": skill_idx,
                "ground_truth": gt_skill,
                "prediction": pred_skill,
                "correct": is_correct,
                "pred_response_preview": pred_response[:200],
            })

            skill_idx += 1

        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(conversations)} conversations: "
                  f"accuracy = {correct/max(total,1):.3f} ({correct}/{total})",
                  flush=True)

    # ── Results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    overall_acc = correct / max(total, 1)
    print(f"\nOverall skill prediction accuracy: {overall_acc:.4f} ({correct}/{total})")

    print(f"\nPer-position accuracy:")
    per_pos_acc = {}
    for pos in sorted(pos_total.keys()):
        acc = pos_correct.get(pos, 0) / pos_total[pos]
        per_pos_acc[pos] = acc
        print(f"  Position {pos}: {acc:.3f} ({pos_correct.get(pos,0)}/{pos_total[pos]})")

    # Per-skill accuracy
    print(f"\nPer-skill accuracy:")
    skill_correct = Counter()
    skill_total = Counter()
    for p in predictions:
        skill_total[p["ground_truth"]] += 1
        if p["correct"]:
            skill_correct[p["ground_truth"]] += 1
    for skill in sorted(VALID_SKILLS):
        t = skill_total.get(skill, 0)
        c = skill_correct.get(skill, 0)
        if t > 0:
            print(f"  {skill:20s}: {c/t:.3f} ({c}/{t})")

    # Most common errors
    print(f"\nTop 10 errors:")
    errors = [(p["ground_truth"], p["prediction"])
              for p in predictions if not p["correct"]]
    for (gt, pred), count in Counter(errors).most_common(10):
        print(f"  {gt:20s} → {pred:20s} ({count} times)")

    # Save results
    results = {
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_PATH),
        "overall_accuracy": overall_acc,
        "per_position_accuracy": {str(k): v for k, v in per_pos_acc.items()},
        "total_samples": total,
        "correct": correct,
        "num_conversations": len(conversations),
    }

    results_path = RESULTS_DIR / "qwen3_eval_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    preds_path = RESULTS_DIR / "qwen3_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
