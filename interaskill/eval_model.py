"""
Evaluate any HF model (zero-shot or LoRA) on multi-turn skill prediction.

Supports:
  - Zero-shot evaluation of any instruction-tuned model
  - LoRA adapter evaluation on top of a base model

Usage:
    # Zero-shot Llama-3.1-70B on WebArena
    python -m interaskill.eval_model \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --dataset wa --max-convs 200

    # Zero-shot Gemma-4-31B on IW
    python -m interaskill.eval_model \
        --model google/gemma-4-31B-it \
        --dataset iw --max-convs 50

    # LoRA-adapted Qwen3-8B
    python -m interaskill.eval_model \
        --model Qwen/Qwen3-8B \
        --adapter results/qwen3_lora/final_adapter \
        --dataset iw --max-convs 50
"""

import argparse
import json
import re
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Patch for DeepSeek-R1: is_torch_fx_available was removed in transformers 5.x
import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, "is_torch_fx_available"):
    _import_utils.is_torch_fx_available = lambda: False

from .data import SKILL_TYPES

# ── Configuration ────────────────────────────────────────────────────

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HF model on skill prediction")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--dataset", choices=["iw", "wa", "bc"], default="iw",
                        help="Dataset: iw (fabricated), wa (WebArena), or bc (BrowseComp-Plus)")
    parser.add_argument("--max-convs", type=int, default=50,
                        help="Max conversations to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Max tokens to generate per response")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, adapter_path: str = None):
    """Load model with 4-bit quantization and optional LoRA adapter."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (4-bit): {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  Using {n_gpus} GPUs")
    return model, tokenizer


def extract_skill_from_response(response: str) -> str:
    """Extract skill name from model response."""
    # Try [Action: skill_name] pattern
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


def _fallback_chat_format(messages: list[dict]) -> str:
    """Simple fallback for models without a chat template."""
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        parts.append(f"### {role}:\n{msg['content']}\n")
    parts.append("### Assistant:\n")
    return "\n".join(parts)


def generate_response(model, tokenizer, messages: list[dict],
                      max_new_tokens: int = 150) -> str:
    """Generate a response given conversation history."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = _fallback_chat_format(messages)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def normalized_edit_distance(pred_seqs: list[list[str]],
                             gt_seqs: list[list[str]]) -> float:
    """Compute mean normalized edit distance."""
    def edit_dist(a, b):
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1),
                )
        return dp[n][m]

    total = 0.0
    for pred, gt in zip(pred_seqs, gt_seqs):
        max_len = max(len(pred), len(gt), 1)
        total += edit_dist(pred, gt) / max_len
    return total / max(len(pred_seqs), 1)


def model_short_name(model_name: str) -> str:
    """Extract a short name for file naming."""
    # meta-llama/Meta-Llama-3.1-70B-Instruct -> llama3.1-70b
    # google/gemma-4-31B-it -> gemma4-31b
    # Qwen/Qwen3-8B -> qwen3-8b
    name = model_name.split("/")[-1].lower()
    name = name.replace("meta-", "").replace("-instruct", "").replace("-it", "")
    # Simplify common patterns
    for old, new in [("llama-3.1", "llama3.1"), ("gemma-4", "gemma4"),
                     ("qwen3", "qwen3")]:
        name = name.replace(old, new)
    # Remove extra hyphens
    name = re.sub(r'-+', '-', name).strip('-')
    return name


def _checkpoint_path(short_name: str, dataset: str) -> Path:
    suffix = f"_{dataset}" if dataset in ("wa", "bc") else ""
    return RESULTS_DIR / f"{short_name}_checkpoint{suffix}.json"


def save_checkpoint(ckpt_path: Path, predictions: list, completed_convs: int):
    """Save evaluation checkpoint for resuming."""
    data = {"completed_convs": completed_convs, "predictions": predictions}
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(ckpt_path)


def load_checkpoint(ckpt_path: Path) -> tuple[list, int]:
    """Load checkpoint. Returns (predictions, completed_convs)."""
    if not ckpt_path.exists():
        return [], 0
    with open(ckpt_path) as f:
        data = json.load(f)
    return data["predictions"], data["completed_convs"]


def main():
    args = parse_args()

    dataset_labels = {"iw": "IW (Fabricated)", "wa": "WebArena", "bc": "BrowseComp-Plus"}
    dataset_label = dataset_labels.get(args.dataset, args.dataset)
    short_name = model_short_name(args.model)

    print("=" * 60)
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Dataset: {dataset_label}")
    print(f"Max conversations: {args.max_convs}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    # Load conversations
    if args.dataset == "wa":
        val_path = DATA_DIR / "wa_conversations.jsonl"
    elif args.dataset == "bc":
        val_path = DATA_DIR / "bc_conversations.jsonl"
    else:
        val_path = DATA_DIR / "val_conversations.jsonl"

    conversations = []
    with open(val_path) as f:
        for line in f:
            conversations.append(json.loads(line))
            if len(conversations) >= args.max_convs:
                break
    print(f"\nLoaded {len(conversations)} {dataset_label} conversations", flush=True)

    # Resume from checkpoint if requested
    ckpt_name = short_name + ("_lora" if args.adapter else "")
    ckpt_path = _checkpoint_path(ckpt_name, args.dataset)
    start_conv = 0
    predictions = []

    if args.resume:
        predictions, start_conv = load_checkpoint(ckpt_path)
        if start_conv > 0:
            print(f"Resuming from conversation {start_conv} "
                  f"({len(predictions)} predictions loaded)", flush=True)

    # Rebuild counters from loaded predictions
    total = 0
    correct = 0
    pos_correct = {}
    pos_total = {}
    for p in predictions:
        total += 1
        correct += int(p["correct"])
        si = p["skill_position"]
        pos_total[si] = pos_total.get(si, 0) + 1
        pos_correct[si] = pos_correct.get(si, 0) + int(p["correct"])

    # Evaluate
    for ci, conv in enumerate(conversations):
        if ci < start_conv:
            continue

        msgs = conv["messages"]
        skill_idx = 0

        for mi in range(len(msgs)):
            if msgs[mi]["role"] != "assistant":
                continue

            gt_response = msgs[mi]["content"]
            gt_skill = extract_skill_from_response(gt_response)

            if gt_skill not in VALID_SKILLS:
                continue

            context = msgs[:mi]

            pred_response = generate_response(
                model, tokenizer, context,
                max_new_tokens=args.max_new_tokens)
            pred_skill = extract_skill_from_response(pred_response)

            is_correct = pred_skill == gt_skill
            correct += int(is_correct)
            total += 1

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
            save_checkpoint(ckpt_path, predictions, ci + 1)

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

    # Top errors
    print(f"\nTop 10 errors:")
    errors = [(p["ground_truth"], p["prediction"])
              for p in predictions if not p["correct"]]
    for (gt, pred), count in Counter(errors).most_common(10):
        print(f"  {gt:20s} -> {pred:20s} ({count} times)")

    # Edit distance
    conv_pred_seqs = {}
    conv_gt_seqs = {}
    for p in predictions:
        cid = p["conversation_id"]
        conv_pred_seqs.setdefault(cid, []).append(p["prediction"])
        conv_gt_seqs.setdefault(cid, []).append(p["ground_truth"])

    pred_seqs = [conv_pred_seqs[cid] for cid in conv_pred_seqs]
    gt_seqs = [conv_gt_seqs[cid] for cid in conv_gt_seqs]
    edit_dist = normalized_edit_distance(pred_seqs, gt_seqs)
    exact_match = sum(1 for p, g in zip(pred_seqs, gt_seqs) if p == g) / max(len(pred_seqs), 1)

    print(f"\nNormalized edit distance: {edit_dist:.4f}")
    print(f"Exact sequence match: {exact_match:.4f} ({sum(1 for p, g in zip(pred_seqs, gt_seqs) if p == g)}/{len(pred_seqs)})")

    # Save results
    suffix = f"_{args.dataset}" if args.dataset in ("wa", "bc") else ""
    if args.adapter:
        short_name += "_lora"
    results = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "overall_accuracy": overall_acc,
        "normalized_edit_distance": edit_dist,
        "exact_sequence_match": exact_match,
        "per_position_accuracy": {str(k): v for k, v in per_pos_acc.items()},
        "total_samples": total,
        "correct": correct,
        "num_conversations": len(conversations),
    }

    results_path = RESULTS_DIR / f"{short_name}_eval_metrics{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    preds_path = RESULTS_DIR / f"{short_name}_predictions{suffix}.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {preds_path}")

    # Clean up checkpoint after successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
