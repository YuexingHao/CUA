"""
LoRA Fine-Tuning Qwen3-8B on Multi-Turn CUA Conversations.

This script fine-tunes Qwen3-8B using LoRA (Low-Rank Adaptation)
on realistic user-agent conversation data where the agent learns
to reason about, execute, and adapt skills in multi-turn workflows.

=== HOW LORA WORKS (for beginners) ===

Normal fine-tuning: Update ALL 8 billion weights → needs ~60GB GPU RAM
LoRA fine-tuning:   Freeze all weights, add tiny trainable adapters → ~18GB GPU RAM

LoRA adds small "adapter" matrices to the attention layers:

    Original layer:     y = W × x           (W is huge, e.g. 4096×4096)
    With LoRA:          y = W × x + (A × B) × x
                            ↑frozen   ↑trainable (A=4096×r, B=r×4096)

    r = "rank" = how many dimensions the adapter uses (we use r=16)
    Fewer dimensions = fewer parameters = faster training

Key hyperparameters:
    - r (rank): 8-64, higher = more capacity but slower
    - lora_alpha: scaling factor, usually = r or 2×r
    - target_modules: which layers get adapters (attention layers)
    - learning_rate: usually 1e-4 to 2e-4 for LoRA

=== USAGE ===

    # Step 1: Generate conversation data
    python data/generate_conversations.py --num 1000

    # Step 2: Fine-tune
    python -m interaskill.finetune_qwen

    # Or submit as SLURM job:
    sbatch run_finetune.sh
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Configuration ────────────────────────────────────────────────────

# Model — change this to try different models
MODEL_NAME = "Qwen/Qwen3-8B"  # HuggingFace model ID

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "qwen3_lora"

# ── LoRA Configuration ──────────────────────────────────────────────
#
# These are the key knobs you can turn:
#
#   r (rank):
#     - Controls adapter size. r=16 means each adapter is 16-dimensional.
#     - Higher r = more capacity, but more memory & slower.
#     - Start with 16, increase to 32 or 64 if underfitting.
#
#   lora_alpha:
#     - Scaling factor. The adapter output is multiplied by (alpha / r).
#     - Rule of thumb: set alpha = r (so scaling = 1.0).
#     - Higher alpha = stronger adapter effect.
#
#   target_modules:
#     - Which layers get LoRA adapters.
#     - For Qwen: attention projections (q, k, v, o) are most important.
#     - Can also add "gate_proj", "up_proj", "down_proj" for MLP layers.
#
#   lora_dropout:
#     - Dropout on adapter outputs. Prevents overfitting.
#     - 0.05-0.1 is typical.

LORA_CONFIG = LoraConfig(
    r=16,                              # Rank: adapter inner dimension
    lora_alpha=16,                     # Scaling factor (alpha/r = 1.0)
    target_modules=[                   # Which layers get adapters
        "q_proj", "k_proj",            #   Query and Key attention
        "v_proj", "o_proj",            #   Value and Output attention
    ],
    lora_dropout=0.05,                 # Dropout for regularization
    bias="none",                       # Don't train bias terms
    task_type="CAUSAL_LM",            # We're doing text generation
)

# ── Quantization (4-bit) ────────────────────────────────────────────
#
# Loading 8B params in full precision = 32GB.
# 4-bit quantization reduces this to ~5GB, with minimal quality loss.
# Combined with LoRA, this is called "QLoRA".

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,                          # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",                  # NormalFloat4 (best quality)
    bnb_4bit_compute_dtype=torch.bfloat16,      # Compute in bfloat16
    bnb_4bit_use_double_quant=True,             # Double quantization saves more memory
)

# ── Training Hyperparameters ─────────────────────────────────────────

TRAINING_ARGS = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,                         # 3 passes through data
    per_device_train_batch_size=2,              # Smaller batch for longer sequences
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,              # Effective batch = 2×8 = 16
    learning_rate=2e-4,                         # LoRA learning rate
    weight_decay=0.01,                          # L2 regularization
    warmup_ratio=0.05,                          # 5% warmup
    lr_scheduler_type="cosine",                 # Cosine learning rate decay
    logging_steps=10,                           # Log every 10 steps
    eval_strategy="steps",                      # Evaluate during training
    eval_steps=50,                              # Evaluate every 50 steps
    save_strategy="steps",                      # Save checkpoints
    save_steps=100,                             # Save every 100 steps
    save_total_limit=2,                         # Keep only 2 best checkpoints
    bf16=True,                                  # Use bfloat16 training
    gradient_checkpointing=True,                # Trade compute for memory
    max_grad_norm=0.3,                          # Gradient clipping
    report_to="none",                           # Disable wandb/tensorboard
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

# Max sequence length for tokenization (conversations are longer than single prompts)
MAX_SEQ_LENGTH = 2048


def format_conversation(example, tokenizer):
    """Format a multi-turn conversation using the chat template.

    Each conversation has a 'messages' field with system, user, and
    assistant turns. The tokenizer's chat template handles the formatting.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print("=" * 60)
    print("LoRA Fine-Tuning: Qwen3-8B on Multi-Turn Conversations")
    print("=" * 60)

    # ── Step 1: Load Data ────────────────────────────────────────
    print("\n[1/5] Loading conversation data...")
    train_path = str(DATA_DIR / "train_conversations.jsonl")
    val_path = str(DATA_DIR / "val_conversations.jsonl")

    if not Path(train_path).exists():
        print("  Data not found! Generating conversations first...")
        import subprocess
        subprocess.run(["python", "data/generate_conversations.py", "--num", "1000"],
                       check=True)

    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path,
    })
    print(f"  Train: {len(dataset['train'])} conversations")
    print(f"  Val:   {len(dataset['validation'])} conversations")

    # Print conversation length stats
    train_turns = [len(ex["messages"]) for ex in dataset["train"]]
    print(f"  Turns per conversation: min={min(train_turns)}, "
          f"max={max(train_turns)}, mean={sum(train_turns)/len(train_turns):.1f}")

    # ── Step 2: Load Tokenizer ───────────────────────────────────
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Format conversations using chat template
    dataset = dataset.map(
        lambda ex: format_conversation(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    # Check token lengths
    sample_tokens = tokenizer(dataset["train"][0]["text"], return_tensors="pt")
    print(f"  Example conversation: {sample_tokens['input_ids'].shape[1]} tokens")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Preview (first 400 chars):\n  {dataset['train'][0]['text'][:400]}...")

    # ── Step 3: Load Model with Quantization ─────────────────────
    print("\n[3/5] Loading Qwen3-8B with 4-bit quantization...")
    print("  This downloads ~5GB on first run (cached after that)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  Model loaded on: {next(model.parameters()).device}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 4: Apply LoRA ───────────────────────────────────────
    print("\n[4/5] Applying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Total parameters:     {total:>14,}")
    print(f"  Trainable (LoRA):     {trainable:>14,} ({100*trainable/total:.2f}%)")
    print(f"  Frozen:               {total-trainable:>14,} ({100*(total-trainable)/total:.2f}%)")

    # ── Step 5: Train ────────────────────────────────────────────
    print("\n[5/5] Starting LoRA fine-tuning on multi-turn conversations...")
    print(f"  Epochs: {TRAINING_ARGS.num_train_epochs}")
    eff_batch = TRAINING_ARGS.per_device_train_batch_size * TRAINING_ARGS.gradient_accumulation_steps
    print(f"  Batch: {TRAINING_ARGS.per_device_train_batch_size} "
          f"× {TRAINING_ARGS.gradient_accumulation_steps} accum = {eff_batch} effective")
    print(f"  Learning rate: {TRAINING_ARGS.learning_rate}")
    print(f"  LoRA: r={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        args=TRAINING_ARGS,
    )

    # Train
    train_result = trainer.train()

    # Results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training loss:    {train_result.training_loss:.4f}")
    print(f"  Training time:    {train_result.metrics['train_runtime']:.0f} seconds")
    print(f"  Samples/second:   {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"  GPU peak memory:  {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # Save LoRA adapter
    adapter_path = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  LoRA adapter saved to: {adapter_path}")
    print(f"  (Only adapter weights — ~34MB, not the full 8B model)")

    # Save training metrics
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Evaluate
    print("\n  Running evaluation...")
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    eval_path = OUTPUT_DIR / "eval_metrics.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"  Eval metrics saved to: {eval_path}")


if __name__ == "__main__":
    main()
