"""
Compare model accuracy across datasets with 95% bootstrap CIs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")

# Models to include (display name -> short_name used in filenames)
MODELS = {
    "Qwen3-8B (zero-shot)": "qwen3-8b",
    "Qwen3-8B LoRA": "qwen3-8b_lora",
    "Llama-3.1-70B": "llama3.1-70b",
    "OLMo-3-7B": "olmo-3-1025-7b",
}

DATASETS = {
    "iw": "IW",
    "wa": "WebArena",
    "bc": "BrowseComp+",
}


def load_predictions(short_name, dataset):
    suffix = f"_{dataset}" if dataset in ("wa", "bc") else ""
    path = RESULTS_DIR / f"{short_name}_predictions{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def per_conversation_accuracy(predictions):
    """Compute accuracy per conversation, returns array of per-conv accuracies."""
    conv_correct = defaultdict(int)
    conv_total = defaultdict(int)
    for p in predictions:
        cid = p["conversation_id"]
        conv_total[cid] += 1
        conv_correct[cid] += int(p["correct"])
    return np.array([conv_correct[c] / conv_total[c] for c in conv_total])


def bootstrap_ci(accs, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for mean accuracy."""
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(accs, size=len(accs), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return accs.mean(), lo, hi


def main():
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(MODELS.keys())
    dataset_keys = list(DATASETS.keys())
    n_models = len(model_names)
    n_datasets = len(dataset_keys)

    bar_width = 0.18
    x = np.arange(n_datasets)

    colors = ["#7EB0D5", "#4C72B0", "#DD8452", "#55A868"]

    for i, model_name in enumerate(model_names):
        short_name = MODELS[model_name]
        means, ci_lo, ci_hi = [], [], []

        for ds in dataset_keys:
            preds = load_predictions(short_name, ds)
            if preds is None:
                means.append(0)
                ci_lo.append(0)
                ci_hi.append(0)
                continue

            accs = per_conversation_accuracy(preds)
            mean, lo, hi = bootstrap_ci(accs)
            means.append(mean)
            ci_lo.append(mean - lo)
            ci_hi.append(hi - mean)

        means = np.array(means)
        errs = np.array([ci_lo, ci_hi])

        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                       yerr=errs, capsize=4,
                       label=model_name, color=colors[i],
                       edgecolor="white", linewidth=0.5,
                       error_kw={"linewidth": 1.5})

        # Add value labels on bars
        for j, (bar, m) in enumerate(zip(bars, means)):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errs[1, j] + 0.01,
                        f"{m:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "N/A", ha="center", va="bottom", fontsize=8,
                        color="gray", fontstyle="italic")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Mean Per-Conversation Accuracy", fontsize=12)
    ax.set_title("Skill Prediction: Model Comparison (95% Bootstrap CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASETS[ds] for ds in dataset_keys], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
