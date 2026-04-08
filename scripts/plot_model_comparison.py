"""
Comprehensive model comparison visualization across all benchmarks.

Generates:
  1. Skill prediction accuracy across IW / WebArena / BrowseComp+ (existing)
  2. Baseline comparison table (SKILL.md / AWM / Frequency / Pipeline)
  3. Placeholder panels for WebShop, Mind2Web, and Multimodal (when results arrive)

Run: python scripts/plot_model_comparison.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")

# ── Style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "Qwen3-8B (zero-shot)": "#7EB0D5",
    "Qwen3-8B LoRA": "#4C72B0",
    "Llama-3.1-70B": "#DD8452",
    "OLMo-3-7B": "#55A868",
    "Gemma-4-31B": "#C44E52",
    "DeepSeek-R1-32B": "#937860",
    "Qwen3-VL-8B": "#8172B3",
    "CLIP zero-shot": "#CCB974",
}

# ── Models & Datasets ────────────────────────────────────────────────

SKILL_PRED_MODELS = {
    "Qwen3-8B (zero-shot)": "qwen3-8b",
    "Qwen3-8B LoRA": "qwen3-8b_lora",
    "Llama-3.1-70B": "llama3.1-70b",
    "OLMo-3-7B": "olmo-3-1025-7b",
    "Gemma-4-31B": "gemma4-31b",
    "DeepSeek-R1-32B": "deepseek-r1-distill-qwen-32b",
}

SKILL_PRED_DATASETS = {
    "iw": "IW (Enterprise)",
    "wa": "WebArena",
    "bc": "BrowseComp+",
}

# ── Helper Functions ─────────────────────────────────────────────────

def load_predictions(short_name, dataset):
    suffix = f"_{dataset}" if dataset in ("wa", "bc") else ""
    path = RESULTS_DIR / f"{short_name}_predictions{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(short_name, dataset):
    suffix = f"_{dataset}" if dataset in ("wa", "bc") else ""
    path = RESULTS_DIR / f"{short_name}_eval_metrics{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def per_conversation_accuracy(predictions):
    conv_correct = defaultdict(int)
    conv_total = defaultdict(int)
    for p in predictions:
        cid = p["conversation_id"]
        conv_total[cid] += 1
        conv_correct[cid] += int(p["correct"])
    return np.array([conv_correct[c] / conv_total[c] for c in conv_total])


def bootstrap_ci(accs, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(accs, size=len(accs), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return float(accs.mean()), float(lo), float(hi)


# ── Panel 1: Skill Prediction Accuracy ───────────────────────────────

def plot_skill_prediction(ax):
    """Bar chart: model accuracy across IW/WebArena/BrowseComp+ with 95% CI."""
    model_names = list(SKILL_PRED_MODELS.keys())
    dataset_keys = list(SKILL_PRED_DATASETS.keys())
    n_models = len(model_names)
    n_datasets = len(dataset_keys)

    bar_width = 0.12
    x = np.arange(n_datasets)

    for i, model_name in enumerate(model_names):
        short_name = SKILL_PRED_MODELS[model_name]
        means, ci_lo, ci_hi = [], [], []

        for ds in dataset_keys:
            preds = load_predictions(short_name, ds)
            if preds is None:
                # Fallback: try loading from metrics file (no predictions file)
                metrics = load_metrics(short_name, ds)
                if metrics:
                    acc = metrics["overall_accuracy"]
                    means.append(acc)
                    ci_lo.append(0)
                    ci_hi.append(0)
                else:
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
        color = COLORS.get(model_name, f"C{i}")
        offset = (i - (n_models - 1) / 2) * bar_width

        bars = ax.bar(x + offset, means, bar_width,
                       yerr=errs, capsize=2,
                       label=model_name, color=color,
                       edgecolor="white", linewidth=0.5,
                       error_kw={"linewidth": 1.0})

        for j, (bar, m) in enumerate(zip(bars, means)):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + errs[1, j] + 0.01,
                        f"{m:.0%}", ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "—", ha="center", va="bottom",
                        fontsize=6, color="gray")

    ax.set_ylabel("Per-Conv Accuracy")
    ax.set_title("(a) Skill Prediction Accuracy (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([SKILL_PRED_DATASETS[ds] for ds in dataset_keys])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel 2: Baselines Comparison ────────────────────────────────────

def plot_baselines(ax):
    """Grouped bar chart comparing baselines on IW and WebArena."""
    baseline_files = {
        "iw": RESULTS_DIR / "baseline_metrics_iw.json",
        "wa": RESULTS_DIR / "baseline_metrics_wa.json",
    }

    baselines_data = {}
    for ds, path in baseline_files.items():
        if path.exists():
            with open(path) as f:
                baselines_data[ds] = json.load(f)

    if not baselines_data:
        ax.text(0.5, 0.5, "No baseline results yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("(b) Baselines Comparison")
        return

    # Methods to compare
    methods = ["frequency", "skillmd", "awm"]
    # Add pipeline + LLM results from metrics.json
    pipeline_file = RESULTS_DIR / "metrics.json"
    method_labels = {
        "frequency": "Frequency",
        "skillmd": "SKILL.md",
        "awm": "AWM",
        "transformer": "Transformer",
        "qwen3_lora": "Qwen3-8B LoRA",
    }

    datasets = ["iw", "wa"]
    ds_labels = {"iw": "IW", "wa": "WebArena"}
    bar_width = 0.13
    x = np.arange(len(datasets))

    # Collect edit distances (lower = better, so we plot 1 - ed for readability)
    method_colors = ["#C0C0C0", "#E8A87C", "#85C1E9", "#82E0AA", "#4C72B0"]

    for i, method in enumerate(methods):
        eds = []
        for ds in datasets:
            if ds in baselines_data and method in baselines_data[ds]:
                ed = baselines_data[ds][method].get("edit_distance", 1.0)
                eds.append(1.0 - ed)  # convert to "accuracy" (1 - edit_dist)
            else:
                eds.append(0)

        offset = (i - 2) * bar_width
        bars = ax.bar(x + offset, eds, bar_width,
                       label=method_labels.get(method, method),
                       color=method_colors[i],
                       edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, eds):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    # Add Transformer pipeline result
    if pipeline_file.exists():
        with open(pipeline_file) as f:
            pipeline = json.load(f)
        if "phase3_transformer" in pipeline:
            ed = pipeline["phase3_transformer"].get("normalized_edit_distance", 1.0)
            offset = (3 - 2) * bar_width
            ax.bar(x[0] + offset, 1 - ed, bar_width,
                    label="Transformer", color=method_colors[3],
                    edgecolor="white", linewidth=0.5)

    # Add Qwen3-8B LoRA
    for ds_i, ds in enumerate(datasets):
        metrics = load_metrics("qwen3-8b_lora", ds)
        if metrics:
            ed = metrics.get("normalized_edit_distance", 1.0)
            offset = (4 - 2) * bar_width
            ax.bar(x[ds_i] + offset, 1 - ed, bar_width,
                    label="Qwen3-8B LoRA" if ds_i == 0 else None,
                    color=method_colors[4],
                    edgecolor="white", linewidth=0.5)

    ax.set_ylabel("1 - Edit Distance")
    ax.set_title("(b) Baselines vs Learned (lower edit dist = better)")
    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels[ds] for ds in datasets])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel 3: LoRA Improvement (zero-shot vs fine-tuned) ──────────────

def plot_lora_improvement(ax):
    """Show the impact of LoRA fine-tuning across datasets."""
    datasets = ["iw", "wa", "bc"]
    ds_labels = ["IW", "WebArena", "BrowseComp+"]

    zs_accs, lora_accs = [], []
    for ds in datasets:
        zs = load_metrics("qwen3-8b", ds)
        lora = load_metrics("qwen3-8b_lora", ds)
        zs_accs.append(zs["overall_accuracy"] if zs else 0)
        lora_accs.append(lora["overall_accuracy"] if lora else 0)

    x = np.arange(len(datasets))
    bar_width = 0.3

    bars_zs = ax.bar(x - bar_width / 2, zs_accs, bar_width,
                      label="Qwen3-8B (zero-shot)", color="#7EB0D5",
                      edgecolor="white")
    bars_lora = ax.bar(x + bar_width / 2, lora_accs, bar_width,
                        label="Qwen3-8B LoRA", color="#4C72B0",
                        edgecolor="white")

    # Add improvement arrows
    for j in range(len(datasets)):
        if zs_accs[j] > 0 and lora_accs[j] > 0:
            diff = lora_accs[j] - zs_accs[j]
            sign = "+" if diff > 0 else ""
            color = "#2ca02c" if diff > 0 else "#d62728"
            mid_y = max(zs_accs[j], lora_accs[j]) + 0.03
            ax.text(x[j], mid_y, f"{sign}{diff:.0%}",
                    ha="center", fontsize=8, fontweight="bold", color=color)
        for bars, vals in [(bars_zs, zs_accs), (bars_lora, lora_accs)]:
            if vals[j] > 0:
                ax.text(bars[j].get_x() + bars[j].get_width() / 2,
                        bars[j].get_height() + 0.005,
                        f"{vals[j]:.0%}", ha="center", va="bottom",
                        fontsize=7)
            elif vals[j] == 0:
                ax.text(bars[j].get_x() + bars[j].get_width() / 2, 0.02,
                        "—", ha="center", fontsize=7, color="gray")

    ax.set_ylabel("Overall Accuracy")
    ax.set_title("(c) LoRA Fine-Tuning Impact")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel 4: End-to-End Benchmarks (WebShop / Mind2Web / Multimodal) ─

def plot_e2e_benchmarks(ax):
    """Show end-to-end results when available, placeholder otherwise."""
    benchmarks = []

    # Check for WebShop results
    for pattern in ["*webshop_metrics.json"]:
        for f in RESULTS_DIR.glob(pattern):
            with open(f) as fh:
                d = json.load(fh)
            benchmarks.append({
                "name": f"WebShop\n{d.get('model','?').split('/')[-1][:15]}",
                "tcr": d.get("level1_tcr", 0),
                "reward": d.get("average_reward", 0),
                "type": "webshop",
            })

    # Check for Mind2Web results
    for pattern in ["*mind2web_metrics*.json"]:
        for f in RESULTS_DIR.glob(pattern):
            with open(f) as fh:
                d = json.load(fh)
            benchmarks.append({
                "name": f"Mind2Web\n{d.get('model','?').split('/')[-1][:15]}",
                "tcr": d.get("level1_tcr", 0),
                "skill_acc": d.get("skill_prediction_accuracy", 0),
                "type": "mind2web",
            })

    # Check for Multimodal results
    for pattern in ["*vlm_metrics.json", "*clip_zeroshot_metrics.json"]:
        for f in RESULTS_DIR.glob(pattern):
            with open(f) as fh:
                d = json.load(fh)
            benchmarks.append({
                "name": f"Multimodal\n{d.get('mode','?')}",
                "skill_acc": d.get("skill_prediction_accuracy", 0),
                "type": "multimodal",
            })

    if not benchmarks:
        # Placeholder with expected experiments
        expected = [
            "WebShop\n(pending)", "Mind2Web\n(pending)",
            "Qwen3-VL\n(pending)", "CLIP\n(pending)",
        ]
        x = np.arange(len(expected))
        ax.bar(x, [0] * len(expected), 0.5, color="#E0E0E0", edgecolor="white")
        for i, label in enumerate(expected):
            ax.text(i, 0.05, label, ha="center", va="bottom",
                    fontsize=8, color="gray", fontstyle="italic")
        ax.set_title("(d) End-to-End & Multimodal (jobs running)")
    else:
        x = np.arange(len(benchmarks))
        values = [b.get("tcr", b.get("skill_acc", 0)) for b in benchmarks]
        type_colors = {
            "webshop": "#DD8452",
            "mind2web": "#55A868",
            "multimodal": "#8172B3",
        }
        colors = [type_colors.get(b["type"], "#999") for b in benchmarks]

        bars = ax.bar(x, values, 0.6, color=colors, edgecolor="white")
        for bar, v, b in zip(bars, values, benchmarks):
            metric = "TCR" if "tcr" in b else "Skill Acc"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([b["name"] for b in benchmarks], fontsize=7)
        ax.set_title("(d) End-to-End & Multimodal Results")

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_skill_prediction(axes[0, 0])
    plot_baselines(axes[0, 1])
    plot_lora_improvement(axes[1, 0])
    plot_e2e_benchmarks(axes[1, 1])

    fig.suptitle("InteraSkill: Comprehensive Model Comparison",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    out_path = figures_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()

    # Also generate a simple single-panel version for quick reference
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_skill_prediction(ax2)
    ax2.set_title("Skill Prediction: All Models (95% Bootstrap CI)", fontsize=14)
    out2 = figures_dir / "model_comparison_simple.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out2}")
    plt.close()


if __name__ == "__main__":
    main()
