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
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

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
    "Qwen3-8B (GRPO)": "#C0392B",
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
    "Qwen3-8B (GRPO)": "qwen3-8b_lora",
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
    path = PREDICTIONS_DIR / f"{short_name}_predictions{suffix}.json"
    if not path.exists():
        # Fallback to legacy flat location
        path = RESULTS_DIR / f"{short_name}_predictions{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(short_name, dataset):
    suffix = f"_{dataset}" if dataset in ("wa", "bc") else ""
    path = METRICS_DIR / f"{short_name}_eval_metrics{suffix}.json"
    if not path.exists():
        # Fallback to legacy flat location
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
        "iw": METRICS_DIR / "baseline_metrics_iw.json",
        "wa": METRICS_DIR / "baseline_metrics_wa.json",
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
    pipeline_file = METRICS_DIR / "pipeline_metrics.json"
    method_labels = {
        "frequency": "Frequency",
        "skillmd": "SKILL.md",
        "awm": "AWM",
        "transformer": "Transformer",
        "qwen3_lora": "Qwen3-8B GRPO",
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
                    label="Qwen3-8B GRPO" if ds_i == 0 else None,
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


# ── Panel 3: Training Method Comparison (zero-shot vs GRPO) ─────────

def plot_training_comparison(ax):
    """Show zero-shot vs GRPO, highlighting GRPO's degradation."""
    datasets = ["iw", "wa", "bc"]
    ds_labels = ["IW", "WebArena", "BrowseComp+"]

    zs_accs, grpo_accs = [], []
    for ds in datasets:
        zs = load_metrics("qwen3-8b", ds)
        grpo = load_metrics("qwen3-8b_lora", ds)
        zs_accs.append(zs["overall_accuracy"] if zs else 0)
        grpo_accs.append(grpo["overall_accuracy"] if grpo else 0)

    x = np.arange(len(datasets))
    bar_width = 0.3

    bars_zs = ax.bar(x - bar_width / 2, zs_accs, bar_width,
                      label="Qwen3-8B (zero-shot)", color="#7EB0D5",
                      edgecolor="white")
    bars_grpo = ax.bar(x + bar_width / 2, grpo_accs, bar_width,
                        label="Qwen3-8B (GRPO)", color="#C0392B",
                        edgecolor="white")

    # Add difference annotations -- red for degradation, green for improvement
    for j in range(len(datasets)):
        if zs_accs[j] > 0 and grpo_accs[j] > 0:
            diff = grpo_accs[j] - zs_accs[j]
            sign = "+" if diff > 0 else ""
            color = "#2ca02c" if diff > 0 else "#d62728"
            mid_y = max(zs_accs[j], grpo_accs[j]) + 0.03
            ax.annotate(
                f"{sign}{diff:.1%}",
                xy=(x[j], mid_y), ha="center",
                fontsize=9, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.8, linewidth=1.2),
            )
        for bars, vals in [(bars_zs, zs_accs), (bars_grpo, grpo_accs)]:
            if vals[j] > 0:
                ax.text(bars[j].get_x() + bars[j].get_width() / 2,
                        bars[j].get_height() + 0.005,
                        f"{vals[j]:.1%}", ha="center", va="bottom",
                        fontsize=7)
            elif vals[j] == 0:
                ax.text(bars[j].get_x() + bars[j].get_width() / 2, 0.02,
                        "---", ha="center", fontsize=7, color="gray")

    ax.set_ylabel("Overall Accuracy")
    ax.set_title("(c) Training Method Comparison (zero-shot vs GRPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel 4: Mind2Web Teacher-Forcing Results ──────────────────────

def _load_mind2web_checkpoint(path):
    """Load a Mind2Web checkpoint and compute summary metrics."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    results = d.get("results", [])
    n = len(results)
    if n == 0:
        return None
    completed = sum(1 for r in results if r.get("completed", False))
    avg_reward = sum(r.get("reward", 0) for r in results) / n
    total_steps, skill_correct = 0, 0
    for r in results:
        for s in r.get("steps", []):
            total_steps += 1
            if s.get("skill_correct", False):
                skill_correct += 1
    skill_acc = skill_correct / total_steps if total_steps else 0
    return {
        "n": n, "tcr": completed / n,
        "avg_reward": avg_reward, "skill_acc": skill_acc,
    }


def plot_e2e_benchmarks(ax):
    """Show Mind2Web teacher-forcing TCR & reward for zero-shot vs GRPO."""
    checkpoint_dir = RESULTS_DIR

    configs = [
        ("Zero-shot\n(test_task)", checkpoint_dir / "qwen3-8b_mind2web_test_task_checkpoint.json"),
        ("GRPO\n(test_task)", checkpoint_dir / "qwen3-8b_lora_mind2web_test_task_checkpoint.json"),
        ("GRPO\n(test_domain)", checkpoint_dir / "qwen3-8b_lora_mind2web_test_domain_checkpoint.json"),
    ]

    labels, tcrs, rewards = [], [], []
    for label, path in configs:
        m = _load_mind2web_checkpoint(path)
        if m:
            labels.append(f"{label}\n(n={m['n']})")
            tcrs.append(m["tcr"])
            rewards.append(m["avg_reward"])

    if not labels:
        ax.text(0.5, 0.5, "No Mind2Web checkpoints found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        ax.set_title("(d) Mind2Web Teacher-Forcing")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    x = np.arange(len(labels))
    bar_width = 0.35

    bars_tcr = ax.bar(x - bar_width / 2, tcrs, bar_width,
                       label="Task Completion Rate", color="#55A868",
                       edgecolor="white")
    bars_rwd = ax.bar(x + bar_width / 2, rewards, bar_width,
                       label="Avg Reward", color="#8172B3",
                       edgecolor="white")

    for bars, vals in [(bars_tcr, tcrs), (bars_rwd, rewards)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("(d) Mind2Web Teacher-Forcing (zero-shot vs GRPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_skill_prediction(axes[0, 0])
    plot_baselines(axes[0, 1])
    plot_training_comparison(axes[1, 0])
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
