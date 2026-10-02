"""
Benchmark comparison across all 10 ML algorithms.

Reads every results/*.json produced by the 01_..10_ algorithm scripts,
builds a summary table, and renders matplotlib charts comparing accuracy,
precision/recall/f1, training time, and the confusion matrix of the
best-performing model. Also plots the majority-class baseline for context,
since every algorithm here is trained on the same weak feature set (see
data_preprocessing.py) and is expected to land close to that baseline.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preprocessing import load_and_preprocess_data, RESULTS_DIR

PLOTS_DIR = Path(__file__).parent / "plots"
SUMMARY_CSV = Path(__file__).parent / "benchmark_summary.csv"

COLORS = {
    "bar": "#4C72B0",
    "best": "#55A868",
    "baseline": "#C44E52",
    "grid": "#DDDDDD",
}


def load_results():
    records = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        records.append(data)
    if not records:
        raise FileNotFoundError(
            f"No result JSON files found in {RESULTS_DIR}. Run the 01_..10_ algorithm scripts first."
        )
    return records


def compute_majority_baseline():
    _, _, y_train, y_test, label_names = load_and_preprocess_data()
    majority_class = np.bincount(y_train).argmax()
    baseline_acc = (y_test == majority_class).mean()
    return baseline_acc, label_names[majority_class]


def build_summary_df(records):
    df = pd.DataFrame.from_records(
        [
            {
                "Algorithm": r["algorithm"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1 Score": r["f1_score"],
                "Train Time (s)": r["train_time_sec"],
            }
            for r in records
        ]
    ).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df


def plot_accuracy_comparison(df, baseline_acc, baseline_label):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [COLORS["best"] if i == 0 else COLORS["bar"] for i in range(len(df))]
    bars = ax.bar(df["Algorithm"], df["Accuracy"], color=colors, edgecolor="white")

    ax.axhline(baseline_acc, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
                label=f"Majority-class baseline ({baseline_label}) = {baseline_acc:.3f}")

    for bar, val in zip(bars, df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across 10 ML Algorithms\n(Alzheimer Dataset-Manifest Benchmark)")
    ax.set_ylim(0, max(df["Accuracy"].max() + 0.1, baseline_acc + 0.1))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_grid(df):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[metric], width=width, label=metric, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 / Accuracy by Algorithm")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_time(df):
    df_sorted = df.sort_values("Train Time (s)", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_sorted["Algorithm"], df_sorted["Train Time (s)"], color=COLORS["bar"])
    for bar, val in zip(bars, df_sorted["Train Time (s)"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.3f}s", va="center", fontsize=9)
    ax.set_xlabel("Training Time, incl. GridSearchCV (seconds)")
    ax.set_title("Training Time Comparison (log scale)")
    ax.set_xscale("log")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_training_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_confusion_matrix(records, df):
    best_name = df.iloc[0]["Algorithm"]
    best_record = next(r for r in records if r["algorithm"] == best_name)
    cm = np.array(best_record["confusion_matrix"])
    labels = best_record["label_names"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Best Model:\n{best_name}", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_best_model_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_results()
    df = build_summary_df(records)
    baseline_acc, baseline_label = compute_majority_baseline()

    df.to_csv(SUMMARY_CSV, index=False)

    plot_accuracy_comparison(df, baseline_acc, baseline_label)
    plot_metric_grid(df)
    plot_training_time(df)
    plot_best_confusion_matrix(records, df)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n===== BENCHMARK SUMMARY (sorted by accuracy) =====")
    print(df.to_string(index=False))
    print(f"\nMajority-class baseline accuracy ({baseline_label}): {baseline_acc:.4f}")
    print(f"Best model: {df.iloc[0]['Algorithm']}  (accuracy={df.iloc[0]['Accuracy']:.4f})")
    print(f"\nSummary CSV saved to: {SUMMARY_CSV}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
