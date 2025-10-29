#!/usr/bin/env python3
"""
Compare baseline vs new variants per disease — Full Summary (Accuracy, Precision, Recall, macro-averaged).

Generates 9 charts total (3 metrics x 3 modalities):
    modality1_accuracy.png, modality1_precision.png, modality1_recall.png,
    modality2_accuracy.png, modality2_precision.png, modality2_recall.png,
    shared_accuracy.png, shared_precision.png, shared_recall.png

Each chart has 4 subplots (one per disease).

Also outputs:
  - metric_summary_macro.csv (aggregated mean ± SD and Δ vs baseline)
  - metric_per_run.csv (raw per-run computed metrics)

Usage:
  python compare_pipelines_summary_full.py \
      --file-a mdl4microbiome-runs.csv \
      --file-b mdl4-1-runs.csv \
      --out images/accuracy/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Full summary: accuracy, precision, recall (macro-averaged)."
    )
    p.add_argument(
        "--file-a", required=True, help="CSV file containing baseline results"
    )
    p.add_argument(
        "--file-b", required=True, help="CSV file containing new kinds results"
    )
    p.add_argument(
        "--out",
        default="images/accuracy",
        help="Output directory (default: images/accuracy)",
    )
    return p.parse_args()


# ----------------------------- Utilities -----------------------------
ORDERED_KINDS = ["baseline", "meta-only", "pretrain-only", "meta-pretrain"]


def ensure_out(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sort_kinds(kinds):
    ordered = [k for k in ORDERED_KINDS if k in kinds]
    extras = [k for k in kinds if k not in ORDERED_KINDS]
    return ordered + extras


# ----------------------------- Metric Computation -----------------------------
def compute_per_run_metrics(df):
    """Compute per-run accuracy, precision, recall for all modalities."""
    metrics = []
    for modality in ["modal1", "modal2", "shared"]:
        tp = pd.to_numeric(df[f"{modality}_TP"], errors="coerce")
        tn = pd.to_numeric(df[f"{modality}_TN"], errors="coerce")
        fp = pd.to_numeric(df[f"{modality}_FP"], errors="coerce")
        fn = pd.to_numeric(df[f"{modality}_FN"], errors="coerce")

        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        for metric_name, values in zip(
            ["accuracy", "precision", "recall"], [acc, prec, rec]
        ):
            for i in range(len(df)):
                metrics.append(
                    {
                        "disease": df.loc[i, "disease"],
                        "kind": df.loc[i, "kind"],
                        "modality": modality.replace("modal", "modality"),
                        "metric": metric_name,
                        "value": values.iloc[i],
                    }
                )

    return pd.DataFrame(metrics)


# ----------------------------- Macro aggregation -----------------------------
def aggregate_macro(per_run_df):
    grouped = (
        per_run_df.groupby(["disease", "kind", "modality", "metric"])
        .agg(mean=("value", "mean"), std=("value", "std"))
        .reset_index()
    )
    return grouped


def compute_deltas(summary_df):
    """Compute Δ vs baseline for each disease, modality, metric."""
    results = []
    for (disease, modality, metric), group in summary_df.groupby(
        ["disease", "modality", "metric"]
    ):
        if "baseline" not in group["kind"].values:
            continue
        base = group[group["kind"] == "baseline"].iloc[0]
        for _, row in group.iterrows():
            delta = row["mean"] - base["mean"]
            results.append(
                {
                    "disease": disease,
                    "kind": row["kind"],
                    "modality": modality,
                    "metric": metric,
                    "mean": row["mean"],
                    "std": row["std"],
                    "delta_vs_baseline": delta,
                }
            )
    return pd.DataFrame(results)


# ----------------------------- Plot -----------------------------
def plot_full_summary(df_deltas, out_dir: Path):
    modalities = ["modality1", "modality2", "shared"]
    metrics = ["accuracy", "precision", "recall"]
    diseases = sorted(df_deltas["disease"].unique())

    for metric in metrics:
        metric_dir = out_dir / metric
        ensure_out(metric_dir)

        for modality in modalities:
            sub = df_deltas[
                (df_deltas["modality"] == modality) & (df_deltas["metric"] == metric)
            ]
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()

            for idx, disease in enumerate(diseases):
                ax = axes[idx]
                dsub = sub[sub["disease"] == disease]
                if dsub.empty:
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5, f"No data for {disease}", ha="center", va="center"
                    )
                    continue

                kinds = sort_kinds(list(dsub["kind"].unique()))
                plot_data = dsub.set_index("kind").reindex(kinds)
                means, stds, deltas = (
                    plot_data["mean"],
                    plot_data["std"],
                    plot_data["delta_vs_baseline"],
                )

                x = np.arange(len(kinds))
                ax.bar(
                    x,
                    means,
                    yerr=stds,
                    capsize=4,
                    alpha=0.8,
                    color="skyblue",
                    edgecolor="black",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(kinds)
                ax.set_title(f"{disease}")
                ax.set_ylabel(metric.capitalize())

                # Ensure visual range (at least 0.3)
                val_min, val_max = (means - stds).min(), (means + stds).max()
                if val_max - val_min < 0.3:
                    mid = (val_max + val_min) / 2
                    val_min, val_max = mid - 0.15, mid + 0.15
                ax.set_ylim(max(0.0, val_min), min(1.0, val_max))

                # Annotate bars
                for i, k in enumerate(kinds):
                    if pd.isna(means.loc[k]):
                        continue
                    m, s, d = means.loc[k], stds.loc[k], deltas.loc[k]
                    label = f"{d:+.3f} ({m:.3f})" if k != "baseline" else f"({m:.3f})"
                    ax.text(i, m + s + 0.005, label, ha="center", fontsize=8)

            for j in range(len(diseases), 4):
                axes[j].axis("off")

            plt.suptitle(
                f"{modality} — {metric.capitalize()} (mean ± SD)",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            out_file = metric_dir / f"{modality}_{metric}.png"
            plt.savefig(out_file, dpi=200)
            plt.close(fig)
            print(f"Saved {out_file}")


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out)
    ensure_out(out_dir)

    # Load CSVs
    dfA = pd.read_csv(args.file_a)
    dfB = pd.read_csv(args.file_b)
    dfA["kind"] = (
        dfA["kind"]
        .str.strip()
        .str.lower()
        .replace({"base": "baseline", "control": "baseline"})
    )
    dfB["kind"] = (
        dfB["kind"]
        .str.strip()
        .str.lower()
        .replace({"base": "baseline", "control": "baseline"})
    )
    df = pd.concat([dfA, dfB], ignore_index=True)

    # Compute per-run metrics
    per_run = compute_per_run_metrics(df)
    per_run_csv = out_dir / "metric_per_run.csv"
    per_run.to_csv(per_run_csv, index=False)
    print(f"Saved per-run metrics: {per_run_csv}")

    # Aggregate and compute deltas
    summary = aggregate_macro(per_run)
    deltas = compute_deltas(summary)

    summary_csv = out_dir / "metric_summary_macro.csv"
    deltas.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    # Plot grouped results into subdirectories
    plot_full_summary(deltas, out_dir)

    print(f"Done. Charts and summaries saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
