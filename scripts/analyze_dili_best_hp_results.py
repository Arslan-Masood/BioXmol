#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Plotting removed per request


# Example filename:
# DILI_Soft_Clip_with_Frozen_Teacher_pretrained_seed_0_downstream_seed_0_fold_1_best_hp.json
RESULTS_FILENAME_RE = re.compile(
    r"DILI_(?P<model_type>[^_]+(?:_[^_]+)*)_pretrained_seed_(?P<pretrained_seed>\d+)_downstream_seed_(?P<downstream_seed>\d+)_fold_(?P<fold>\d+)_best_hp\.json$"
)


@dataclass(frozen=True)
class ParsedMeta:
    model_type: str
    pretrained_seed: int
    downstream_seed: int
    fold: int


def parse_filename(path: Path) -> ParsedMeta:
    m = RESULTS_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path}")
    return ParsedMeta(
        model_type=m.group("model_type"),
        pretrained_seed=int(m.group("pretrained_seed")),
        downstream_seed=int(m.group("downstream_seed")),
        fold=int(m.group("fold")),
    )


def load_results(results_dir: Path) -> pd.DataFrame:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    rows: List[Dict] = []
    files: List[Path] = sorted(
        results_dir.rglob("DILI_*_pretrained_seed_*_downstream_seed_*_fold_*_best_hp.json")
    )
    if not files:
        # Fallback to any json
        files = sorted(results_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    for p in files:
        try:
            meta = parse_filename(p)
        except Exception:
            # Skip unexpected files
            continue
        with p.open("r") as f:
            metrics: Dict[str, float] = json.load(f)

        row: Dict[str, float] = {
            "model_type": meta.model_type,
            "pretrained_seed": meta.pretrained_seed,
            "downstream_seed": meta.downstream_seed,
            "fold": meta.fold,
        }
        # Flatten metric keys: replace '/' with '_' to make CSV friendly
        for k, v in metrics.items():
            row[k.replace("/", "_")] = v
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No parsable result files found in {results_dir}")
    return pd.DataFrame(rows)


def create_combined_results_file(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "combined_best_hp_results.csv"
    # Order a few common columns first if present
    base_cols = [
        "model_type",
        "pretrained_seed",
        "downstream_seed",
        "fold",
        "training_epochs",
        "best_lr",
        "best_weight_decay",
    ]
    ordered_cols = [c for c in base_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining
    df[final_cols].to_csv(combined_path, index=False)
    return combined_path


def summarize_across_folds(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    # Aggregate across folds per model_type, pretrained_seed, downstream_seed
    group_cols = ["model_type", "pretrained_seed", "downstream_seed"]
    value_cols = [m.replace("/", "_") for m in metrics if m.replace("/", "_") in df.columns]
    if not value_cols:
        value_cols = [c for c in df.columns if c.startswith("val_") or c.startswith("test_")]
    if not value_cols:
        return
    mean_df = (
        df.groupby(group_cols)[value_cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )
    std_df = (
        df.groupby(group_cols)[value_cols]
        .std(ddof=0)
        .reset_index()
        .sort_values(group_cols)
    )
    mean_df.to_csv(out_dir / "best_hp_results_mean_across_folds.csv", index=False)
    std_df.to_csv(out_dir / "best_hp_results_std_across_folds.csv", index=False)


def summarize_by_model(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    """Aggregate by model_type only and write CSV summary (no plots)."""
    value_cols = [m.replace("/", "_") for m in metrics if m.replace("/", "_") in df.columns]
    if not value_cols:
        value_cols = [c for c in df.columns if c.startswith("val_") or c.startswith("test_")]
    if not value_cols:
        return
    
    # Aggregate by model_type
    model_summary = (
        df.groupby("model_type")[value_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    
    # Flatten column names properly
    new_columns = []
    for col in model_summary.columns:
        if isinstance(col, tuple):
            if col[1]:  # Has aggregation name
                new_columns.append(f"{col[0]}_{col[1]}")
            else:  # Just the base column name
                new_columns.append(col[0])
        else:
            new_columns.append(col)
    model_summary.columns = new_columns
    model_summary.to_csv(out_dir / "best_hp_results_by_model.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze best-HP DILI test results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILI_Gold_best_hp_runs/best_hp_test_results"),
        help="Directory containing DILI_*_pretrained_seed_*_downstream_seed_*_fold_*_best_hp.json files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILI_Gold_best_hp_runs/analysis"),
        help="Directory to write CSV summaries",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "val/auprc_mean",
            "val/auroc_mean",
            "val/f1_mean",
            "val/balanced_accuracy_mean",
            "val/mcc_mean",
            "val/cohen_kappa_mean",
            "val/enrichment_factor_mean",
            "val/spearman_rho_mean",
            "val/kendall_tau_mean",
            "val/spearman_rho_top10_mean",
            "val/kendall_tau_top10_mean",
            "val/spearman_rho_top20_mean",
            "val/kendall_tau_top20_mean",
            "val/spearman_rho_top50_mean",
            "val/kendall_tau_top50_mean",
            "val/ECE_mean",
            "val/ppv_mean",
            "val/pos_lr_mean",
            "val/optimal_threshold_mean",
            "val/loss",
            # Decision-making metrics for drug discovery
            "val/recall_at_precision_75_mean",
            "val/recall_at_precision_80_mean", 
            "val/recall_at_precision_85_mean",
            "val/recall_at_precision_90_mean",
            "val/recall_at_precision_95_mean",
            "val/tnr_at_recall_75_mean",
            "val/tnr_at_recall_80_mean",
            "val/tnr_at_recall_85_mean",
            "val/tnr_at_recall_90_mean",
            "val/tnr_at_recall_95_mean",
            "test/auprc_mean",
            "test/auroc_mean",
            "test/f1_mean",
            "test/balanced_accuracy_mean",
            "test/mcc_mean",
            "test/cohen_kappa_mean",
            "test/enrichment_factor_mean",
            "test/spearman_rho_mean",
            "test/kendall_tau_mean",
            "test/spearman_rho_top10_mean",
            "test/kendall_tau_top10_mean",
            "test/spearman_rho_top20_mean",
            "test/kendall_tau_top20_mean",
            "test/spearman_rho_top50_mean",
            "test/kendall_tau_top50_mean",
            "test/ECE_mean",
            "test/ppv_mean",
            "test/pos_lr_mean",
            "test/optimal_threshold_mean",
            "test/loss",
            # Decision-making metrics for drug discovery
            "test/recall_at_precision_75_mean",
            "test/recall_at_precision_80_mean",
            "test/recall_at_precision_85_mean", 
            "test/recall_at_precision_90_mean",
            "test/recall_at_precision_95_mean",
            "test/tnr_at_recall_75_mean",
            "test/tnr_at_recall_80_mean",
            "test/tnr_at_recall_85_mean",
            "test/tnr_at_recall_90_mean",
            "test/tnr_at_recall_95_mean",
        ],
        help="Metrics to include in summaries (keys from JSON)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} results")
    print(f"Model types: {sorted(df['model_type'].unique())}")
    print(f"Pretrained seeds: {sorted(df['pretrained_seed'].unique())}")
    print(f"Downstream seeds: {sorted(df['downstream_seed'].unique())}")
    print(f"Folds: {sorted(df['fold'].unique())}")

    combined_path = create_combined_results_file(df, out_dir)
    print(f"Wrote combined CSV: {combined_path}")

    summarize_across_folds(df, out_dir, args.metrics)
    print(f"Wrote fold-mean and fold-std summaries to {out_dir}")
    
    summarize_by_model(df, out_dir, args.metrics)
    print(f"Wrote model-level summary to {out_dir}")

    # ------------------------------------------------------------------
    # Research-quality violin plots per metric across model types
    # ------------------------------------------------------------------
    def plot_violin_by_model(metric_key: str) -> None:
        col = metric_key.replace("/", "_")
        if col not in df.columns:
            return
        plot_df = df[["model_type", col]].dropna().copy()
        if plot_df.empty:
            return
        # Model order
        model_order = [
            "Vanilla_Clip_without_VAE",
            "Vanilla_Clip_with_VAE",
            "Vanilla_Clip_with_Frozen_VAE",
            "Soft_Clip_with_Frozen_Teacher",
            "Soft_Clip_with_Teacher",
            "Soft_Clip_with_Teacher_with_centering",
        ]
        plot_df = plot_df[plot_df["model_type"].isin(model_order)]
        if plot_df.empty:
            return

        # Aesthetics
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        # Violin with quartiles
        sns.violinplot(
            data=plot_df,
            x="model_type",
            y=col,
            order=model_order,
            inner=None,
            cut=0,
            linewidth=1.0,
            ax=ax,
            color="#d0d7e5",
        )
        # Overlay boxplot for quartiles
        sns.boxplot(
            data=plot_df,
            x="model_type",
            y=col,
            order=model_order,
            showcaps=True,
            boxprops={"facecolor": "white", "alpha": 0.9},
            showfliers=False,
            whiskerprops={"linewidth": 1.2},
            width=0.2,
            ax=ax,
        )
        # Overlay jittered points
        sns.stripplot(
            data=plot_df,
            x="model_type",
            y=col,
            order=model_order,
            color="#4e79a7",
            alpha=0.5,
            dodge=False,
            size=3,
            ax=ax,
        )

        ax.set_xlabel("Model type")
        # Human-friendly metric title
        ylabel = metric_key.replace("_", " ").replace("val/", "val ").replace("test/", "test ")
        # Special formatting for decision-making metrics
        if "recall_at_precision" in metric_key:
            precision_val = metric_key.split("_")[-2]  # Extract precision value (e.g., "75")
            ylabel = f"Recall@{precision_val}% Precision"
        elif "tnr_at_recall" in metric_key:
            recall_val = metric_key.split("_")[-2]  # Extract recall value (e.g., "80")
            ylabel = f"TNR@{recall_val}% Recall"
        
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by model type (best HP, test runs)")
        ax.set_xticklabels(model_order, rotation=30, ha="right")
        ax.margins(x=0.02)
        sns.despine(ax=ax)
        fig.tight_layout()
        out_path = out_dir / f"violin_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)

    # Generate one plot per requested metric (that exists in data)
    for m in args.metrics:
        plot_violin_by_model(m)
    print(f"Wrote violin plots per metric to {out_dir}")


if __name__ == "__main__":
    main()


