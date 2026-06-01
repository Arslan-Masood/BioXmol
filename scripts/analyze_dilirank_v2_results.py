#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


# Example filename: DILIRank_v2_Soft_Clip_with_Frozen_Teacher_seed_0_lr_5.0e-05.json
RESULTS_FILENAME_RE = re.compile(
    r"DILIRank_v2_(?P<model_type>[^_]+(?:_[^_]+)*)_seed_(?P<seed>\d+)_lr_(?P<lr>[0-9eE+\.-]+)\.json$"
)


@dataclass(frozen=True)
class ParsedMeta:
    model_type: str
    seed: int
    lr: float


def parse_filename(path: Path) -> ParsedMeta:
    m = RESULTS_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path}")
    model_type = m.group("model_type")
    seed = int(m.group("seed"))
    # Support scientific notation like 5.0e-05
    lr = float(m.group("lr"))
    return ParsedMeta(
        model_type=model_type,
        seed=seed,
        lr=lr
    )


def load_results(results_dir: Path) -> pd.DataFrame:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    rows: List[Dict] = []
    # Search for DILIRank_v2 result files
    files: List[Path] = sorted(results_dir.rglob("DILIRank_v2_*_seed_*_lr_*.json"))
    if not files:
        # Fallback to any json in test_results directories
        files = sorted(results_dir.rglob("test_results/*.json"))
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
            "seed": meta.seed,
            "lr": meta.lr,
        }
        # Flatten metric keys by replacing '/' with '_'
        for k, v in metrics.items():
            row[k.replace("/", "_")] = v
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return pd.DataFrame(rows)


def summarize_by_model(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize results by model type (across seeds)."""
    group_cols = ["model_type"]
    value_cols = [m.replace("/", "_") for m in metrics if m.replace("/", "_") in df.columns]
    if not value_cols:
        raise ValueError("None of the requested metrics found in results.")
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
    return mean_df, std_df


def summarize_by_model_seed(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Get individual results by model type and seed (no aggregation needed)."""
    value_cols = [m.replace("/", "_") for m in metrics if m.replace("/", "_") in df.columns]
    if not value_cols:
        raise ValueError("None of the requested metrics found in results.")
    # Just return the dataframe with model_type, seed, lr and metrics
    result_cols = ["model_type", "seed", "lr"] + value_cols
    return df[result_cols].sort_values(["model_type", "seed"])


def plot_barplots(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    """Create bar plots comparing models (mean across seeds with std error bars)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_order = [
        "Vanilla_Clip_without_VAE",
        "Vanilla_Clip_with_VAE",
        "Soft_Clip_with_Frozen_Teacher",
        "Soft_Clip_with_Teacher",
        "Soft_Clip_with_Teacher_with_centering",
    ]
    # Filter to only models that exist in data
    model_order = [m for m in model_order if m in mean_df["model_type"].unique()]
    
    for metric in metrics:
        col = metric.replace("/", "_")
        if col not in mean_df.columns or col not in std_df.columns:
            continue
        plt.figure(figsize=(10, 6))
        means = [mean_df[mean_df["model_type"] == m][col].values[0] for m in model_order]
        stds = [std_df[std_df["model_type"] == m][col].values[0] for m in model_order]
        bars = plt.bar(range(len(model_order)), means, color="#4e79a7", alpha=0.7, 
                       yerr=stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        plt.xticks(range(len(model_order)), model_order, rotation=45, ha="right")
        plt.ylabel(col.replace("_", " ").title())
        plt.title(f"{col.replace('_', ' ').title()} by Model Type (mean ± std across seeds)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig_path = out_dir / f"barplot_{col}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()


def plot_violin_by_model(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    """Create violin plots for metrics by model type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_order = [
        "Vanilla_Clip_without_VAE",
        "Vanilla_Clip_with_VAE",
        "Soft_Clip_with_Frozen_Teacher",
        "Soft_Clip_with_Teacher",
        "Soft_Clip_with_Teacher_with_centering",
    ]
    # Filter to only models that exist in data
    model_order = [m for m in model_order if m in df["model_type"].unique()]
    
    higher_is_better_map = {
        "val_auprc_mean": True,
        "val_auroc_mean": True,
        "val_f1_mean": True,
        "val_balanced_accuracy_mean": True,
        "val_mcc_mean": True,
        "val_ECE_mean": False,
        "val_ppv_mean": True,
        "val_pos_lr_mean": True,
        "val_optimal_threshold_mean": False,
        "val_loss": False,
        "val_supervised_loss": False,
    }
    
    for metric in metrics:
        col = metric.replace("/", "_")
        if col not in df.columns:
            continue
        plot_df = df[["model_type", col]].dropna().copy()
        if plot_df.empty:
            continue
        plot_df = plot_df[plot_df["model_type"].isin(model_order)]
        if plot_df.empty:
            continue

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
        ylabel = col.replace("_", " ").replace("val ", "val ").title()
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by model type (DILIRank v2.0)")
        ax.set_xticklabels(model_order, rotation=30, ha="right")
        ax.margins(x=0.02)
        sns.despine(ax=ax)
        fig.tight_layout()
        out_path = out_dir / f"violin_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)


def create_combined_results_file(df: pd.DataFrame, out_dir: Path) -> None:
    """Create combined file with all individual results"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rename columns for clarity
    combined_df = df.copy()
    combined_df = combined_df.rename(columns={
        'seed': 'seed',
        'lr': 'LR',
    })
    
    # Reorder columns to match requested format
    base_columns = ['model_type', 'seed', 'LR']
    metric_columns = [col for col in combined_df.columns if col not in base_columns]
    final_columns = base_columns + sorted(metric_columns)
    
    combined_df = combined_df[final_columns]
    combined_df.to_csv(out_dir / "combined_results.csv", index=False)
    print(f"Created combined_results.csv with {len(combined_df)} rows and {len(final_columns)} columns")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot DILIRank v2.0 fine-tuning results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILIRank_v2_FineTuning"),
        help="Directory containing DILIRank_v2_*_seed_*_lr_*.json files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILIRank_v2_FineTuning/analysis"),
        help="Directory to write CSV summaries and plots",
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
            "val/ECE_mean",
            "val/ppv_mean",
            "val/pos_lr_mean",
            "val/optimal_threshold_mean",
            "val/loss",
            "val/supervised_loss",
            "val/recall_mean",
            "val/sensitivity_mean",
            "val/specificity_mean",
            "val/precision_mean",
            "val/cohen_kappa_mean",
            "val/enrichment_factor_mean",
            "val/average_precision_mean",
        ],
        help="Metrics to aggregate and plot (as keys from JSON)",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip generating figures",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    
    # Print summary statistics
    print(f"Loaded {len(df)} results")
    print(f"Model types: {sorted(df['model_type'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Learning rates: {sorted(df['lr'].unique())}")

    # Step 1: Create combined results file
    print("\n=== Step 1: Creating combined results file ===")
    create_combined_results_file(df, out_dir)
    
    # Step 2: Summarize by model type (across seeds)
    print("\n=== Step 2: Summarizing by model type (across seeds) ===")
    mean_df, std_df = summarize_by_model(df, args.metrics)
    mean_df.to_csv(out_dir / "results_by_model_mean.csv", index=False)
    std_df.to_csv(out_dir / "results_by_model_std.csv", index=False)
    # Single aggregated CSV with mean and std columns
    agg_df = mean_df.merge(std_df, on=["model_type"], suffixes=("_mean", "_std"))
    agg_df.to_csv(out_dir / "results_by_model_agg.csv", index=False)
    print(f"Created summaries by model type")
    
    # Step 3: Save individual results by model type and seed
    print("\n=== Step 3: Saving individual results by model type and seed ===")
    results_by_seed = summarize_by_model_seed(df, args.metrics)
    results_by_seed.to_csv(out_dir / "results_by_model_seed.csv", index=False)
    print(f"Created individual results by model type and seed")

    # Optional: Generate figures
    if not args.no_figures:
        print("\n=== Generating figures ===")
        plot_barplots(mean_df, std_df, out_dir, args.metrics)
        plot_violin_by_model(df, out_dir, args.metrics)
        print(f"Created figures in {out_dir}")

    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()

