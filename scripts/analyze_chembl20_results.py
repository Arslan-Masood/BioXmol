#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# Example filename: Chembl20_Soft_Clip_with_Frozen_Teacher_frac_100_seed_0_split_0_lr_5.0e-05.json
RESULTS_FILENAME_RE = re.compile(
    r"Chembl20_(?P<model_type>[^_]+(?:_[^_]+)*)_frac_(?P<frac>\d+)_seed_(?P<seed>\d+)_split_(?P<split>\d+)_lr_(?P<lr>[0-9eE+\.-]+)\.json$"
)


@dataclass(frozen=True)
class ParsedMeta:
    model_type: str
    frac: int
    seed: int
    split: int
    lr: float


def parse_filename(path: Path) -> ParsedMeta:
    m = RESULTS_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path}")
    model_type = m.group("model_type")
    frac = int(m.group("frac"))
    seed = int(m.group("seed"))
    split = int(m.group("split"))
    # Support scientific notation like 5.0e-05
    lr = float(m.group("lr"))
    return ParsedMeta(
        model_type=model_type,
        frac=frac,
        seed=seed,
        split=split,
        lr=lr
    )


def load_results(results_dir: Path) -> pd.DataFrame:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    rows: List[Dict] = []
    # Search for Chembl20 result files
    files: List[Path] = sorted(results_dir.rglob("Chembl20_*_frac_*_seed_*_split_*_lr_*.json"))
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
            "frac": meta.frac,
            "seed": meta.seed,
            "split": meta.split,
            "lr": meta.lr,
        }
        # Flatten metric keys by replacing '/' with '_'
        for k, v in metrics.items():
            row[k.replace("/", "_")] = v
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return pd.DataFrame(rows)


def summarize_by_model_frac(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize results by model type and fraction (across seeds/splits)."""
    group_cols = ["model_type", "frac"]
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


def summarize_by_model(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize results by model type (across all fracs, seeds, splits)."""
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


def summarize_by_model_seed_split(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Get individual results by model type, frac, seed, and split (no aggregation needed)."""
    value_cols = [m.replace("/", "_") for m in metrics if m.replace("/", "_") in df.columns]
    if not value_cols:
        raise ValueError("None of the requested metrics found in results.")
    # Just return the dataframe with model_type, frac, seed, split, lr and metrics
    result_cols = ["model_type", "frac", "seed", "split", "lr"] + value_cols
    return df[result_cols].sort_values(["model_type", "frac", "seed", "split"])


def plot_line_by_frac(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Create line plot with fraction on x-axis, metric on y-axis, and lines colored by model type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    col = metric.replace("/", "_")
    if col not in mean_df.columns or col not in std_df.columns:
        raise ValueError(f"Metric {metric} (column {col}) not found in results.")
    
    model_order = [
        "Vanilla_Clip_without_VAE",
        "Vanilla_Clip_with_VAE",
        "Soft_Clip_with_Frozen_Teacher",
        "Soft_Clip_with_Teacher",
        "Soft_Clip_with_Teacher_with_centering",
    ]
    # Filter to only models that exist in data
    available_models = [m for m in model_order if m in mean_df["model_type"].unique()]
    frac_order = sorted(mean_df["frac"].unique())
    
    # Define colors for each model type
    colors = plt.cm.tab10(range(len(available_models)))
    color_map = {model: colors[i] for i, model in enumerate(available_models)}
    
    plt.figure(figsize=(10, 6))
    
    # Plot a line for each model type
    for model in available_models:
        model_data = mean_df[mean_df["model_type"] == model].sort_values("frac")
        model_std = std_df[std_df["model_type"] == model].sort_values("frac")
        
        fracs = model_data["frac"].values
        means = model_data[col].values
        stds = model_std[col].values
        
        plt.plot(fracs, means, marker='o', label=model, color=color_map[model], linewidth=2, markersize=6)
        plt.fill_between(fracs, means - stds, means + stds, alpha=0.2, color=color_map[model])
    
    plt.xlabel("Fraction (%)", fontsize=14)
    plt.ylabel(col.replace("_", " ").title(), fontsize=14)
    plt.title(f"{col.replace('_', ' ').title()} by Fraction (mean ± std across seeds/splits)", fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(frac_order)
    plt.tight_layout()
    
    fig_path = out_dir / f"lineplot_{col}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()


def create_combined_results_file(df: pd.DataFrame, out_dir: Path) -> None:
    """Create combined file with all individual results"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rename columns for clarity
    combined_df = df.copy()
    combined_df = combined_df.rename(columns={
        'frac': 'frac',
        'seed': 'seed',
        'split': 'split',
        'lr': 'LR',
    })
    
    # Reorder columns to match requested format
    base_columns = ['model_type', 'frac', 'seed', 'split', 'LR']
    metric_columns = [col for col in combined_df.columns if col not in base_columns]
    final_columns = base_columns + sorted(metric_columns)
    
    combined_df = combined_df[final_columns]
    combined_df.to_csv(out_dir / "combined_results.csv", index=False)
    print(f"Created combined_results.csv with {len(combined_df)} rows and {len(final_columns)} columns")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot ChEMBL20 fine-tuning results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/Chembl_20_FineTuning"),
        help="Directory containing Chembl20_*_frac_*_seed_*_split_*_lr_*.json files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/Chembl_20_FineTuning/analysis"),
        help="Directory to write CSV summaries and plots",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val/auprc_mean",
        help="Metric to plot (as key from JSON, e.g., val/auprc_mean)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    
    # Print summary statistics
    print(f"Loaded {len(df)} results")
    print(f"Model types: {sorted(df['model_type'].unique())}")
    print(f"Fractions: {sorted(df['frac'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Splits: {sorted(df['split'].unique())}")
    print(f"Learning rates: {sorted(df['lr'].unique())}")

    # Step 1: Create combined results file
    print("\n=== Step 1: Creating combined results file ===")
    create_combined_results_file(df, out_dir)
    
    # Step 2: Summarize by model type and fraction (across seeds/splits)
    print("\n=== Step 2: Summarizing by model type and fraction (across seeds/splits) ===")
    metrics_list = [args.metric]  # Convert single metric to list for summarize functions
    mean_df_frac, std_df_frac = summarize_by_model_frac(df, metrics_list)
    mean_df_frac.to_csv(out_dir / "results_by_model_frac_mean.csv", index=False)
    std_df_frac.to_csv(out_dir / "results_by_model_frac_std.csv", index=False)
    # Single aggregated CSV with mean and std columns
    agg_df_frac = mean_df_frac.merge(std_df_frac, on=["model_type", "frac"], suffixes=("_mean", "_std"))
    agg_df_frac.to_csv(out_dir / "results_by_model_frac_agg.csv", index=False)
    print(f"Created summaries by model type and fraction")
    
    # Step 3: Summarize by model type (across all fracs, seeds, splits)
    print("\n=== Step 3: Summarizing by model type (across all fracs/seeds/splits) ===")
    mean_df, std_df = summarize_by_model(df, metrics_list)
    mean_df.to_csv(out_dir / "results_by_model_mean.csv", index=False)
    std_df.to_csv(out_dir / "results_by_model_std.csv", index=False)
    # Single aggregated CSV with mean and std columns
    agg_df = mean_df.merge(std_df, on=["model_type"], suffixes=("_mean", "_std"))
    agg_df.to_csv(out_dir / "results_by_model_agg.csv", index=False)
    print(f"Created summaries by model type")
    
    # Step 4: Save individual results by model type, frac, seed, and split
    print("\n=== Step 4: Saving individual results by model type, frac, seed, and split ===")
    results_by_seed_split = summarize_by_model_seed_split(df, metrics_list)
    results_by_seed_split.to_csv(out_dir / "results_by_model_frac_seed_split.csv", index=False)
    print(f"Created individual results by model type, frac, seed, and split")

    # Generate line plot
    print(f"\n=== Generating line plot for {args.metric} ===")
    plot_line_by_frac(mean_df_frac, std_df_frac, out_dir, args.metric)
    print(f"Created line plot in {out_dir}")

    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()

