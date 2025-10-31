#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


RESULTS_FILENAME_RE = re.compile(
    r"DILI_(?P<model_type>[^_]+(?:_[^_]+)*)_pretrained_seed_(?P<pretrained_seed>\d+)_downstream_seed_(?P<downstream_seed>\d+)_fold_(?P<fold>\d+)_lr_(?P<lr>[0-9eE+\.-]+)_wd_(?P<wd>[0-9eE+\.-]+)\.json$"
)


@dataclass(frozen=True)
class ParsedMeta:
    model_type: str
    pretrained_seed: int
    downstream_seed: int
    fold: int
    lr: float
    wd: float


def parse_filename(path: Path) -> ParsedMeta:
    m = RESULTS_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path}")
    model_type = m.group("model_type")
    pretrained_seed = int(m.group("pretrained_seed"))
    downstream_seed = int(m.group("downstream_seed"))
    fold = int(m.group("fold"))
    # Support scientific notation like 1e-3
    lr = float(m.group("lr"))
    wd = float(m.group("wd"))
    return ParsedMeta(
        model_type=model_type,
        pretrained_seed=pretrained_seed,
        downstream_seed=downstream_seed,
        fold=fold,
        lr=lr,
        wd=wd
    )


def load_results(results_dir: Path) -> pd.DataFrame:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    rows: List[Dict] = []
    # Updated pattern to match new filename format
    files: List[Path] = sorted(results_dir.rglob("DILI_*_pretrained_seed_*_downstream_seed_*_fold_*_lr_*_wd_*.json"))
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
            "lr": meta.lr,
            "wd": meta.wd,
        }
        # Flatten metric keys by replacing '/' with '_'
        for k, v in metrics.items():
            row[k.replace("/", "_")] = v
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return pd.DataFrame(rows)


def summarize_by_hparams(df: pd.DataFrame, metrics: List[str], group_by_model: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_by_model:
        group_cols = ["model_type", "pretrained_seed", "lr", "wd"]
    else:
        group_cols = ["lr", "wd"]
    
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


def plot_heatmaps(mean_df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep lr sorted ascending, wd sorted ascending
    lrs = sorted(mean_df["lr"].unique())
    wds = sorted(mean_df["wd"].unique())
    higher_is_better_map = {
        "val/auprc_mean": True,
        "val/auroc_mean": True,
        "val/f1_mean": True,
        "val/balanced_accuracy_mean": True,
        "val/mcc_mean": True,
        "val/ECE_mean": False,
        "val/ppv_mean": True,
        "val/pos_lr_mean": True,
        "val/optimal_threshold_mean": False,  # Not really a performance metric
        "val/loss": False,
    }
    for metric in metrics:
        col = metric.replace("/", "_")
        if col not in mean_df.columns:
            continue
        pivot = mean_df.pivot(index="lr", columns="wd", values=col).reindex(index=lrs, columns=wds)
        plt.figure(figsize=(1.2 + 0.8 * len(wds), 1.2 + 0.8 * len(lrs)))
        ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"{metric} (mean across seeds/folds)")
        plt.xlabel("weight_decay")
        plt.ylabel("learning_rate")
        # Highlight best cell
        hib = higher_is_better_map.get(metric, True)
        best_idx = pivot.values.argmax() if hib else pivot.values.argmin()
        r, c = divmod(int(best_idx), pivot.shape[1])
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor='red', linewidth=2))
        plt.tight_layout()
        fig_path = out_dir / f"heatmap_{col}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()


def plot_multi_heatmap(mean_df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lrs = sorted(mean_df["lr"].unique())
    wds = sorted(mean_df["wd"].unique())
    higher_is_better_map = {
        "val/auprc_mean": True,
        "val/auroc_mean": True,
        "val/f1_mean": True,
        "val/balanced_accuracy_mean": True,
        "val/mcc_mean": True,
        "val/ECE_mean": False,
        "val/ppv_mean": True,
        "val/pos_lr_mean": True,
        "val/optimal_threshold_mean": False,  # Not really a performance metric
        "val/loss": False,
    }
    cols = 3
    valid_metrics = [m for m in metrics if m.replace("/", "_") in mean_df.columns]
    if not valid_metrics:
        return
    rows = (len(valid_metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    # Plot each metric
    for idx, metric in enumerate(valid_metrics):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        col = metric.replace("/", "_")
        pivot = mean_df.pivot(index="lr", columns="wd", values=col).reindex(index=lrs, columns=wds)
        hm = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("weight_decay")
        ax.set_ylabel("learning_rate")
        hib = higher_is_better_map.get(metric, True)
        best_idx = pivot.values.argmax() if hib else pivot.values.argmin()
        r, c = divmod(int(best_idx), pivot.shape[1])
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor='red', linewidth=2))
    # Hide any unused subplots
    for idx in range(len(valid_metrics), rows*cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    fig_path = out_dir / "heatmaps_all_metrics.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

def add_rankings(mean_df: pd.DataFrame, metric: str, higher_is_better: bool = True, top_k: int = 10) -> pd.DataFrame:
    col = metric.replace("/", "_")
    if col not in mean_df.columns:
        raise ValueError(f"Metric not in summary: {metric}")
    return (
        mean_df.sort_values(col, ascending=not higher_is_better)
        .head(top_k)
        .reset_index(drop=True)
    )


def create_combined_results_file(df: pd.DataFrame, out_dir: Path) -> None:
    """Create combined file with all individual results"""
    # Rename columns for clarity
    combined_df = df.copy()
    combined_df = combined_df.rename(columns={
        'downstream_seed': 'downstream_seed',
        'lr': 'LR',
        'wd': 'weight_decay'
    })
    
    # Reorder columns to match requested format
    base_columns = ['model_type', 'pretrained_seed', 'fold', 'downstream_seed', 'LR', 'weight_decay']
    metric_columns = [col for col in combined_df.columns if col not in base_columns]
    final_columns = base_columns + sorted(metric_columns)
    
    combined_df = combined_df[final_columns]
    combined_df.to_csv(out_dir / "combined_results.csv", index=False)
    print(f"Created combined_results.csv with {len(combined_df)} rows and {len(final_columns)} columns")


def find_best_hyperparameters(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Find best hyperparameters for each model_type, pretrained_seed, fold, downstream_seed combination"""
    # Rename columns for clarity
    df_renamed = df.copy()
    df_renamed = df_renamed.rename(columns={
        'downstream_seed': 'downstream_seed',
        'lr': 'LR',
        'wd': 'weight_decay'
    })
    
    # Get the metric column
    metric_col = metric.replace("/", "_")
    if metric_col not in df_renamed.columns:
        print(f"Warning: Metric {metric} not found in data. Available metrics: {[col for col in df_renamed.columns if 'val_' in col]}")
        return
    
    # Determine if higher is better for this metric
    higher_is_better_map = {
        "val_auprc_mean": True,
        "val_auroc_mean": True,
        "val_f1_mean": True,
        "val_balanced_accuracy_mean": True,
        "val_mcc_mean": True,
        "val_ECE_mean": False,
        "val_ppv_mean": True,
        "val_pos_lr_mean": True,
        "val_optimal_threshold_mean": False,  # Not really a performance metric
        "val_loss": False,
    }
    higher_is_better = higher_is_better_map.get(metric_col, True)
    
    # Group by model_type, pretrained_seed, fold, downstream_seed and find best hyperparameters
    group_cols = ['model_type', 'pretrained_seed', 'fold', 'downstream_seed']
    
    if higher_is_better:
        best_hp = df_renamed.loc[df_renamed.groupby(group_cols)[metric_col].idxmax()]
    else:
        best_hp = df_renamed.loc[df_renamed.groupby(group_cols)[metric_col].idxmin()]
    
    # Select all relevant columns for output (hyperparameters + all metrics)
    # Get all metric columns (exclude the grouping columns and hyperparameter columns)
    metric_columns = [col for col in df_renamed.columns if col not in group_cols + ['LR', 'weight_decay']]
    output_columns = group_cols + ['LR', 'weight_decay'] + metric_columns
    best_hp_output = best_hp[output_columns].copy()
    
    # Sort by group columns for better readability
    best_hp_output = best_hp_output.sort_values(group_cols)
    
    best_hp_output.to_csv(out_dir / f"best_hyperparameters_by_{metric.replace('/', '_')}.csv", index=False)
    print(f"Created best_hyperparameters_by_{metric.replace('/', '_')}.csv with {len(best_hp_output)} rows")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot DILI array validation results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILI_Gold_array_runs/test_results"),
        help="Directory containing DILI_*_pretrained_seed_*_downstream_seed_*_fold_*_lr_*_wd_*.json files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILI_Gold_array_runs/analysis"),
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
        ],
        help="Metrics to aggregate and plot (as keys from JSON)",
    )
    parser.add_argument(
        "--group_by_model",
        action="store_true",
        help="Group results by model type and pretrained seed in addition to lr/wd",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val/auprc_mean",
        help="Metric to use for finding best hyperparameters",
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
    print(f"Pretrained seeds: {sorted(df['pretrained_seed'].unique())}")
    print(f"Downstream seeds: {sorted(df['downstream_seed'].unique())}")
    print(f"Folds: {sorted(df['fold'].unique())}")
    print(f"Learning rates: {sorted(df['lr'].unique())}")
    print(f"Weight decays: {sorted(df['wd'].unique())}")

    # Step 1: Create combined results file
    print("\n=== Step 1: Creating combined results file ===")
    create_combined_results_file(df, out_dir)
    
    # Step 2: Find best hyperparameters
    print(f"\n=== Step 2: Finding best hyperparameters based on {args.best_metric} ===")
    find_best_hyperparameters(df, args.best_metric, out_dir)

    # Optional: Generate additional analysis files
    if not args.no_figures:
        print("\n=== Generating additional analysis files ===")
        mean_df, std_df = summarize_by_hparams(df, args.metrics, group_by_model=args.group_by_model)
        
        if args.group_by_model:
            mean_df.to_csv(out_dir / "results_by_model_pretrained_seed_lr_wd_mean.csv", index=False)
            std_df.to_csv(out_dir / "results_by_model_pretrained_seed_lr_wd_std.csv", index=False)
            # Single aggregated CSV with mean and std columns
            agg_df = mean_df.merge(std_df, on=["model_type", "pretrained_seed", "lr", "wd"], suffixes=("_mean", "_std"))
            agg_df.to_csv(out_dir / "results_by_model_pretrained_seed_lr_wd_agg.csv", index=False)
        else:
            mean_df.to_csv(out_dir / "results_by_lr_wd_mean.csv", index=False)
            std_df.to_csv(out_dir / "results_by_lr_wd_std.csv", index=False)
            # Single aggregated CSV with mean and std columns
            agg_df = mean_df.merge(std_df, on=["lr", "wd"], suffixes=("_mean", "_std"))
            agg_df.to_csv(out_dir / "results_by_lr_wd_agg.csv", index=False)

        # Produce heatmaps for each metric (only if not grouping by model)
        if not args.group_by_model:
            plot_heatmaps(mean_df, out_dir, args.metrics)
            # Produce a single multi-panel figure for all metrics
            plot_multi_heatmap(mean_df, out_dir, args.metrics)
        else:
            print("Skipping heatmap generation when grouping by model (too many dimensions)")

    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()


