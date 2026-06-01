#!/usr/bin/env python3
"""
Activity Cliff Results Visualization (Seaborn Edition)
=======================================================

Clean, publication-ready plots using seaborn's high-level API.

Usage:
------
    python plot_activity_cliff_seaborn.py \
        --results_root /path/to/results \
        --output_dir /path/to/figures \
        --auto_discover
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def setup_style():
    """Configure seaborn style for publication-ready plots."""
    sns.set_theme(
        style="whitegrid",
        context="paper",  # or "talk" for presentations
        font_scale=1.2,
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_models(results_root: str) -> List[str]:
    """Auto-discover model directories in results root."""
    results_path = Path(results_root)
    models = []
    
    for model_dir in sorted(results_path.iterdir()):
        if model_dir.is_dir():
            for seed_dir in model_dir.iterdir():
                if seed_dir.is_dir() and (seed_dir / "activity_cliff_summary.csv").exists():
                    models.append(model_dir.name)
                    break
    return models


def load_all_data(
    results_root: str,
    models: List[str],
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary and pairs data for all models/seeds."""
    
    summary_records, pairs_records = [], []
    
    for model in models:
        for seed in seeds:
            base_path = Path(results_root) / model / f"seed_{seed}"
            
            summary_path = base_path / "activity_cliff_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df["model"] = model
                df["seed"] = seed
                summary_records.append(df)
            
            pairs_path = base_path / "activity_cliff_pairs.csv"
            if pairs_path.exists():
                df = pd.read_csv(pairs_path)
                df["model"] = model
                df["seed"] = seed
                pairs_records.append(df)
    
    summary_df = pd.concat(summary_records, ignore_index=True) if summary_records else pd.DataFrame()
    pairs_df = pd.concat(pairs_records, ignore_index=True) if pairs_records else pd.DataFrame()
    
    return summary_df, pairs_df


def shorten_model_name(name: str, max_len: int = 20) -> str:
    """Shorten model names for display."""
    replacements = {
        "_with_": "+", "_without_": "-", "Soft_Clip": "SC",
        "Vanilla_Clip": "VC", "Frozen_Teacher": "FT", 
        "centering": "ctr", "hydrochloride": "HCl",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name[:max_len-3] + "..." if len(name) > max_len else name


# Mapping from internal model identifiers to human-friendly / LaTeX display names
NAME_MAPPING = {
    "GNN_Soft_Clip_with_Teacher_with_centering_seed0_second_fc": r"Bio$\mathcal{X}$Mol-soft contrastive",
    "GNN_Vanilla_Clip_without_VAE_seed0_second_fc": r"Bio$\mathcal{X}$Mol-hard contrastive",
    "ECFP_1024_2": "ECFP",
    "MoLFormer-XL-both-10pct": "MolFormer",
    "ChemBERTa-MTR": "ChemBERTa-MTR",
    "ChemBERTa-MLM": "ChemBERTa-MLM",
}


def get_display_model_name(name: str, max_len: int = 40) -> str:
    """
    Return the display name for a model.
    Prefer the explicit NAME_MAPPING; fall back to a shortened raw name.
    """
    if name in NAME_MAPPING:
        return NAME_MAPPING[name]
    return shorten_model_name(name, max_len=max_len)


def prepare_data(summary_df: pd.DataFrame, pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add display-friendly columns."""
    summary_df = summary_df.copy()
    pairs_df = pairs_df.copy()
    
    # Use display names (with LaTeX support) for all plots
    summary_df["Model"] = summary_df["model"].apply(get_display_model_name)
    pairs_df["Model"] = pairs_df["model"].apply(get_display_model_name)
    pairs_df["Pair"] = pairs_df["safe_drug"].str[:12] + " /\n" + pairs_df["toxic_drug"].str[:12]
    
    return summary_df, pairs_df


# =============================================================================
# PLOT 1: Grouped Bar Chart
# =============================================================================

def plot_grouped_bars(summary_df: pd.DataFrame, output_path: str):
    """
    Grouped bar chart for key metrics with three bars per model.
    Order: Per-Drug Accuracy -> Both Correct Rate -> Pairwise Ranking Accuracy.
    """
    # 1. Define the metrics in the desired plotting order
    metrics = ["per_drug_accuracy", "both_correct_rate", "pairwise_accuracy"]
    
    df_long = summary_df.melt(
        id_vars=["Model", "seed"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    # 2. Map names for display
    metric_map = {
        "per_drug_accuracy": "Per-Drug Accuracy",
        "both_correct_rate": "Both Correct Rate",
        "pairwise_accuracy": "Pairwise Ranking Accuracy",
    }
    df_long["Metric"] = df_long["Metric"].map(metric_map)
    
    # Define hue_order explicitly to match the map keys above
    hue_order = ["Per-Drug Accuracy", "Both Correct Rate", "Pairwise Ranking Accuracy"]
    
    # 3. Order models by mean 'per_drug_accuracy' (Descending)
    model_order = (summary_df.groupby("Model")["per_drug_accuracy"]
                   .mean().sort_values(ascending=False).index.tolist())
    
    fig, ax = plt.subplots(figsize=(12, 6)) # Slightly wider for 3 bars
    
    sns.barplot(
        data=df_long,
        x="Model",
        y="Score",
        hue="Metric",
        hue_order=hue_order,
        order=model_order,
        palette="Set2",
        edgecolor="black",
        linewidth=0.8,
        capsize=0.05,
        errwidth=1.5,
        ax=ax,
    )
    
    # Baseline and layout
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Random")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Activity Cliff Performance by Model (Sorted by Per-Drug Accuracy)")
    ax.legend(title="", loc="upper right")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# PLOT 2: Heatmap
# =============================================================================

def plot_heatmap(pairs_df: pd.DataFrame, output_path: str, value: str = "delta"):
    """
    Heatmap of Model × Pair performance.
    """
    # Pivot table with mean across seeds
    pivot = pairs_df.pivot_table(
        index="Model",
        columns="Pair", 
        values=value,
        aggfunc="mean"
    )
    
    # Order by mean performance
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    
    # Figure size based on data
    figsize = (max(10, len(pivot.columns) * 0.9), max(5, len(pivot.index) * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    
    if value == "delta":
        # Diverging colormap centered at 0
        vmax = np.abs(pivot.values).max()
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"label": "Δ = P(toxic) − P(safe)"},
            ax=ax,
        )
    else:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": value.replace("_", " ").title()},
            ax=ax,
        )
    
    ax.set_xlabel("Drug Pair")
    ax.set_ylabel("Model")
    ax.set_title(f"Activity Cliff Heatmap: {value.replace('_', ' ').title()}")
    plt.xticks(rotation=45, ha="right")
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# PLOT 3: Delta Distribution (Box + Strip)
# =============================================================================

def plot_delta_distribution(pairs_df: pd.DataFrame, output_path: str):
    """
    Box plot with overlaid strip plot showing delta distribution per model.
    """
    # Order by median delta
    model_order = (pairs_df.groupby("Model")["delta"]
                   .median().sort_values(ascending=False).index.tolist())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Box plot
    sns.boxplot(
        data=pairs_df,
        x="Model",
        y="delta",
        order=model_order,
        palette="Set2",
        width=0.5,
        fliersize=0,  # Hide outliers (strip will show them)
        ax=ax,
    )
    
    # Overlay strip plot
    sns.stripplot(
        data=pairs_df,
        x="Model",
        y="delta",
        order=model_order,
        color="black",
        alpha=0.5,
        size=4,
        jitter=0.15,
        ax=ax,
    )
    
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Δ = 0 (threshold)")
    ax.set_xlabel("")
    ax.set_ylabel("Δ = P(toxic) − P(safe)")
    ax.set_title("Delta Distribution Across Activity Cliff Pairs")
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_pair_label(row: pd.Series) -> str:
    """Create a compact label for a drug pair with a line break."""
    # Shorten names to 15 chars and add newline
    safe = row["safe_drug"][:15] + "..." if len(row["safe_drug"]) > 15 else row["safe_drug"]
    toxic = row["toxic_drug"][:15] + "..." if len(row["toxic_drug"]) > 15 else row["toxic_drug"]
    return f"{safe} /\n{toxic}"

def plot_dumbbell(
    pairs_df: pd.DataFrame,
    output_path: str,
    model: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Dumbbell plot sorted by Delta (Top=High, Bottom=Low) with two-line labels.
    """
    # Filter to single model/seed
    if model is None:
        model = pairs_df["model"].iloc[0]
    if seed is None:
        seed = pairs_df[pairs_df["model"] == model]["seed"].iloc[0]
    
    df = pairs_df[(pairs_df["model"] == model) & (pairs_df["seed"] == seed)].copy()
    
    # Generate the two-line labels using the updated helper
    df["pair_label"] = df.apply(create_pair_label, axis=1)
    
    # SORTING: ascending=True means index 0 is lowest Delta. 
    # Because y=0 is the bottom of the plot, the highest Delta will be at the TOP.
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)
    
    # Increase height multiplier to 0.6 to accommodate two-line y-axis labels
    n_pairs = len(df)
    figsize = (10, max(5, n_pairs * 0.6)) 
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_pairs)
    
    for i, row in df.iterrows():
        y = y_positions[i]
        p_safe = row["p_safe"]
        p_toxic = row["p_toxic"]
        correct = row["pairwise_correct"]
        
        # Color based on correctness
        color = "green" if correct else "red"
        
        # Draw connecting line
        ax.plot([p_safe, p_toxic], [y, y], color=color, linewidth=2, alpha=0.7, zorder=1)
        
        # Draw points
        ax.scatter(p_safe, y, color="blue", s=80, zorder=2)
        ax.scatter(p_toxic, y, color="orange", s=80, zorder=2)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["pair_label"].tolist())
    ax.set_xlabel("Predicted Probability of Toxicity")
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="P(safe)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="P(toxic)"),
        plt.Line2D([0], [0], color="green", linewidth=2, label="Correct (↗)"),
        plt.Line2D([0], [0], color="red", linewidth=2, label="Incorrect (↘)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, shadow=True)
    
    ax.set_title(
        f"Probability Separation: {get_display_model_name(model, 40)}\n(Sorted by Delta P)",
        pad=20,
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# PLOT 5: Scatter Plot with Regression
# =============================================================================

def plot_scatter_with_regression(summary_df: pd.DataFrame, output_path: str):
    """
    Scatter plot: General performance (ROC-AUC) vs Activity Cliff performance.
    Uses seaborn's regplot for optional trend line.
    """
    # Aggregate across seeds
    agg_df = (summary_df.groupby("Model")
              .agg({
                  "per_drug_roc_auc": ["mean", "std"],
                  "pairwise_accuracy": ["mean", "std"],
              })
              .reset_index())
    agg_df.columns = ["Model", "x_mean", "x_std", "y_mean", "y_std"]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Scatter with error bars
    ax.errorbar(
        agg_df["x_mean"], agg_df["y_mean"],
        xerr=agg_df["x_std"], yerr=agg_df["y_std"],
        fmt="none", color="gray", alpha=0.5, capsize=3, zorder=1
    )
    
    sns.scatterplot(
        data=agg_df,
        x="x_mean",
        y="y_mean",
        hue="Model",
        palette="Set2",
        s=150,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        zorder=2,
    )
    
    # Reference lines
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Per-Drug ROC-AUC (General Performance)")
    ax.set_ylabel("Pairwise Ranking Accuracy (Cliff Performance)")
    ax.set_title("General vs Activity Cliff Performance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# BONUS: Faceted Point Plot by Pair
# =============================================================================

def plot_pointplot_by_pair(pairs_df: pd.DataFrame, output_path: str):
    """
    Faceted point plot showing pairwise accuracy per pair across models.
    Great for seeing which pairs are universally hard.
    """
    g = sns.catplot(
        data=pairs_df,
        x="Model",
        y="delta",
        col="Pair",
        col_wrap=4,
        kind="point",
        palette="Set2",
        capsize=0.1,
        height=3,
        aspect=1.2,
    )
    
    # Add reference line to each facet
    for ax in g.axes.flat:
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    
    g.set_xticklabels(rotation=45, ha="right")
    g.set_axis_labels("", "Δ = P(toxic) − P(safe)")
    g.figure.suptitle("Delta by Model for Each Drug Pair", y=1.02)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate activity cliff plots (seaborn edition)")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names")
    parser.add_argument("--auto_discover", action="store_true")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dumbbell_model", type=str, default=None,
                        help="Model to use for dumbbell plot (default: first model)")
    
    args = parser.parse_args()
    
    # Setup
    setup_style()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    seeds = [int(s) for s in args.seeds.split(",")]
    models = discover_models(args.results_root) if args.auto_discover else args.models.split(",")
    
    print(f"Found {len(models)} models: {models}")
    
    # Load and prepare data
    summary_df, pairs_df = load_all_data(args.results_root, models, seeds)
    summary_df, pairs_df = prepare_data(summary_df, pairs_df)
    
    print(f"Summary: {len(summary_df)} rows | Pairs: {len(pairs_df)} rows")
    
    # Generate all plots
    fmt = args.format
    dumbbell_model = args.dumbbell_model or models[0]

    plot_grouped_bars(summary_df, f"{args.output_dir}/01_grouped_bars.{fmt}")
    plot_heatmap(pairs_df, f"{args.output_dir}/02a_heatmap_delta.{fmt}", value="delta")
    plot_heatmap(pairs_df, f"{args.output_dir}/02b_heatmap_correct.{fmt}", value="pairwise_correct")
    plot_delta_distribution(pairs_df, f"{args.output_dir}/03_delta_distribution.{fmt}")
    plot_dumbbell(pairs_df, f"{args.output_dir}/04_dumbbell.{fmt}", model=dumbbell_model)
    plot_scatter_with_regression(summary_df, f"{args.output_dir}/05_scatter.{fmt}")
    plot_pointplot_by_pair(pairs_df, f"{args.output_dir}/06_faceted_by_pair.{fmt}")

    print(f"\n✓ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()