#!/usr/bin/env python3
"""
Ordinal Activity Cliff Dumbbell Plot — Side-by-Side Comparison
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# =============================================================================
# STYLE
# =============================================================================

def setup_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "cm", 
        },
    )

COLORS = {
    "lower":     "#1565C0",
    "higher":    "#FF8F00",
    "correct":   "#2E7D32",
    "incorrect": "#C62828",
    "dist1":     "#78909C",
    "dist2":     "#37474F",
}

# =============================================================================
# HELPERS
# =============================================================================

def shorten_drug(name: str, max_len: int = 18) -> str:
    for old, short in [(" hydrochloride", " hyd..."),
                       (" mesylate", " mes..."),
                       (" bisulfate", " bis..."),
                       (" tartrate", " tar..."),
                       (" citrate", " citr..."),
                       (" sodium", " sodiu...")]:
        name = name.replace(old, short)
    if len(name) > max_len:
        return name[:max_len - 3] + "..."
    return name

def make_pair_label(row: pd.Series) -> str:
    lower = shorten_drug(row["lower_drug"])
    higher = shorten_drug(row["higher_drug"])
    return f"{lower} /\n{higher}"

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pair_label"] = df.apply(make_pair_label, axis=1)
    df["midpoint"] = (df["score_lower"] + df["score_higher"]) / 2
    df["pair_key"] = df["lower_drug"] + " | " + df["higher_drug"]
    return df

# =============================================================================
# SIDE-BY-SIDE DUMBBELL
# =============================================================================

def plot_sidebyside_dumbbell(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    label_left: str,
    label_right: str,
    output_path: str,
) -> None:
    df_left = prepare_df(df_left)
    df_right = prepare_df(df_right)

    df_left = df_left.sort_values(
        ["ordinal_distance", "midpoint"], ascending=[False, True]
    ).reset_index(drop=True)

    pair_order = df_left["pair_key"].tolist()
    df_right["_sort_key"] = df_right["pair_key"].map(
        {k: i for i, k in enumerate(pair_order)}
    )
    df_right = df_right.sort_values("_sort_key").reset_index(drop=True)

    n_pairs = len(df_left)

    all_scores = np.concatenate([
        df_left["score_lower"].values, df_left["score_higher"].values,
        df_right["score_right" if "score_right" in df_right else "score_lower"].values, 
        df_right["score_higher"].values,
    ]) if "score_right" in df_right else np.concatenate([
        df_left["score_lower"].values, df_left["score_higher"].values,
        df_right["score_lower"].values, df_right["score_higher"].values,
    ])
    
    x_min = max(0, np.min(all_scores) - 0.15)
    x_max = min(2.0, np.max(all_scores) + 0.25)

    fig_height = max(6, n_pairs * 0.65)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, fig_height), sharey=True
    )

    y_positions = np.arange(n_pairs)

    for ax, df, label in [(ax_left, df_left, label_left),
                          (ax_right, df_right, label_right)]:
        _draw_dumbbell_panel(ax, df, y_positions)
        
        ax.grid(False, axis='x')


        n_d2 = len(df_left[df_left["ordinal_distance"] == 2])
        if 0 < n_d2 < n_pairs:
            ax.axhline(n_d2 - 0.5, color="black", linestyle=":",
                        linewidth=0.8, alpha=0.4)

        ax.set_xlim(x_min, x_max)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("E[Y]")

        pw_acc = df["pairwise_correct"].mean()
        ax.text(
            0.98, 0.02, f"Pairwise Acc: {pw_acc:.0%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(df_left["pair_label"].tolist())

    for i, row in df_left.iterrows():
        y = y_positions[i]
        dist = row["ordinal_distance"]
        badge_color = COLORS["dist2"] if dist == 2 else COLORS["dist1"]
        ax_right.annotate(
            f"d={dist}", xy=(x_max + 0.02, y),
            fontsize=8, color=badge_color, fontweight="bold",
            va="center", annotation_clip=False,
        )

    # --- LEGEND MOVED TO TOP ---
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["lower"],
               markersize=10, label="Lower severity drug"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["higher"],
               markersize=10, label="Higher severity drug"),
        Line2D([0], [0], color=COLORS["correct"], linewidth=2,
               label="Correct ranking (↗)"),
        Line2D([0], [0], color=COLORS["incorrect"], linewidth=2,
               label="Incorrect ranking (↘)"),
    ]
    
    # Positioned at upper center, just below the main title
    fig.legend(
        handles=handles, loc="upper center", ncol=4,
        frameon=True, fontsize=10,
        bbox_to_anchor=(0.5, 0.98),
    )

    # Main title with more padding to accommodate legend
    fig.suptitle(
        "Activity Cliff Pair Ranking: Expected Severity Separation",
        fontsize=16, fontweight="bold", y=1.06,
    )

    # rect parameter used to push subplots down to make room for top elements
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def _draw_dumbbell_panel(ax, df: pd.DataFrame, y_positions: np.ndarray) -> None:
    for i, row in df.iterrows():
        y = y_positions[i]
        e_lower = row["score_lower"]
        e_higher = row["score_higher"]
        correct = bool(row["pairwise_correct"])
        dist = row["ordinal_distance"]

        line_color = COLORS["correct"] if correct else COLORS["incorrect"]
        line_width = 2.5 if dist == 2 else 1.5

        ax.plot([e_lower, e_higher], [y, y], color=line_color, linewidth=line_width, alpha=0.75, zorder=1)
        ax.scatter(e_lower, y, color=COLORS["lower"], s=90, zorder=2, edgecolors="white", linewidths=0.5)
        ax.scatter(e_higher, y, color=COLORS["higher"], s=90, zorder=2, edgecolors="white", linewidths=0.5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv_left", type=str, required=True)
    parser.add_argument("--pairs_csv_right", type=str, required=True)
    parser.add_argument("--label_left", type=str, default="Model A")
    parser.add_argument("--label_right", type=str, default="Model B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="png")

    args = parser.parse_args()

    setup_style()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_left = pd.read_csv(args.pairs_csv_left)
    df_right = pd.read_csv(args.pairs_csv_right)

    plot_sidebyside_dumbbell(
        df_left, df_right,
        args.label_left, args.label_right,
        os.path.join(args.output_dir, f"dumbbell_comparison.{args.format}"),
    )

if __name__ == "__main__":
    main()