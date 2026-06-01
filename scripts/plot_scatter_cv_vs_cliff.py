#!/usr/bin/env python3
"""
Publication-Ready Scatter: CV ROC-AUC vs Activity Cliff Accuracy
(With Auto-Jitter and 2-Decimal Axis Formatting)
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
})

NAME_MAPPING = {
    "GNN_Soft_Clip_with_Teacher_seed2_second_fc":   r"Bio$\mathcal{X}$Mol-Soft",
    "GNN_Vanilla_Clip_without_VAE_seed2_second_fc": r"Bio$\mathcal{X}$Mol-Hard",
    "ECFP_1024_2":                                  "ECFP",
    "MoLFormer-XL-both-10pct":                      "MoLFormer",
    "ChemBERTa-MTR":                                "ChemBERTa-MTR",
    "ChemBERTa-MLM":                                "ChemBERTa-MLM",
}

STYLE = {
    "GNN_Soft_Clip_with_Teacher_seed2_second_fc":   ("#D62728", "*", 300), 
    "GNN_Vanilla_Clip_without_VAE_seed2_second_fc": ("#FF7F0E", "*", 300), 
    "ECFP_1024_2":                                  ("#1F77B4", "D", 160), 
    "MoLFormer-XL-both-10pct":                      ("#2CA02C", "s", 160), 
    "ChemBERTa-MTR":                                ("#9467BD", "^", 160), 
    "ChemBERTa-MLM":                                ("#8C564B", "v", 160), 
}
DEFAULT_STYLE = ("#7F7F7F", "o", 100)

# Still helpful to nudge labels away from the jittered point
LABEL_OFFSETS = {
    "Bio$\mathcal{X}$Mol-Soft": (0.003, 0.01),
    "Bio$\mathcal{X}$Mol-Hard": (0.003, 0.01),
    "ECFP": (0.003, -0.02),
    "MoLFormer": (-0.02, 0.015),          # Moved up/left
    "ChemBERTa-MTR": (0.003, -0.025),
    "ChemBERTa-MLM": (0.005, 0.005),      # Moved down/right
}

def parse_args():
    parser = argparse.ArgumentParser(description="Scatter: CV vs Cliff metrics")
    parser.add_argument("--cv_file", type=str, required=True)
    parser.add_argument("--ac_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output_file", type=str, default="scatter_cv_vs_cliff_final.png")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducible "auto-jitter"
    np.random.seed(42)

    # 1. LOAD & MERGE
    cv = pd.read_csv(args.cv_file)
    ac = pd.read_csv(args.ac_file)
    merged = cv.merge(ac, on="feature")
    
    models_to_plot = args.models if args.models else list(NAME_MAPPING.keys())
    df = merged[merged["feature"].isin(models_to_plot)].copy()
    
    if df.empty:
        print("No matching models found.")
        return

    df["display_name"] = df["feature"].map(lambda x: NAME_MAPPING.get(x, x))

    # 2. PLOT
    fig, ax = plt.subplots(figsize=(7, 6))

    # --- AUTO JITTER LOGIC ---
    # We apply a tiny random noise to coordinates to prevent exact overlap
    jitter_strength = 0.005  # Adjust this if they are still overlapping
    
    # Store jittered coordinates so labels match points
    df['x_jittered'] = df["roc_auc_macro_ovr"] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    df['y_jittered'] = df["pairwise_accuracy"] + np.random.uniform(-jitter_strength, jitter_strength, len(df))

    for _, row in df.iterrows():
        feat = row["feature"]
        color, marker, size = STYLE.get(feat, DEFAULT_STYLE)
        
        ax.scatter(
            row['x_jittered'],
            row['y_jittered'],
            c=color, marker=marker, s=size,
            edgecolors="white", linewidths=0.8,
            zorder=5
        )

    # 3. ANNOTATIONS
    for _, row in df.iterrows():
        name = row["display_name"]
        x = row['x_jittered']
        y = row['y_jittered']
        
        dx, dy = LABEL_OFFSETS.get(name, (0.005, 0.005))
        ax.text(x + dx, y + dy, name, fontsize=11, fontweight='medium', zorder=6)

    # 4. HIGHLIGHT ZONES
    # Failure Zone
    ax.axhline(y=0.5, ls="--", color="#555555", alpha=0.6, lw=1.2, zorder=1)
    ax.fill_between([0.5, 0.7], 0.35, 0.5, color='red', alpha=0.1, zorder=0)
    ax.text(0.505, 0.48, "Failure Zone (Worse than random)", color='#D62728', fontsize=11, va='top', fontweight='medium')

    # 5. FORMATTING
    ax.set_xlabel("CV ROC-AUC (Macro OvR)", fontweight='bold')
    ax.set_ylabel("Activity Cliff Pairwise Accuracy", fontweight='bold')
    
    # --- AXIS FORMATTING FIX ---
    # Force 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.set_xlim(0.5, 0.7)
    ax.set_ylim(0.35, 0.85)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='-', alpha=0.15, color='black', zorder=0)
    ax.tick_params(width=1.2)

    plt.tight_layout()
    
    # 6. SAVE
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.output_file)
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved jittered figure: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()