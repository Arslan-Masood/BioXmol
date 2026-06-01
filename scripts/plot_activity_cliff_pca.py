#!/usr/bin/env python3
"""
Publication-Ready PCA Activity Cliff Plot (Geometric Shapes & External Legend)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
})

# --- PAIR DEFINITIONS ---
PAIRS_D2 = [
    ("Enzalutamide", "Nilutamide"),
    ("Sarecycline hydrochloride", "Minocycline hydrochloride"),
    ("Sarecycline hydrochloride", "Tigecycline"),
    ("Voclosporin", "Cyclosporine"),
    ("Rifamycin sodium", "Rifampin"),
]

PAIRS_D1 = [
    ("Ibuprofen", "Ibufenac"),
    ("Pioglitazone hydrochloride", "Troglitazone"),
    ("Entacapone", "Tolcapone"),
    ("Clopidogrel bisulfate", "Ticlopidine hydrochloride"),
    ("Moxifloxacin hydrochloride", "Trovafloxacin mesylate"),
    ("Clomiphene citrate", "Cyclofenil"),
    ("Doxycycline", "Minocycline hydrochloride"),
]

ALL_PAIRS = PAIRS_D2 + PAIRS_D1
ALL_COMPOUNDS = list(set(d for p in ALL_PAIRS for d in p))

# Distinct markers to assign to pairs
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']

# =============================================================================
# HELPERS
# =============================================================================
def maccs_tanimoto(smiles_a, smiles_b):
    mol_a = Chem.MolFromSmiles(str(smiles_a))
    mol_b = Chem.MolFromSmiles(str(smiles_b))
    if mol_a is None or mol_b is None: return float("nan")
    return DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(mol_a), MACCSkeys.GenMACCSKeys(mol_b))

def clean_name(name):
    replacements = {
        " hydrochloride": " HCl", " mesylate": " mes.", " bisulfate": " bis.",
        " tartrate": " tart.", " sodium": " Na", " citrate": " cit."
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def get_euclidean_distance(pos_a, pos_b):
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b))

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="activity_cliff_geometric.pdf")
    args = parser.parse_args()

    # 1. LOAD & PREP
    df = pd.read_csv(args.features_file).drop_duplicates(subset=["SMILES_Normalized"])
    dataset_compounds = set(df["Name"].unique())
    found = dataset_compounds & set(ALL_COMPOUNDS)
    df_cliff = df[df["Name"].isin(found)].copy().reset_index(drop=True)
    
    # 2. PCA
    feature_cols = [c for c in df_cliff.columns if c.startswith("feature_")]
    X = df_cliff[feature_cols].values.astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    name_to_info = {}
    for i, row in df_cliff.iterrows():
        name_to_info[row["Name"]] = {
            "x": X_2d[i, 0], "y": X_2d[i, 1],
            "smiles": row["SMILES_Normalized"], "label": row["vDILI-Concern_standardized"]
        }

    # 3. PLOT SETUP
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)

    colors = {
        "vno-dili-concern": "#2ecc71",     # Green
        "vless-dili-concern": "#f39c12",   # Orange
        "vmost-dili-concern": "#e74c3c"    # Red
    }
    
    panels = [
        {"ax": axes[0], "title_base": "A. No-DILI vs. Most-DILI", "pair_list": PAIRS_D2},
        {"ax": axes[1], "title_base": "B. Less-DILI vs. Most-DILI", "pair_list": PAIRS_D1}
    ]

    for panel in panels:
        ax = panel["ax"]
        pair_list = panel["pair_list"]
        
        distances = []
        legend_elements = []
        
        # --- ITERATE PAIRS IN THIS PANEL ---
        for i, (drug_a, drug_b) in enumerate(pair_list):
            if drug_a not in name_to_info or drug_b not in name_to_info: continue
            
            info_a, info_b = name_to_info[drug_a], name_to_info[drug_b]
            marker = MARKERS[i % len(MARKERS)] # Assign unique shape
            
            # 1. Calculate Distance & Similarity
            dist = get_euclidean_distance((info_a["x"], info_a["y"]), (info_b["x"], info_b["y"]))
            distances.append(dist)
            ts = maccs_tanimoto(info_a["smiles"], info_b["smiles"])
            
            # 2. Draw Connection Line
            ax.plot([info_a["x"], info_b["x"]], [info_a["y"], info_b["y"]],
                    linestyle="--", color="#999999", linewidth=1.0, alpha=0.5, zorder=1)
            
            # 3. Draw Points (Drug A)
            ax.scatter(info_a["x"], info_a["y"], 
                       c=colors[info_a["label"]], marker=marker,
                       s=180, edgecolors="black", linewidth=0.8, zorder=3)
            
            # 4. Draw Points (Drug B)
            ax.scatter(info_b["x"], info_b["y"], 
                       c=colors[info_b["label"]], marker=marker,
                       s=180, edgecolors="black", linewidth=0.8, zorder=3)
            
            # 5. Add to Legend
            # We create a black marker for the legend to represent the pair shape
            label_text = f"{clean_name(drug_a)} / {clean_name(drug_b)} (TS={ts:.2f})"
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=label_text,
                                          markerfacecolor='gray', markersize=10, markeredgecolor='black'))

        # --- PANEL METRICS ---
        mean_dist = np.mean(distances) if distances else 0.0
        ax.set_title(f"{panel['title_base']}\n(Mean Euclidean Dist: {mean_dist:.2f})", 
                     fontsize=12, fontweight="bold", pad=15)

        # --- FORMATTING ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='-', alpha=0.1, color='black')
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontweight='bold')
        
        # --- LEGEND OUTSIDE ---
        # Place legend to the right of the subplot
        # bbox_to_anchor=(x, y) coordinates are in axes fraction
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, -0.15),
                  ncol=1, frameon=False, fontsize=9, title="Activity Cliff Pairs (Shape ID)")

    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontweight='bold')
    
    # Adjust layout to make room for the bottom legends
    plt.tight_layout()
    # Extra adjustment to ensure bottom legends aren't cut off
    plt.subplots_adjust(bottom=0.35) 
    
    plt.savefig(args.output_file, dpi=600, bbox_inches="tight")
    print(f"Saved geometric figure with external legend: {args.output_file}")

if __name__ == "__main__":
    main()