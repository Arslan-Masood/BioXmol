#!/usr/bin/env python3
"""
Plot PCA of activity cliff molecules
====================================

This script:
  - Loads a DILI feature CSV (same format used in `DILI_linear_probing_updated.py`)
  - Selects only activity cliff compounds
  - Projects their features into 2D using PCA
  - Colors points by label (safe vs toxic)
  - Connects each activity cliff pair with a dotted line
  - Annotates each line with Tanimoto similarity based on MACCS fingerprints

Example:
--------
python plot_activity_cliff_pca.py \\
    --features_file /path/to/DILIrank_2.0_normalized_ECFP_1024_2.csv \\
    --output_path /path/to/output/activity_cliff_pca_ECFP.png \\
    --label_col binary_label \\
    --smiles_col SMILES_Normalized \\
    --compound_name_col Name
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys


# =============================================================================
# ACTIVITY CLIFF DEFINITIONS (copied from DILI_linear_probing_updated.py)
# =============================================================================

# Activity cliff pairs: (safe_drug, toxic_drug) with high structural similarity
# but divergent DILI outcomes. Sources: DILIrank 2.0 (Table 2), Chen et al. 2016 (Table 3)
ACTIVITY_CLIFF_PAIRS: List[Tuple[str, str]] = [
    # DILIrank 2.0 - NEW pairs
    ("Enzalutamide", "Nilutamide"),
    ("Rifamycin sodium", "Rifampin"),
    ("Sarecycline hydrochloride", "Tigecycline"),
    ("Sarecycline hydrochloride", "Minocycline hydrochloride"),
    ("Voclosporin", "Cyclosporine"),
    # Chen et al. 2016 - ORIGINAL pairs
    ("Minocycline hydrochloride", "Doxycycline"),
    ("Trovafloxacin mesylate", "Moxifloxacin hydrochloride"),
    ("Benzbromarone", "Amiodarone hydrochloride"),
    ("Ticlopidine hydrochloride", "Clopidogrel bisulfate"),
    ("Ibufenac", "Ibuprofen"),
    ("Alpidem", "Zolpidem tartrate"),
    ("Ticrynafen", "Ethacrynic acid"),
    ("Tolcapone", "Entacapone"),
    ("Cyclofenil", "Clomiphene citrate"),
    ("Troglitazone", "Pioglitazone hydrochloride"),
]

ACTIVITY_CLIFF_COMPOUNDS = list(
    set(drug for pair in ACTIVITY_CLIFF_PAIRS for drug in pair)
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names (columns starting with 'feature_')."""
    cols = [c for c in df.columns if c.startswith("feature_")]
    if not cols:
        raise ValueError(
            "No feature columns found. Expected columns starting with 'feature_'."
        )
    return cols


def compute_maccs_tanimoto(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two SMILES using MACCS fingerprints."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return float("nan")
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def load_activity_cliff_data(
    features_file: str,
    smiles_col: str,
    label_col: str,
    compound_name_col: str,
) -> pd.DataFrame:
    """
    Load features and filter to activity cliff compounds present in the dataset.
    """
    df = pd.read_csv(features_file)

    # Basic validation
    for col in [smiles_col, label_col, compound_name_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {features_file}.")

    # Keep unique molecules by SMILES (consistent with other scripts)
    df = df.drop_duplicates(subset=[smiles_col])

    dataset_compounds = set(df[compound_name_col].unique())
    found = dataset_compounds & set(ACTIVITY_CLIFF_COMPOUNDS)
    missing = set(ACTIVITY_CLIFF_COMPOUNDS) - dataset_compounds

    print(f"Found {len(found)}/{len(ACTIVITY_CLIFF_COMPOUNDS)} activity cliff compounds.")
    if missing:
        print("Missing compounds (not present in this features file):")
        print(sorted(missing))

    cliff_mask = df[compound_name_col].isin(found)
    df_cliff = df[cliff_mask].copy()

    print(f"Number of activity cliff molecules in this file: {len(df_cliff)}")
    return df_cliff


def plot_activity_cliff_pca(
    df_cliff: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    compound_name_col: str,
    output_path: str,
    title: str = None,
) -> None:
    """Perform PCA and generate the activity cliff plot."""
    if df_cliff.empty:
        raise ValueError("No activity cliff molecules found to plot.")

    feature_cols = get_feature_columns(df_cliff)
    X = df_cliff[feature_cols].values.astype(np.float32)
    labels = df_cliff[label_col].values
    names = df_cliff[compound_name_col].astype(str).values
    smiles = df_cliff[smiles_col].astype(str).values

    # PCA to 2D
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)

    df_plot = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels,
            "name": names,
            "smiles": smiles,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Label mapping (assumes 0 = safe, 1 = toxic; falls back to numeric otherwise)
    unique_labels = sorted(pd.unique(df_plot["label"]))
    for lab in unique_labels:
        sub = df_plot[df_plot["label"] == lab]
        if lab == 0:
            color, marker, lab_name = "tab:blue", "o", "Safe (0)"
        elif lab == 1:
            color, marker, lab_name = "tab:red", "^", "Toxic (1)"
        else:
            color, marker, lab_name = "tab:gray", "s", f"Label {lab}"

        ax.scatter(
            sub["x"],
            sub["y"],
            c=color,
            marker=marker,
            label=lab_name,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.5,
        )

    # Index lookup by compound name
    name_to_idx: Dict[str, int] = {
        n: i for i, n in enumerate(df_plot["name"].values)
    }

    # Connect each valid pair with a dotted line and annotate Tanimoto similarity
    for safe_drug, toxic_drug in ACTIVITY_CLIFF_PAIRS:
        if safe_drug not in name_to_idx or toxic_drug not in name_to_idx:
            continue

        i_safe = name_to_idx[safe_drug]
        i_toxic = name_to_idx[toxic_drug]

        x1, y1 = df_plot.loc[i_safe, ["x", "y"]]
        x2, y2 = df_plot.loc[i_toxic, ["x", "y"]]

        # Draw dotted line
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle="--",
            color="gray",
            linewidth=1.0,
            alpha=0.8,
        )

        # Compute MACCS Tanimoto similarity
        s1 = df_plot.loc[i_safe, "smiles"]
        s2 = df_plot.loc[i_toxic, "smiles"]
        tanimoto = compute_maccs_tanimoto(s1, s2)

        if np.isfinite(tanimoto):
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(
                xm,
                ym,
                f"{tanimoto:.2f}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.8,
                ),
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if title is None:
        title = "PCA of Activity Cliff Molecules"
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved PCA plot to: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCA plot of activity cliff molecules with MACCS Tanimoto annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--features_file",
        type=str,
        required=True,
        help="Path to features CSV (same format as used in DILI linear probing).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the PCA plot (e.g., /path/to/plot.png).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="binary_label",
        help="Name of binary label column (default: binary_label).",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="SMILES_Normalized",
        help="Name of SMILES column (default: SMILES_Normalized).",
    )
    parser.add_argument(
        "--compound_name_col",
        type=str,
        default="Name",
        help="Compound name column (default: Name).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the plot.",
    )

    args = parser.parse_args()

    df_cliff = load_activity_cliff_data(
        features_file=args.features_file,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        compound_name_col=args.compound_name_col,
    )

    plot_activity_cliff_pca(
        df_cliff=df_cliff,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        compound_name_col=args.compound_name_col,
        output_path=args.output_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()


