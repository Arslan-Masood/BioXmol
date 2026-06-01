#!/usr/bin/env python3
"""
DILI Linear Probing with Two Evaluation Modes
==============================================

Evaluates molecular representations for Drug-Induced Liver Injury prediction using
Logistic Regression with two complementary approaches:

1. **Standard 5-Fold CV** (`--eval_mode cv`):
   General performance assessment using nested cross-validation with scaffold splitting.

2. **Activity Cliff Evaluation** (`--eval_mode activity_cliff`):
   Tests model's ability to distinguish structurally similar drugs with opposite
   hepatotoxicity profiles (held-out challenging pairs from DILIrank 2.0).

Usage:
------
# Standard CV evaluation
python DILI_linear_probing_v2.py \\
    --features_file features.csv --label_col binary_label \\
    --output_dir results/cv --seed 42

# Activity cliff evaluation
python DILI_linear_probing_v2.py \\
    --features_file features.csv --label_col binary_label \\
    --output_dir results/cliff --seed 42 \\
    --eval_mode activity_cliff --compound_name_col CompoundName
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold


# =============================================================================
# CONSTANTS
# =============================================================================

# Activity cliff pairs: (safe_drug, toxic_drug) with high structural similarity
# but divergent DILI outcomes. Sources: DILIrank 2.0 (Table 2), Chen et al. 2016 (Table 3)
'''
ACTIVITY_CLIFF_PAIRS = [
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
'''
# Manually curated activty cliff compounds
df = pd.read_csv("/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/similar_pairs_maccs_0.6_opposite_labels.csv")

# Build the pairs list
ACTIVITY_CLIFF_PAIRS = []
for _, row in df.iterrows():
    # Order as (safe, toxic) based on labels
    if row['Compound1_binary_label'] == 0:
        safe = row['Compound1_Name']
        toxic = row['Compound2_Name']
    else:
        safe = row['Compound2_Name']
        toxic = row['Compound1_Name']
    
    ACTIVITY_CLIFF_PAIRS.append((safe, toxic))

ACTIVITY_CLIFF_COMPOUNDS = list(set(
    drug for pair in ACTIVITY_CLIFF_PAIRS for drug in pair
))


# =============================================================================
# HELPER FUNCTIONS - Data Loading & Preprocessing
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names (columns starting with 'feature_')."""
    cols = [c for c in df.columns if c.startswith("feature_")]
    if not cols:
        raise ValueError("No feature columns found. Expected columns starting with 'feature_'.")
    return cols


def compute_scaffold(smiles: str) -> str:
    """Compute Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def compute_scaffolds(smiles_list: List[str]) -> List[str]:
    """Compute Murcko scaffolds for a list of SMILES."""
    return [compute_scaffold(s) for s in smiles_list]


def load_data(
    features_file: str,
    smiles_col: str,
    label_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load features and labels from CSV.
    
    Returns:
        df: Full DataFrame
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        smiles: List of SMILES strings
    """
    df = pd.read_csv(features_file).drop_duplicates(subset=[smiles_col])
    
    # Validate columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    
    # Extract features
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values
    smiles = df[smiles_col].astype(str).tolist()
    
    # Print summary
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    ratio = n_pos / n_neg if n_neg > 0 else float("inf")
    print(f"Loaded {len(df)} molecules | Features: {len(feature_cols)} | "
          f"Pos: {n_pos} | Neg: {n_neg} | Ratio: {ratio:.3f}")
    
    return df, X, y, smiles


# =============================================================================
# HELPER FUNCTIONS - Model Training
# =============================================================================

def make_param_grid(penalties: List[str], c_values: List[float]) -> List[Dict]:
    """Construct valid sklearn LogisticRegression parameter grid."""
    grid = []
    for penalty in penalties:
        if penalty == "l1":
            grid.append({"penalty": ["l1"], "C": c_values, "solver": ["liblinear"]})
        elif penalty == "l2":
            grid.append({"penalty": ["l2"], "C": c_values, "solver": ["lbfgs", "liblinear"]})
        else:
            raise ValueError(f"Unsupported penalty '{penalty}'. Use 'l1' or 'l2'.")
    return grid


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find optimal classification threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: List[str],
    c_values: List[float],
    penalties: List[str],
    inner_splits: int,
    seed: int,
) -> Tuple[LogisticRegression, float, Dict, float]:
    """
    Train LogisticRegression with GridSearchCV and optimize decision threshold.
    
    Returns:
        model: Fitted LogisticRegression with best hyperparameters
        threshold: Optimal decision threshold (Youden's J)
        best_params: Best hyperparameter dict
        best_score: Best inner CV ROC-AUC score
    """
    scaffolds = compute_scaffolds(smiles_train)
    param_grid = make_param_grid(penalties, c_values)
    
    inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    base_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    
    grid = GridSearchCV(
        base_clf,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=scaffolds)
    
    # Optimize threshold on training predictions
    train_proba = grid.best_estimator_.predict_proba(X_train)[:, 1]
    threshold = optimize_threshold(y_train, train_proba)
    
    return grid.best_estimator_, threshold, grid.best_params_, grid.best_score_


# =============================================================================
# HELPER FUNCTIONS - Metrics Computation
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict:
    """Compute comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pos_lr = sensitivity / (1 - specificity) if specificity != 1 else float("inf")
    
    # Sensitivity at 100% specificity
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
    sens_at_spec100 = float(max(tpr_curve[fpr_curve == 0], default=0.0)) if (fpr_curve == 0).any() else 0.0
    
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "pos_LR": pos_lr,
        "sensitivity_at_spec100": sens_at_spec100,
    }


def compute_metrics_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict:
    """Compute metrics with error handling for edge cases (e.g., single class)."""
    try:
        return compute_metrics(y_true, y_pred, y_proba)
    except ValueError:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "matthews_corrcoef": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "pos_LR": float("nan"),
            "sensitivity_at_spec100": float("nan"),
        }


# =============================================================================
# EVALUATION MODE 1: Standard 5-Fold Cross-Validation
# =============================================================================

def scaffold_kfold_indices(
    smiles: List[str],
    labels: np.ndarray,
    n_splits: int,
    seed: int,
):
    """Generate stratified scaffold-based k-fold indices."""
    scaffolds = compute_scaffolds(smiles)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(np.zeros(len(labels)), labels, groups=scaffolds):
        yield train_idx, val_idx


def run_kfold_cv(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    compound_name_col: str,
    output_dir: str,
    seed: int,
    c_values: List[float],
    penalties: List[str],
    inner_splits: int,
):
    """
    Run 5-fold nested cross-validation with scaffold splitting.
    
    Outer loop: 5-fold scaffold CV for unbiased evaluation
    Inner loop: 5-fold scaffold CV for hyperparameter tuning
    """
    print("\n" + "=" * 70)
    print("EVALUATION MODE: 5-Fold Nested Cross-Validation")
    print("=" * 70)
    
    df_main, df_cliff, valid_pairs = split_activity_cliff_data(df, compound_name_col)
    #df_main = df.copy()

    # Extract features from non-cliff data
    feature_cols = get_feature_columns(df_main)
    X = df_main[feature_cols].values.astype(np.float32)
    y = df_main[label_col].values
    smiles = df_main[smiles_col].astype(str).tolist()

    n_splits = 5
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(
        scaffold_kfold_indices(smiles, y, n_splits, seed), start=1
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        smiles_train = [smiles[i] for i in train_idx]
        
        # Print fold statistics
        n_train_pos, n_train_neg = (y_train == 1).sum(), (y_train == 0).sum()
        n_val_pos, n_val_neg = (y_val == 1).sum(), (y_val == 0).sum()
        print(f"\n[Fold {fold}] Train: {len(train_idx)} (pos={n_train_pos}, neg={n_train_neg}) | "
              f"Val: {len(val_idx)} (pos={n_val_pos}, neg={n_val_neg})")
        
        # Train model
        model, threshold, best_params, best_score = train_model(
            X_train, y_train, smiles_train, c_values, penalties, inner_splits, seed
        )
        
        # Evaluate on validation set
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= threshold).astype(int)
        metrics = compute_metrics(y_val, pred, proba)
        
        # Add fold info
        metrics.update({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "threshold": threshold,
            "best_penalty": best_params["penalty"],
            "best_C": best_params["C"],
            "best_solver": best_params["solver"],
            "best_inner_score": best_score,
        })
        all_metrics.append(metrics)
        
        # Print fold results
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
              f"BA: {metrics['balanced_accuracy']:.4f} | MCC: {metrics['matthews_corrcoef']:.4f}")
        print(f"    Best: {best_params['penalty']}, C={best_params['C']} | "
              f"Inner ROC-AUC: {best_score:.4f} | Threshold: {threshold:.4f}")
    
    # Save and summarize
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "cv_metrics.csv"), index=False)
    
    print("\n" + "-" * 70)
    print("MEAN CV METRICS:")
    print("-" * 70)
    summary_cols = ["roc_auc", "pr_auc", "balanced_accuracy", "matthews_corrcoef",
                    "sensitivity", "specificity", "precision"]
    for col in summary_cols:
        print(f"  {col}: {metrics_df[col].mean():.4f} ± {metrics_df[col].std():.4f}")
    
    print(f"\nResults saved to: {os.path.join(output_dir, 'cv_metrics.csv')}")


# =============================================================================
# EVALUATION MODE 2: Activity Cliff Evaluation
# =============================================================================

def split_activity_cliff_data(
    df: pd.DataFrame,
    compound_name_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    """
    Split dataset into main (training) and activity cliff (test) sets.
    
    Returns:
        df_main: Training data (activity cliff compounds removed)
        df_cliff: Test data (only activity cliff compounds)
        valid_pairs: List of complete (safe, toxic) pairs found in dataset
    """
    if compound_name_col not in df.columns:
        raise ValueError(f"Column '{compound_name_col}' not found. Available: {list(df.columns)}")
    
    dataset_compounds = set(df[compound_name_col].unique())
    found = dataset_compounds & set(ACTIVITY_CLIFF_COMPOUNDS)
    missing = set(ACTIVITY_CLIFF_COMPOUNDS) - dataset_compounds
    
    # Check pair completeness
    valid_pairs = [
        (safe, toxic) for safe, toxic in ACTIVITY_CLIFF_PAIRS
        if safe in dataset_compounds and toxic in dataset_compounds
    ]
    
    # Print summary
    print(f"\n{'='*60}")
    print("ACTIVITY CLIFF DATA SPLIT")
    print(f"{'='*60}")
    print(f"Compounds found: {len(found)}/{len(ACTIVITY_CLIFF_COMPOUNDS)}")
    print(f"Complete pairs:  {len(valid_pairs)}/{len(ACTIVITY_CLIFF_PAIRS)}")
    
    if missing:
        print(f"\nMissing compounds: {sorted(missing)}")
    
    # Split data
    cliff_mask = df[compound_name_col].isin(found)
    df_cliff = df[cliff_mask].copy()
    df_main = df[~cliff_mask].copy()
    
    print(f"\nTraining set: {len(df_main)} molecules")
    print(f"Test set:     {len(df_cliff)} molecules")
    print(f"{'='*60}")
    
    return df_main, df_cliff, valid_pairs


def evaluate_activity_cliff_pairs(
    model: LogisticRegression,
    df_cliff: pd.DataFrame,
    valid_pairs: List[Tuple[str, str]],
    label_col: str,
    compound_name_col: str,
    threshold: float,
) -> Dict:
    """
    Evaluate model on activity cliff pairs.
    
    Metrics:
        - Pairwise ranking accuracy: P(toxic) > P(safe)
        - Both correct rate: Both drugs classified correctly
        - Delta scores: P(toxic) - P(safe)
    """
    feature_cols = get_feature_columns(df_cliff)
    X = df_cliff[feature_cols].values.astype(np.float32)
    y = df_cliff[label_col].values
    
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Create compound lookup
    compound_to_idx = {
        name: idx for idx, name in enumerate(df_cliff[compound_name_col].values)
    }
    
    # Evaluate pairs
    pair_results = []
    for safe_drug, toxic_drug in valid_pairs:
        if safe_drug not in compound_to_idx or toxic_drug not in compound_to_idx:
            continue
        
        i_safe, i_toxic = compound_to_idx[safe_drug], compound_to_idx[toxic_drug]
        p_safe, p_toxic = proba[i_safe], proba[i_toxic]
        
        safe_correct = (pred[i_safe] == y[i_safe])
        toxic_correct = (pred[i_toxic] == y[i_toxic])
        
        pair_results.append({
            "safe_drug": safe_drug,
            "toxic_drug": toxic_drug,
            "p_safe": p_safe,
            "p_toxic": p_toxic,
            "delta": p_toxic - p_safe,
            "pairwise_correct": p_toxic > p_safe,
            "safe_correct": safe_correct,
            "toxic_correct": toxic_correct,
            "both_correct": safe_correct and toxic_correct,
            "safe_label": y[i_safe],
            "toxic_label": y[i_toxic],
            "safe_pred": pred[i_safe],
            "toxic_pred": pred[i_toxic],
        })
    
    # Aggregate metrics
    n_pairs = len(pair_results)
    deltas = [p["delta"] for p in pair_results]
    
    per_drug_metrics = compute_metrics_safe(y, pred, proba)
    
    return {
        "n_pairs": n_pairs,
        "pairwise_accuracy": sum(p["pairwise_correct"] for p in pair_results) / n_pairs if n_pairs else 0,
        "both_correct_rate": sum(p["both_correct"] for p in pair_results) / n_pairs if n_pairs else 0,
        "mean_delta": np.mean(deltas) if deltas else 0,
        "median_delta": np.median(deltas) if deltas else 0,
        "std_delta": np.std(deltas) if deltas else 0,
        "min_delta": np.min(deltas) if deltas else 0,
        "max_delta": np.max(deltas) if deltas else 0,
        "n_positive_delta": sum(d > 0 for d in deltas),
        "n_compounds": len(y),
        "per_drug_accuracy": per_drug_metrics["accuracy"],
        "per_drug_roc_auc": per_drug_metrics["roc_auc"],
        "per_drug_pr_auc": per_drug_metrics["pr_auc"],
        "pair_results": pair_results,
    }


def print_activity_cliff_results(results: Dict, best_params: Dict, threshold: float) -> None:
    """Pretty print activity cliff evaluation results."""
    print(f"\n{'='*70}")
    print("ACTIVITY CLIFF EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nModel: penalty={best_params['penalty']}, C={best_params['C']}, threshold={threshold:.4f}")
    
    print(f"\n--- Pairwise Metrics (n={results['n_pairs']} pairs) ---")
    print(f"  Pairwise Ranking Accuracy:  {results['pairwise_accuracy']:.1%}")
    print(f"  Both Correct Rate:          {results['both_correct_rate']:.1%}")
    print(f"  Mean Δ (P_toxic - P_safe):  {results['mean_delta']:.4f}")
    print(f"  Positive Δ Count:           {results['n_positive_delta']}/{results['n_pairs']}")
    
    print(f"\n--- Per-Drug Metrics (n={results['n_compounds']} compounds) ---")
    print(f"  Accuracy:  {results['per_drug_accuracy']:.4f}")
    print(f"  ROC-AUC:   {results['per_drug_roc_auc']:.4f}")
    print(f"  PR-AUC:    {results['per_drug_pr_auc']:.4f}")
    
    print(f"\n--- Per-Pair Breakdown ---")
    print(f"{'Safe Drug':<32} {'Toxic Drug':<32} {'P(s)':<7} {'P(t)':<7} {'Δ':<7} {'Rank':<5} {'Both':<5}")
    print("-" * 100)
    
    for p in results["pair_results"]:
        rank_sym = "✓" if p["pairwise_correct"] else "✗"
        both_sym = "✓" if p["both_correct"] else "✗"
        print(f"{p['safe_drug']:<32} {p['toxic_drug']:<32} "
              f"{p['p_safe']:<7.3f} {p['p_toxic']:<7.3f} {p['delta']:<7.3f} "
              f"{rank_sym:<5} {both_sym:<5}")
    
    print(f"{'='*70}\n")


def run_activity_cliff_evaluation(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    compound_name_col: str,
    output_dir: str,
    seed: int,
    c_values: List[float],
    penalties: List[str],
    inner_splits: int,
) -> None:
    """
    Run activity cliff evaluation pipeline:
    
    1. Split data: hold out activity cliff compounds
    2. Train: GridSearchCV on main dataset
    3. Optimize: Find threshold via Youden's J
    4. Evaluate: Test on activity cliff pairs
    """
    print("\n" + "=" * 70)
    print("EVALUATION MODE: Activity Cliff")
    print("=" * 70)
    
    # Step 1: Split data
    df_main, df_cliff, valid_pairs = split_activity_cliff_data(df, compound_name_col)
    
    if not valid_pairs:
        raise ValueError("No complete activity cliff pairs found in dataset!")
    
    # Prepare training data
    feature_cols = get_feature_columns(df_main)
    X_main = df_main[feature_cols].values.astype(np.float32)
    y_main = df_main[label_col].values
    smiles_main = df_main[smiles_col].astype(str).tolist()
    
    # Step 2 & 3: Train model
    print(f"\nTraining on {len(X_main)} molecules...")
    model, threshold, best_params, best_score = train_model(
        X_main, y_main, smiles_main, c_values, penalties, inner_splits, seed
    )
    print(f"  Best: {best_params['penalty']}, C={best_params['C']} | "
          f"Inner ROC-AUC: {best_score:.4f} | Threshold: {threshold:.4f}")
    
    # Step 4: Evaluate on activity cliff pairs
    results = evaluate_activity_cliff_pairs(
        model, df_cliff, valid_pairs, label_col, compound_name_col, threshold
    )
    
    # Print results
    print_activity_cliff_results(results, best_params, threshold)
    
    # Save results
    summary = {
        "n_pairs": results["n_pairs"],
        "pairwise_accuracy": results["pairwise_accuracy"],
        "both_correct_rate": results["both_correct_rate"],
        "mean_delta": results["mean_delta"],
        "median_delta": results["median_delta"],
        "std_delta": results["std_delta"],
        "n_positive_delta": results["n_positive_delta"],
        "per_drug_accuracy": results["per_drug_accuracy"],
        "per_drug_roc_auc": results["per_drug_roc_auc"],
        "per_drug_pr_auc": results["per_drug_pr_auc"],
        "n_train": len(X_main),
        "n_test": results["n_compounds"],
        "best_penalty": best_params["penalty"],
        "best_C": best_params["C"],
        "best_solver": best_params["solver"],
        "inner_cv_score": best_score,
        "threshold": threshold,
        "seed": seed,
    }
    
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "activity_cliff_summary.csv"), index=False
    )
    pd.DataFrame(results["pair_results"]).to_csv(
        os.path.join(output_dir, "activity_cliff_pairs.csv"), index=False
    )
    
    print(f"Results saved to:")
    print(f"  {os.path.join(output_dir, 'activity_cliff_summary.csv')}")
    print(f"  {os.path.join(output_dir, 'activity_cliff_pairs.csv')}")


# =============================================================================
# MAIN
# =============================================================================

def parse_comma_floats(value: str) -> List[float]:
    """Parse comma-separated float values."""
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_comma_strings(value: str) -> List[str]:
    """Parse comma-separated string values."""
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="DILI Linear Probing with CV and Activity Cliff Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--features_file", type=str, required=True,
                        help="Path to features CSV")
    parser.add_argument("--label_col", type=str, required=True,
                        help="Name of binary label column")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    
    # Optional arguments
    parser.add_argument("--smiles_col", type=str, default="SMILES_Normalized",
                        help="Name of SMILES column (default: SMILES_Normalized)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--c_values", type=parse_comma_floats,
                        default="0.0001,0.001,0.01,0.1,1.0",
                        help="Comma-separated C values (default: 0.0001,0.001,0.01,0.1,1.0)")
    parser.add_argument("--penalties", type=parse_comma_strings,
                        default="l1,l2",
                        help="Comma-separated penalties (default: l1,l2)")
    parser.add_argument("--inner_splits", type=int, default=5,
                        help="Inner CV splits for hyperparameter tuning (default: 5)")
    parser.add_argument("--compound_name_col", type=str, default="CompoundName",
                        help="Compound name column (for activity_cliff mode)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Read data file
    df = pd.read_csv(args.features_file).drop_duplicates(subset=[args.smiles_col])

    # Run 5-fold CV evaluation (automatically holds out activity cliff compounds)
    run_kfold_cv(
            df=df,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
            compound_name_col=args.compound_name_col,
            output_dir=args.output_dir,
            seed=args.seed,
            c_values=args.c_values,
            penalties=args.penalties,
            inner_splits=args.inner_splits,
        )
    # run activity cliff evaluation
    run_activity_cliff_evaluation(
        df=df,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        compound_name_col=args.compound_name_col,
        output_dir=args.output_dir,
        seed=args.seed,
        c_values=args.c_values,
        penalties=args.penalties,
        inner_splits=args.inner_splits,
    )

        


if __name__ == "__main__":
    main()