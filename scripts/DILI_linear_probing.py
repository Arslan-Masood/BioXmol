#!/usr/bin/env python3
"""
Linear probing script with true 5-fold stratified scaffold CV.
Uses Murcko scaffolds + StratifiedGroupKFold (n_splits=5).

Works for any tabular feature CSV that contains:
  - A SMILES column (default: SMILES_Normalized)
  - One label column (binary) specified via --label_col
  - Feature columns (all remaining non-id columns are treated as features)

Usage example:
  python scripts/DILI_linear_probing.py \
      --features_file /path/to/features_with_labels_and_features.csv \
      --label_col binary_label \
      --smiles_col SMILES_Normalized \
      --output_dir results/linear_probe \
      --seed 42
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

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
from sklearn.metrics import make_scorer


def load_data(features_file: str, smiles_col: str, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Load features and labels from a single CSV and return dataframe, X, y, smiles."""
    df = pd.read_csv(features_file).drop_duplicates(subset=[smiles_col])
    print(f"unique smiles in features file: {df[smiles_col].nunique()}")

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in features file.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in features file.")

    # Dataset summary
    n_total = len(df)
    label_counts = df[label_col].value_counts().to_dict()
    n_pos = label_counts.get(1, 0)
    n_neg = label_counts.get(0, 0)
    ratio = (n_pos / n_neg) if n_neg > 0 else float("inf")
    print(f"After merge: total molecules={n_total}, label 1={n_pos}, label 0={n_neg}, pos/neg ratio={ratio:.3f}")

    # Identify feature columns: those starting with "feature_"
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns starting with 'feature_').")
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values
    smiles_list = df[smiles_col].astype(str).tolist()
    return df, X, y, smiles_list


def parse_comma_separated_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_comma_separated_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def compute_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_kfold_indices(smiles: List[str], labels: np.ndarray, n_splits: int, seed: int):
    scaffolds = [compute_scaffold(s) for s in smiles]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(np.zeros(len(labels)), labels, groups=scaffolds):
        yield train_idx, val_idx


def make_param_grid(penalties: List[str], c_values: List[float]):
    """Construct a solver/penalty/C grid that is valid for sklearn LogisticRegression."""
    grid = []
    for penalty in penalties:
        if penalty == "l1":
            grid.append({"penalty": ["l1"], "C": c_values, "solver": ["liblinear"]})
        elif penalty == "l2":
            grid.append({"penalty": ["l2"], "C": c_values, "solver": ["lbfgs", "liblinear"]})
        else:
            raise ValueError(f"Unsupported penalty '{penalty}'. Use one of ['l1', 'l2'].")
    return grid


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    smiles: List[str],
    output_dir: str,
    seed: int,
    c_values: List[float],
    penalties: List[str],
    inner_splits: int = 3,
):
    n_splits = 5
    metrics = []
    param_grid = make_param_grid(penalties, c_values)

    for fold, (train_idx, val_idx) in enumerate(scaffold_kfold_indices(smiles, y, n_splits=n_splits, seed=seed), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_smiles = [smiles[i] for i in train_idx]

        # Fold label statistics
        n_train_pos = (y_train == 1).sum()
        n_train_neg = (y_train == 0).sum()
        n_val_pos = (y_val == 1).sum()
        n_val_neg = (y_val == 0).sum()
        train_ratio = (n_train_pos / n_train_neg) if n_train_neg > 0 else float("inf")
        val_ratio = (n_val_pos / n_val_neg) if n_val_neg > 0 else float("inf")
        print(f"[Fold {fold}] train={len(train_idx)} (pos={n_train_pos}, neg={n_train_neg}, pos/neg={train_ratio:.3f}); "
              f"val={len(val_idx)} (pos={n_val_pos}, neg={n_val_neg}, pos/neg={val_ratio:.3f})")

        # Inner CV for hyperparameter search (nested CV)
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
        grid.fit(X_train, y_train, groups=[compute_scaffold(s) for s in train_smiles])
        best_clf = grid.best_estimator_

        # Optimize decision threshold on training fold using Youden's J (tpr - fpr)
        train_proba = best_clf.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, train_proba)
        j_scores = tpr - fpr
        best_thresh_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_thresh_idx]

        proba = best_clf.predict_proba(X_val)[:, 1]
        pred = (proba >= best_thresh).astype(int)

        # Extended metrics to mirror RF script
        cm = confusion_matrix(y_val, pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pos_lr = sensitivity / (1 - specificity) if specificity != 1 else float("inf")
        ba = balanced_accuracy_score(y_val, pred)
        mcc = matthews_corrcoef(y_val, pred)
        ppv = precision_score(y_val, pred, zero_division=0)
        avg_precision = average_precision_score(y_val, proba)

        # Sensitivity at 100% specificity (requires fp=0)
        fpr_curve, tpr_curve, _ = roc_curve(y_val, proba)
        sens_at_spec100 = max(tpr_curve[fpr_curve == 0], default=0.0) if (fpr_curve == 0).any() else 0.0

        fold_metrics = {
            "fold": fold,
            "roc_auc": roc_auc_score(y_val, proba),
            "pr_auc": avg_precision,
            "accuracy": accuracy_score(y_val, pred),
            "f1": f1_score(y_val, pred),
            "balanced_accuracy": ba,
            "matthews_corrcoef": mcc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "pos_LR": pos_lr,
            "precision": ppv,
            "sensitivity_at_spec100": sens_at_spec100,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_penalty": grid.best_params_.get("penalty"),
            "best_C": grid.best_params_.get("C"),
            "best_solver": grid.best_params_.get("solver"),
            "best_inner_mean_score": grid.best_score_,
        }
        metrics.append(fold_metrics)
        print(f"[Fold {fold}] ROC-AUC={fold_metrics['roc_auc']:.4f} "
              f"PR-AUC={fold_metrics['pr_auc']:.4f} "
              f"Acc={fold_metrics['accuracy']:.4f} F1={fold_metrics['f1']:.4f} "
              f"BA={fold_metrics['balanced_accuracy']:.4f} MCC={fold_metrics['matthews_corrcoef']:.4f} "
              f"Sens={fold_metrics['sensitivity']:.4f} Spec={fold_metrics['specificity']:.4f} "
              f"PPV={fold_metrics['precision']:.4f} PosLR={fold_metrics['pos_LR']:.4f} "
              f"Sens@Spec100={fold_metrics['sensitivity_at_spec100']:.4f} "
              f"Thresh={best_thresh:.4f} "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"[Fold {fold}] Best params: penalty={fold_metrics['best_penalty']}, C={fold_metrics['best_C']}, "
              f"solver={fold_metrics['best_solver']}, inner ROC-AUC={fold_metrics['best_inner_mean_score']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "cv_metrics.csv"), index=False)

    # Summary
    summary_metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "balanced_accuracy",
        "matthews_corrcoef",
        "sensitivity",
        "specificity",
        "precision",
        "pos_LR",
        "sensitivity_at_spec100",
    ]
    summary = metrics_df[summary_metrics].mean().to_dict()
    print("\nMean CV Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Linear probing with 5-fold stratified scaffold CV")
    parser.add_argument("--features_file", type=str, required=True, help="Path to features CSV")
    parser.add_argument("--label_col", type=str, required=True, help="Name of the label column (binary)")
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="SMILES_Normalized",
        help="Name of the SMILES column (must be present in the features file)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CV metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--c_values",
        type=parse_comma_separated_floats,
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated list of C values for grid search",
    )
    parser.add_argument(
        "--penalties",
        type=parse_comma_separated_strings,
        default="l1,l2",
        help="Comma-separated list of penalties to consider (subset of l1,l2)",
    )
    parser.add_argument(
        "--inner_splits",
        type=int,
        default=5,
        help="Number of inner CV splits for hyperparameter search",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _, X, y, smiles = load_data(args.features_file, args.smiles_col, args.label_col)
    run_kfold_cv(
        X,
        y,
        smiles,
        args.output_dir,
        args.seed,
        args.c_values if isinstance(args.c_values, list) else parse_comma_separated_floats(args.c_values),
        args.penalties if isinstance(args.penalties, list) else parse_comma_separated_strings(args.penalties),
        args.inner_splits,
    )


if __name__ == "__main__":
    main()

