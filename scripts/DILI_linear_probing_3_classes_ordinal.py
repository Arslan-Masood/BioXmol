#!/usr/bin/env python3
"""
DILI 3-Class Ordinal Regression - Unified CV + Activity Cliff Evaluation
=========================================================================

Evaluates molecular representations for Drug-Induced Liver Injury prediction
using **ordinal logistic regression** (Frank & Hall 2001 decomposition) with
three ordered severity classes:

    0 = vno-dili-concern
    1 = vless-dili-concern
    2 = vmost-dili-concern

The ordinal approach trains k-1 binary classifiers for k ordered classes:
    Classifier 0: P(Y > 0)  — separates vNo from {vLess, vMost}
    Classifier 1: P(Y > 1)  — separates {vNo, vLess} from vMost

Class probabilities are derived from cumulative probabilities with
monotonicity enforcement, producing a proper (n, 3) probability matrix.

Two evaluations run in sequence:

1. **5-Fold Nested CV** on non-cliff molecules
   Metrics: accuracy, balanced accuracy, weighted-kappa, macro/weighted F1,
            Spearman correlation, macro OvR ROC-AUC, confusion matrix

2. **Activity Cliff Evaluation** on held-out pairs
   Ranking score: E[Y] = proba @ [0, 1, 2]  (expected severity)
   Pairs stratified by ordinal distance (1=adjacent, 2=extreme)

Usage:
------
    python DILI_ordinal_regression_3_classes.py \\
        --features_file features.csv \\
        --label_col vDILI-Concern_standardized \\
        --compound_name_col Name \\
        --output_dir results/ \\
        --seed 42
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold


# =============================================================================
# ORDINAL LABEL SCHEME
# =============================================================================

ORDINAL_LABEL_MAP = {
    "vno-dili-concern":   0,
    "vless-dili-concern": 1,
    "vmost-dili-concern": 2,
}
ORDINAL_LABEL_NAMES = {v: k for k, v in ORDINAL_LABEL_MAP.items()}
SEVERITY_WEIGHTS = np.array([0.0, 1.0, 2.0])  # for E[Y] score


# =============================================================================
# ORDINAL CLASSIFIER (Frank & Hall 2001)
# =============================================================================

#
class OrdinalClassifier(ClassifierMixin, BaseEstimator):  # <--- Mixin MUST be first
    """
    Ordinal classification via cumulative binary decomposition.
    """
    # Explicit tag is good, but inheritance order is critical
    _estimator_type = "classifier"

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def _get_base(self):
        # Helper to ensure we always have a valid base
        if self.base_estimator is None:
            # Default to LR if none provided
            return LogisticRegression(max_iter=2000) 
        return self.base_estimator

    def fit(self, X, y, **fit_params):
        fit_params.pop("groups", None) # Clean up kwargs

        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(f"Need >= 2 classes, got {self.n_classes_}.")

        self.classifiers_ = []
        self.degenerate_values_ = []

        for i in range(self.n_classes_ - 1):
            # Binary target: 1 if y > classes_[i], else 0
            binary_y = (y > self.classes_[i]).astype(int)
            n_unique = len(np.unique(binary_y))

            if n_unique < 2:
                self.classifiers_.append(None)
                self.degenerate_values_.append(float(binary_y[0]))
            else:
                clf = clone(self._get_base()) # Clone ensures fresh state
                clf.fit(X, binary_y, **fit_params)
                self.classifiers_.append(clf)
                self.degenerate_values_.append(None)

        return self

    def predict_proba(self, X):
        n = X.shape[0]
        k = self.n_classes_

        # 1. Compute cumulative probabilities
        cum_proba = np.zeros((n, k - 1))
        for i, (clf, degen_val) in enumerate(zip(self.classifiers_, self.degenerate_values_)):
            if clf is None:
                cum_proba[:, i] = degen_val
            else:
                # Use [:, 1] for probability of class 1 (Target > Threshold)
                cum_proba[:, i] = clf.predict_proba(X)[:, 1]

        # 2. Enforce monotonicity
        for i in range(1, k - 1):
            cum_proba[:, i] = np.minimum(cum_proba[:, i], cum_proba[:, i - 1])

        # 3. Derive class probabilities
        class_proba = np.zeros((n, k))
        class_proba[:, 0] = 1.0 - cum_proba[:, 0]
        for j in range(1, k - 1):
            class_proba[:, j] = cum_proba[:, j - 1] - cum_proba[:, j]
        class_proba[:, k - 1] = cum_proba[:, k - 2]
        
        # Clip & Normalize
        class_proba = np.clip(class_proba, 0.0, 1.0)
        row_sums = class_proba.sum(axis=1, keepdims=True)
        class_proba /= (row_sums + 1e-10) # Avoid division by zero

        return class_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # REMOVED get_params and set_params 
    # BaseEstimator handles this automatically for 'base_estimator'.


# =============================================================================
# ACTIVITY CLIFF PAIRS
# =============================================================================
# Loaded as unordered (drug_a, drug_b).
# Ordinal direction is resolved at runtime from dataset labels,
# so this is independent of any binary labeling scheme.

ACTIVITY_CLIFF_PAIRS_RAW = [
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

ACTIVITY_CLIFF_COMPOUNDS = list(set(
    drug for pair in ACTIVITY_CLIFF_PAIRS_RAW for drug in pair
))


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return columns starting with 'feature_'."""
    cols = [c for c in df.columns if c.startswith("feature_")]
    if not cols:
        raise ValueError("No feature columns found (expected prefix 'feature_').")
    return cols


def compute_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def compute_scaffolds(smiles_list: List[str]) -> List[str]:
    return [compute_scaffold(s) for s in smiles_list]


def encode_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map string DILI labels -> integer {0, 1, 2}."""
    encoded = np.array([
        ORDINAL_LABEL_MAP.get(str(l).lower().strip(), -1)
        for l in raw_labels
    ])
    bad = (encoded == -1)
    if bad.any():
        unknown = set(str(l).lower().strip() for l in raw_labels[bad])
        raise ValueError(
            f"{bad.sum()} unknown labels: {unknown}\n"
            f"Expected one of: {list(ORDINAL_LABEL_MAP.keys())}"
        )
    return encoded


def load_and_encode(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract (X, y, smiles) from a DataFrame.

    Returns:
        X:      (n, d) float32 feature matrix
        y:      (n,)  int labels {0, 1, 2}
        smiles: list of SMILES strings
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = encode_labels(df[label_col].values)
    smiles = df[smiles_col].astype(str).tolist()

    counts = {ORDINAL_LABEL_NAMES[i]: int((y == i).sum()) for i in range(3)}
    print(
        f"  Loaded {len(df)} molecules | Features: {len(feature_cols)}\n"
        f"  vNo={counts['vno-dili-concern']}  "
        f"vLess={counts['vless-dili-concern']}  "
        f"vMost={counts['vmost-dili-concern']}"
    )
    return X, y, smiles


# =============================================================================
# MODEL TRAINING
# =============================================================================

def make_param_grid(penalties: List[str], c_values: List[float]) -> List[Dict]:
    """
    Build GridSearchCV parameter grid for OrdinalClassifier.

    Parameters are prefixed with base_estimator__ to reach through the
    OrdinalClassifier wrapper to the underlying LogisticRegression.

    For L1 penalty: uses liblinear (supports L1 for binary classifiers).
    For L2 penalty: uses lbfgs (fast, supports L2).
    """
    grid = []
    for penalty in penalties:
        if penalty == "l1":
            grid.append({
                "base_estimator__penalty": ["l1"],
                "base_estimator__C": c_values,
                "base_estimator__solver": ["liblinear"],
            })
        elif penalty == "l2":
            grid.append({
                "base_estimator__penalty": ["l2"],
                "base_estimator__C": c_values,
                "base_estimator__solver": ["lbfgs", "liblinear"],
            })
        else:
            raise ValueError(f"Unsupported penalty '{penalty}'. Use 'l1' or 'l2'.")
    return grid


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: List[str],
    c_values: List[float],
    penalties: List[str],
    inner_splits: int,
    seed: int,
) -> Tuple[OrdinalClassifier, Dict, float]:
    """
    Fit OrdinalClassifier with GridSearchCV.

    Inner CV criterion: weighted OvR ROC-AUC
    (works because OrdinalClassifier.predict_proba returns proper (n, 3) matrix)

    Returns:
        model:       Fitted OrdinalClassifier (best hyperparams)
        best_params: {'penalty', 'C', 'solver'} — cleaned of prefix
        best_score:  Best inner CV ROC-AUC
    """
    scaffolds = compute_scaffolds(smiles_train)
    param_grid = make_param_grid(penalties, c_values)

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )

    base_lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=1,
    )
    ordinal_clf = OrdinalClassifier(base_estimator=base_lr)

    grid = GridSearchCV(
        ordinal_clf,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc_ovr_weighted",   # weighted macro OvR AUC
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=scaffolds)

    # Clean best_params: strip 'base_estimator__' prefix for display
    raw_params = grid.best_params_
    clean_params = {}
    for k, v in raw_params.items():
        clean_key = k.replace("base_estimator__", "")
        clean_params[clean_key] = v

    return grid.best_estimator_, clean_params, grid.best_score_


def expected_severity(proba: np.ndarray) -> np.ndarray:
    """
    Continuous ranking score: E[Y] = proba @ [0, 1, 2].

    Represents the model's expected severity for each compound.
    Used as the ranking signal for activity cliff evaluation.
    Shape: (n,)
    """
    return proba @ SEVERITY_WEIGHTS


# =============================================================================
# METRICS
# =============================================================================

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,          # (n, 3)
) -> Dict:
    """
    Comprehensive 3-class classification metrics.

    Args:
        y_true: True labels {0, 1, 2}
        y_pred: Predicted labels {0, 1, 2}
        proba:  Predicted probability matrix (n, 3)
    """
    scores = expected_severity(proba)   # continuous ranking score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            roc_auc_macro = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro"
            )
            roc_auc_weighted = roc_auc_score(
                y_true, proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            roc_auc_macro = roc_auc_weighted = float("nan")

        try:
            kappa_linear = cohen_kappa_score(y_true, y_pred, weights="linear")
            kappa_quad   = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        except Exception:
            kappa_linear = kappa_quad = float("nan")

        spearman_r, _ = spearmanr(y_true, scores)

        mae           = float(np.mean(np.abs(y_true - y_pred)))
        adj_acc       = float(np.mean(np.abs(y_true - y_pred) <= 1))

    return {
        "accuracy":             accuracy_score(y_true, y_pred),
        "balanced_accuracy":    balanced_accuracy_score(y_true, y_pred),
        "f1_macro":             f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":          f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mae":                  mae,
        "adjacent_accuracy":    adj_acc,
        "kappa_linear":         float(kappa_linear),
        "kappa_quadratic":      float(kappa_quad),
        "spearman_r":           float(spearman_r),
        "roc_auc_macro_ovr":    roc_auc_macro,
        "roc_auc_weighted_ovr": roc_auc_weighted,
    }


def compute_multiclass_metrics_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> Dict:
    try:
        return compute_multiclass_metrics(y_true, y_pred, proba)
    except Exception as e:
        print(f"  [warn] Metric computation failed: {e}")
        return {k: float("nan") for k in [
            "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
            "mae", "adjacent_accuracy", "kappa_linear", "kappa_quadratic",
            "spearman_r", "roc_auc_macro_ovr", "roc_auc_weighted_ovr",
        ]}


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = [0, 1, 2]
    names  = ["vNo", "vLess", "vMost"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f"{'':>8}" + "".join(f"{n:>8}" for n in names) + "  (predicted)"
    print(f"  Confusion matrix:")
    print(f"  {header}")
    for i, name in enumerate(names):
        row = "".join(f"{cm[i, j]:>8}" for j in range(3))
        print(f"  {name:>8}{row}")


# =============================================================================
# ACTIVITY CLIFF SPLIT
# =============================================================================

def split_activity_cliff_data(
    df: pd.DataFrame,
    compound_name_col: str,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str, int, int]]]:
    """
    Split dataset into main (CV training) and activity cliff (held-out test).

    Cliff pairs are resolved directionally using each compound's actual ordinal
    label in the dataset. Pairs where both drugs share the same label are dropped.

    Returns:
        df_main:     Non-cliff molecules for CV training
        df_cliff:    Activity cliff molecules (held-out)
        valid_pairs: List of (lower_drug, higher_drug, lower_label, higher_label)
                     ordered by ordinal severity
    """
    if compound_name_col not in df.columns:
        raise ValueError(f"Column '{compound_name_col}' not found.")

    dataset_compounds = set(df[compound_name_col].unique())
    found   = dataset_compounds & set(ACTIVITY_CLIFF_COMPOUNDS)
    missing = set(ACTIVITY_CLIFF_COMPOUNDS) - dataset_compounds

    # Build compound -> ordinal label lookup from dataset
    cliff_df_sub = df[df[compound_name_col].isin(found)]
    name_to_label: Dict[str, int] = {}
    for _, row in cliff_df_sub.iterrows():
        name = row[compound_name_col]
        raw  = str(row[label_col]).lower().strip()
        if raw in ORDINAL_LABEL_MAP:
            name_to_label[name] = ORDINAL_LABEL_MAP[raw]

    # Resolve pairs — keep only those with different ordinal labels
    valid_pairs = []
    skipped_same = 0
    for drug_a, drug_b in ACTIVITY_CLIFF_PAIRS_RAW:
        if drug_a not in dataset_compounds or drug_b not in dataset_compounds:
            continue
        label_a = name_to_label.get(drug_a)
        label_b = name_to_label.get(drug_b)
        if label_a is None or label_b is None:
            continue
        if label_a == label_b:
            skipped_same += 1
            continue
        # Order: lower ordinal first
        if label_a < label_b:
            valid_pairs.append((drug_a, drug_b, label_a, label_b))
        else:
            valid_pairs.append((drug_b, drug_a, label_b, label_a))

    # Stratify by ordinal distance
    dist1 = [(a, b, la, lb) for a, b, la, lb in valid_pairs if lb - la == 1]
    dist2 = [(a, b, la, lb) for a, b, la, lb in valid_pairs if lb - la == 2]

    print(f"\n{'='*60}")
    print("ACTIVITY CLIFF DATA SPLIT")
    print(f"{'='*60}")
    print(f"  Cliff compounds found:    {len(found)}/{len(ACTIVITY_CLIFF_COMPOUNDS)}")
    print(f"  Valid pairs (diff label): {len(valid_pairs)}/{len(ACTIVITY_CLIFF_PAIRS_RAW)}")
    print(f"  Skipped (same label):     {skipped_same}")
    if missing:
        print(f"  Missing: {sorted(missing)}")
    print(f"\n  By ordinal distance:")
    print(f"    Distance 1 (adjacent):  {len(dist1)}  [vNo<->vLess or vLess<->vMost]")
    print(f"    Distance 2 (extreme):   {len(dist2)}  [vNo<->vMost]")

    cliff_mask = df[compound_name_col].isin(found)
    df_cliff   = df[cliff_mask].copy()
    df_main    = df[~cliff_mask].copy()

    print(f"\n  Training set: {len(df_main)} molecules")
    print(f"  Test set:     {len(df_cliff)} molecules (activity cliffs)")
    print(f"{'='*60}")

    return df_main, df_cliff, valid_pairs


# =============================================================================
# EVALUATION MODE 1: 5-Fold Nested CV
# =============================================================================

def scaffold_kfold_indices(
    smiles: List[str],
    labels: np.ndarray,
    n_splits: int,
    seed: int,
):
    """StratifiedGroupKFold using Murcko scaffold groups."""
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
) -> None:
    """
    5-fold nested cross-validation on non-cliff molecules.

    Outer fold: performance estimation
    Inner fold: hyperparameter selection via weighted OvR ROC-AUC
    Model: OrdinalClassifier (Frank & Hall decomposition)
    """
    print("\n" + "=" * 70)
    print("EVALUATION MODE: 5-Fold Nested CV (3-Class Ordinal)")
    print("=" * 70)

    # Hold out cliff compounds before CV
    df_main, _, _ = split_activity_cliff_data(df, compound_name_col, label_col)

    X, y, smiles = load_and_encode(df_main, smiles_col, label_col)

    n_splits    = 5
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        scaffold_kfold_indices(smiles, y, n_splits, seed), start=1
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        smiles_train   = [smiles[i] for i in train_idx]

        # Per-class counts
        def counts(arr):
            return {ORDINAL_LABEL_NAMES[i]: int((arr == i).sum()) for i in range(3)}

        tc, vc = counts(y_train), counts(y_val)
        print(f"\n[Fold {fold}]")
        print(f"  Train {len(train_idx):4d}:  "
              f"vNo={tc['vno-dili-concern']}  "
              f"vLess={tc['vless-dili-concern']}  "
              f"vMost={tc['vmost-dili-concern']}")
        print(f"  Val   {len(val_idx):4d}:  "
              f"vNo={vc['vno-dili-concern']}  "
              f"vLess={vc['vless-dili-concern']}  "
              f"vMost={vc['vmost-dili-concern']}")

        # Train ordinal model
        model, best_params, best_score = train_model(
            X_train, y_train, smiles_train, c_values, penalties, inner_splits, seed
        )

        # Predict
        proba  = model.predict_proba(X_val)    # (n_val, 3) — proper ordinal probs
        y_pred = model.predict(X_val)
        metrics = compute_multiclass_metrics_safe(y_val, y_pred, proba)

        metrics.update({
            "fold":             fold,
            "n_train":          len(train_idx),
            "n_val":            len(val_idx),
            "best_penalty":     best_params["penalty"],
            "best_C":           best_params["C"],
            "best_solver":      best_params["solver"],
            "inner_auc_score":  best_score,
        })
        all_metrics.append(metrics)

        print_confusion_matrix(y_val, y_pred)
        print(f"\n  acc={metrics['accuracy']:.4f} | "
              f"bal_acc={metrics['balanced_accuracy']:.4f} | "
              f"MAE={metrics['mae']:.4f} | "
              f"adj_acc={metrics['adjacent_accuracy']:.4f}")
        print(f"  f1_macro={metrics['f1_macro']:.4f} | "
              f"kappa_lin={metrics['kappa_linear']:.4f} | "
              f"Spearman={metrics['spearman_r']:.4f} | "
              f"ROC-AUC(macro)={metrics['roc_auc_macro_ovr']:.4f}")
        print(f"  Best: {best_params['penalty']}, C={best_params['C']} | "
              f"Inner AUC={best_score:.4f}")

    # Save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "cv_metrics.csv"), index=False)

    print("\n" + "-" * 70)
    print("MEAN CV METRICS (mean +/- std over 5 folds):")
    print("-" * 70)
    summary_cols = [
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "mae", "adjacent_accuracy", "kappa_linear", "kappa_quadratic",
        "spearman_r", "roc_auc_macro_ovr", "roc_auc_weighted_ovr",
    ]
    for col in summary_cols:
        if col in metrics_df.columns:
            print(f"  {col:<28}: {metrics_df[col].mean():.4f} +/- {metrics_df[col].std():.4f}")

    print(f"\nResults saved -> {os.path.join(output_dir, 'cv_metrics.csv')}")


# =============================================================================
# EVALUATION MODE 2: Activity Cliff Evaluation
# =============================================================================

def evaluate_activity_cliff_pairs(
    model,  # OrdinalClassifier
    df_cliff: pd.DataFrame,
    valid_pairs: List[Tuple[str, str, int, int]],
    label_col: str,
    compound_name_col: str,
) -> Dict:
    """
    Evaluate ordinal model on activity cliff pairs.

    Ranking score: E[Y] = proba @ [0, 1, 2]  (expected severity)

    A pair is correctly ranked when:
        E[Y](higher_label_drug) > E[Y](lower_label_drug)

    Pairs are stratified by ordinal distance:
        distance 1: adjacent classes  (vNo<->vLess or vLess<->vMost)
        distance 2: extreme classes   (vNo<->vMost)
    """
    feature_cols = get_feature_columns(df_cliff)
    X = df_cliff[feature_cols].values.astype(np.float32)
    y = encode_labels(df_cliff[label_col].values)

    proba  = model.predict_proba(X)          # (n, 3) — ordinal probabilities
    y_pred = model.predict(X)
    scores = expected_severity(proba)         # (n,) continuous ranking score

    # Compound name -> row index in df_cliff
    compound_to_idx = {
        name: idx for idx, name in enumerate(df_cliff[compound_name_col].values)
    }

    pair_results = []
    for lower_drug, higher_drug, lower_label, higher_label in valid_pairs:
        if lower_drug not in compound_to_idx or higher_drug not in compound_to_idx:
            continue

        i_lower  = compound_to_idx[lower_drug]
        i_higher = compound_to_idx[higher_drug]

        score_lower  = float(scores[i_lower])
        score_higher = float(scores[i_higher])

        lower_correct  = int(y_pred[i_lower]  == y[i_lower])
        higher_correct = int(y_pred[i_higher] == y[i_higher])

        pair_results.append({
            "lower_drug":         lower_drug,
            "higher_drug":        higher_drug,
            "lower_label":        lower_label,
            "higher_label":       higher_label,
            "lower_label_name":   ORDINAL_LABEL_NAMES[lower_label],
            "higher_label_name":  ORDINAL_LABEL_NAMES[higher_label],
            "ordinal_distance":   higher_label - lower_label,
            "score_lower":        score_lower,
            "score_higher":       score_higher,
            "delta":              score_higher - score_lower,
            "pairwise_correct":   int(score_higher > score_lower),
            "lower_pred":         int(y_pred[i_lower]),
            "higher_pred":        int(y_pred[i_higher]),
            "lower_correct":      lower_correct,
            "higher_correct":     higher_correct,
            "both_correct":       int(lower_correct and higher_correct),
            # Raw probabilities for each class
            "lower_p_vno":        float(proba[i_lower,  0]),
            "lower_p_vless":      float(proba[i_lower,  1]),
            "lower_p_vmost":      float(proba[i_lower,  2]),
            "higher_p_vno":       float(proba[i_higher, 0]),
            "higher_p_vless":     float(proba[i_higher, 1]),
            "higher_p_vmost":     float(proba[i_higher, 2]),
            # Cumulative probabilities (ordinal-specific diagnostic)
            "lower_cum_p_gt0":    float(proba[i_lower, 1] + proba[i_lower, 2]),
            "higher_cum_p_gt0":   float(proba[i_higher, 1] + proba[i_higher, 2]),
        })

    n_pairs = len(pair_results)
    if n_pairs == 0:
        raise ValueError("No valid pairs resolved in df_cliff.")

    deltas   = [p["delta"] for p in pair_results]
    per_drug = compute_multiclass_metrics_safe(y, y_pred, proba)

    # Stratified accuracy by ordinal distance
    pairs_d1 = [p for p in pair_results if p["ordinal_distance"] == 1]
    pairs_d2 = [p for p in pair_results if p["ordinal_distance"] == 2]

    def pairwise_acc(pairs):
        return sum(p["pairwise_correct"] for p in pairs) / len(pairs) if pairs else float("nan")

    def both_rate(pairs):
        return sum(p["both_correct"] for p in pairs) / len(pairs) if pairs else float("nan")

    return {
        # Overall pairwise
        "n_pairs":            n_pairs,
        "pairwise_accuracy":  pairwise_acc(pair_results),
        "both_correct_rate":  both_rate(pair_results),
        "mean_delta":         float(np.mean(deltas)),
        "median_delta":       float(np.median(deltas)),
        "std_delta":          float(np.std(deltas)),
        "n_positive_delta":   int(sum(d > 0 for d in deltas)),
        # Stratified by distance
        "n_pairs_dist1":      len(pairs_d1),
        "pairwise_acc_dist1": pairwise_acc(pairs_d1),
        "both_correct_dist1": both_rate(pairs_d1),
        "n_pairs_dist2":      len(pairs_d2),
        "pairwise_acc_dist2": pairwise_acc(pairs_d2),
        "both_correct_dist2": both_rate(pairs_d2),
        # Per-drug metrics on cliff compounds
        "n_compounds":        len(y),
        **{f"per_drug_{k}": v for k, v in per_drug.items()},
        # Full per-pair breakdown (for CSV)
        "pair_results":       pair_results,
    }


def print_activity_cliff_results(results: Dict, best_params: Dict) -> None:
    print(f"\n{'='*70}")
    print("ACTIVITY CLIFF EVALUATION RESULTS (3-Class Ordinal)")
    print(f"{'='*70}")
    print(f"\nModel: OrdinalClassifier (Frank & Hall)")
    print(f"  Base LR: {best_params['penalty']}, C={best_params['C']}, "
          f"solver={best_params['solver']}")

    print(f"\n--- Overall Pairwise Metrics (n={results['n_pairs']} pairs) ---")
    print(f"  Pairwise Ranking Accuracy:     {results['pairwise_accuracy']:.1%}")
    print(f"  Both Correct Rate:             {results['both_correct_rate']:.1%}")
    print(f"  Mean delta E[Y] (high - low):  {results['mean_delta']:.4f}")
    print(f"  Positive delta count:          "
          f"{results['n_positive_delta']}/{results['n_pairs']}")

    print(f"\n--- Stratified by Ordinal Distance ---")
    if results["n_pairs_dist1"] > 0:
        print(f"  Distance 1 (adjacent):  n={results['n_pairs_dist1']}  "
              f"rank_acc={results['pairwise_acc_dist1']:.1%}  "
              f"both_correct={results['both_correct_dist1']:.1%}")
    if results["n_pairs_dist2"] > 0:
        print(f"  Distance 2 (extreme):   n={results['n_pairs_dist2']}  "
              f"rank_acc={results['pairwise_acc_dist2']:.1%}  "
              f"both_correct={results['both_correct_dist2']:.1%}")

    print(f"\n--- Per-Drug Metrics on Cliff Compounds (n={results['n_compounds']}) ---")
    for key in ["per_drug_accuracy", "per_drug_balanced_accuracy", "per_drug_mae",
                "per_drug_adjacent_accuracy", "per_drug_kappa_linear",
                "per_drug_spearman_r", "per_drug_roc_auc_macro_ovr"]:
        if key in results:
            label = key.replace("per_drug_", "")
            print(f"  {label:<28}: {results[key]:.4f}")

    print(f"\n--- Per-Pair Breakdown ---")
    header = (f"{'Lower Drug':<32} {'Higher Drug':<32} "
              f"{'Labels':<12} {'Dist':>4} "
              f"{'E[Y]_low':>9} {'E[Y]_high':>10} {'delta':>7} "
              f"{'Rank':>5} {'Both':>5}")
    print(header)
    print("-" * len(header))

    for p in results["pair_results"]:
        label_str = f"{p['lower_label']}->{p['higher_label']}"
        rank_sym  = "Y" if p["pairwise_correct"] else "N"
        both_sym  = "Y" if p["both_correct"]     else "N"
        print(f"{p['lower_drug']:<32} {p['higher_drug']:<32} "
              f"{label_str:<12} {p['ordinal_distance']:>4} "
              f"{p['score_lower']:>9.3f} {p['score_higher']:>10.3f} "
              f"{p['delta']:>7.3f} {rank_sym:>5} {both_sym:>5}")

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
    Activity cliff pipeline:
      1. Hold out cliff compounds
      2. Train ordinal model on all non-cliff molecules
      3. Evaluate pairwise ranking on held-out cliff pairs
    """
    print("\n" + "=" * 70)
    print("EVALUATION MODE: Activity Cliff (3-Class Ordinal)")
    print("=" * 70)

    df_main, df_cliff, valid_pairs = split_activity_cliff_data(
        df, compound_name_col, label_col
    )

    if not valid_pairs:
        raise ValueError("No valid activity cliff pairs found in dataset!")

    X_main, y_main, smiles_main = load_and_encode(df_main, smiles_col, label_col)

    print(f"\nTraining OrdinalClassifier on {len(X_main)} molecules...")
    model, best_params, best_score = train_model(
        X_main, y_main, smiles_main, c_values, penalties, inner_splits, seed
    )
    print(f"  Best: {best_params['penalty']}, C={best_params['C']} | "
          f"Inner AUC={best_score:.4f}")

    results = evaluate_activity_cliff_pairs(
        model, df_cliff, valid_pairs, label_col, compound_name_col
    )

    print_activity_cliff_results(results, best_params)

    # Save summary (exclude pair_results list)
    summary = {k: v for k, v in results.items() if k != "pair_results"}
    summary.update({
        "n_train":          len(X_main),
        "model_type":       "OrdinalClassifier_FrankHall",
        "best_penalty":     best_params["penalty"],
        "best_C":           best_params["C"],
        "best_solver":      best_params["solver"],
        "inner_auc_score":  best_score,
        "seed":             seed,
    })
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "activity_cliff_summary.csv"), index=False
    )
    pd.DataFrame(results["pair_results"]).to_csv(
        os.path.join(output_dir, "activity_cliff_pairs.csv"), index=False
    )
    print(f"Results saved ->")
    print(f"  {os.path.join(output_dir, 'activity_cliff_summary.csv')}")
    print(f"  {os.path.join(output_dir, 'activity_cliff_pairs.csv')}")


# =============================================================================
# MAIN
# =============================================================================

def parse_comma_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_comma_strings(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="DILI 3-Class Ordinal Regression: CV + Activity Cliff Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--features_file",     type=str, required=True,
                        help="Path to features CSV")
    parser.add_argument("--label_col",         type=str, required=True,
                        help="Ordinal label column, e.g. 'vDILI-Concern_standardized'")
    parser.add_argument("--output_dir",        type=str, required=True,
                        help="Directory to save results")

    # Optional
    parser.add_argument("--smiles_col",        type=str, default="SMILES_Normalized")
    parser.add_argument("--compound_name_col", type=str, default="Name")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--c_values",          type=parse_comma_floats,
                        default=[1, 0.1, 0.01, 0.001, 0.0001],
                        help="Comma-separated C values (default: 1,0.1,0.01,0.001,0.0001)")
    parser.add_argument("--penalties",         type=parse_comma_strings,
                        default=["l1", "l2"],
                        help="Comma-separated penalties (default: l1,l2)")
    parser.add_argument("--inner_splits",      type=int, default=5)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("DILI 3-Class Ordinal Regression (Frank & Hall)")
    print(f"{'='*70}")
    print(f"  Features:     {args.features_file}")
    print(f"  Label col:    {args.label_col}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Seed:         {args.seed}")
    print(f"  C values:     {args.c_values}")
    print(f"  Penalties:    {args.penalties}")
    print(f"  Model:        OrdinalClassifier (k-1 binary decomposition)")

    df = pd.read_csv(args.features_file).drop_duplicates(subset=[args.smiles_col])

    '''
    # 5-fold CV on non-cliff molecules
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
    '''
    # Activity cliff evaluation on held-out pairs
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