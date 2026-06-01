#!/usr/bin/env python3
"""
Combine DILI Linear Probing Results Across All Models
======================================================

Aggregates results from multiple feature sets into consolidated CSV files:
- Combined 5-fold CV metrics (with mean/std per model)
- Combined activity cliff summaries
- Combined activity cliff pair-level results

Usage:
    python combine_dili_results.py --output_root /path/to/results --save_dir /path/to/combined
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np


def find_result_files(output_root: str, eval_mode: str, filename: str) -> List[Dict]:
    """
    Find all result files matching the expected directory structure.
    
    Expected structure:
        output_root/<eval_mode>/<feature_name>/seed_<N>/<filename>
    
    Returns list of dicts with file path and metadata.
    """
    results = []
    mode_dir = Path(output_root) / eval_mode
    
    if not mode_dir.exists():
        print(f"Warning: Directory not found: {mode_dir}")
        return results
    
    for feature_dir in mode_dir.iterdir():
        if not feature_dir.is_dir():
            continue
        
        feature_name = feature_dir.name
        
        for seed_dir in feature_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            
            seed = int(seed_dir.name.replace("seed_", ""))
            filepath = seed_dir / filename
            
            if filepath.exists():
                results.append({
                    "filepath": str(filepath),
                    "feature": feature_name,
                    "seed": seed,
                })
    
    return results


def combine_cv_results(output_root: str) -> pd.DataFrame:
    """
    Combine 5-fold CV results from all models.
    
    Returns DataFrame with columns:
        feature, seed, fold, roc_auc, pr_auc, balanced_accuracy, ...
    """
    files = find_result_files(output_root, "cv", "cv_metrics.csv")
    
    if not files:
        print("No CV result files found!")
        return pd.DataFrame()
    
    dfs = []
    for info in files:
        try:
            df = pd.read_csv(info["filepath"])
            df["feature"] = info["feature"]
            df["experiment_seed"] = info["seed"]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {info['filepath']}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns
    id_cols = ["feature", "experiment_seed", "fold"]
    other_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + other_cols]
    
    return combined


def summarize_cv_results(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std across folds for each model/seed.
    """
    if cv_df.empty:
        return pd.DataFrame()
    
    metric_cols = [
        "roc_auc", "pr_auc", "accuracy", "f1", "balanced_accuracy",
        "matthews_corrcoef", "sensitivity", "specificity", "precision",
        "pos_LR", "sensitivity_at_spec100"
    ]
    
    # Filter to existing columns
    metric_cols = [c for c in metric_cols if c in cv_df.columns]
    
    # Group by feature and seed
    grouped = cv_df.groupby(["feature", "experiment_seed"])
    
    # Compute mean and std
    mean_df = grouped[metric_cols].mean().reset_index()
    std_df = grouped[metric_cols].std().reset_index()
    
    # Rename columns
    mean_df.columns = ["feature", "experiment_seed"] + [f"{c}_mean" for c in metric_cols]
    std_df.columns = ["feature", "experiment_seed"] + [f"{c}_std" for c in metric_cols]
    
    # Merge
    summary = mean_df.merge(std_df, on=["feature", "experiment_seed"])
    
    # Interleave mean/std columns for readability
    ordered_cols = ["feature", "experiment_seed"]
    for col in metric_cols:
        ordered_cols.extend([f"{col}_mean", f"{col}_std"])
    
    summary = summary[ordered_cols]
    
    # Sort by ROC-AUC descending
    if "roc_auc_mean" in summary.columns:
        summary = summary.sort_values("roc_auc_mean", ascending=False)
    
    return summary


def combine_activity_cliff_summaries(output_root: str) -> pd.DataFrame:
    """
    Combine activity cliff summary results from all models.
    """
    files = find_result_files(output_root, "activity_cliff", "activity_cliff_summary.csv")
    
    if not files:
        print("No activity cliff summary files found!")
        return pd.DataFrame()
    
    dfs = []
    for info in files:
        try:
            df = pd.read_csv(info["filepath"])
            df["feature"] = info["feature"]
            df["experiment_seed"] = info["seed"]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {info['filepath']}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns
    id_cols = ["feature", "experiment_seed"]
    other_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + other_cols]
    
    # Sort by pairwise accuracy descending
    if "pairwise_accuracy" in combined.columns:
        combined = combined.sort_values("pairwise_accuracy", ascending=False)
    
    return combined


def combine_activity_cliff_pairs(output_root: str) -> pd.DataFrame:
    """
    Combine activity cliff pair-level results from all models.
    """
    files = find_result_files(output_root, "activity_cliff", "activity_cliff_pairs.csv")
    
    if not files:
        print("No activity cliff pair files found!")
        return pd.DataFrame()
    
    dfs = []
    for info in files:
        try:
            df = pd.read_csv(info["filepath"])
            df["feature"] = info["feature"]
            df["experiment_seed"] = info["seed"]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {info['filepath']}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns
    id_cols = ["feature", "experiment_seed", "safe_drug", "toxic_drug"]
    other_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + other_cols]
    
    return combined


def create_pair_pivot_table(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table: rows = drug pairs, columns = features, values = pairwise_correct.
    
    Useful for seeing which pairs are easy/hard across models.
    """
    if pairs_df.empty:
        return pd.DataFrame()
    
    # Create pair identifier
    pairs_df = pairs_df.copy()
    pairs_df["pair"] = pairs_df["safe_drug"] + " vs " + pairs_df["toxic_drug"]
    
    # Pivot
    pivot = pairs_df.pivot_table(
        index="pair",
        columns="feature",
        values="pairwise_correct",
        aggfunc="mean"  # Average across seeds if multiple
    )
    
    # Add summary columns
    pivot["n_correct"] = pivot.sum(axis=1)
    pivot["pct_correct"] = pivot.mean(axis=1)
    
    # Sort by difficulty (hardest pairs first)
    pivot = pivot.sort_values("pct_correct", ascending=True)
    
    return pivot


def main():
    parser = argparse.ArgumentParser(
        description="Combine DILI linear probing results across all models"
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Root directory containing cv/ and activity_cliff/ subdirectories"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save combined results (default: output_root/combined)"
    )
    
    args = parser.parse_args()
    
    save_dir = args.save_dir or os.path.join(args.output_root, "combined")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMBINING DILI LINEAR PROBING RESULTS")
    print("=" * 70)
    print(f"Input:  {args.output_root}")
    print(f"Output: {save_dir}")
    print()
    
    # 1. Combine CV results
    print("1. Processing 5-fold CV results...")
    cv_df = combine_cv_results(args.output_root)
    if not cv_df.empty:
        cv_path = os.path.join(save_dir, "combined_cv_all_folds.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"   Saved: {cv_path} ({len(cv_df)} rows)")
        
        # Summary
        cv_summary = summarize_cv_results(cv_df)
        cv_summary_path = os.path.join(save_dir, "combined_cv_summary.csv")
        cv_summary.to_csv(cv_summary_path, index=False)
        print(f"   Saved: {cv_summary_path} ({len(cv_summary)} models)")
        
        # Print top models
        print("\n   Top 5 models by ROC-AUC:")
        if "roc_auc_mean" in cv_summary.columns:
            top5 = cv_summary.head(5)[["feature", "roc_auc_mean", "roc_auc_std", 
                                        "balanced_accuracy_mean", "matthews_corrcoef_mean"]]
            print(top5.to_string(index=False))
    else:
        print("   No CV results found.")
    
    print()
    
    # 2. Combine activity cliff summaries
    print("2. Processing activity cliff summaries...")
    cliff_summary_df = combine_activity_cliff_summaries(args.output_root)
    if not cliff_summary_df.empty:
        cliff_summary_path = os.path.join(save_dir, "combined_activity_cliff_summary.csv")
        cliff_summary_df.to_csv(cliff_summary_path, index=False)
        print(f"   Saved: {cliff_summary_path} ({len(cliff_summary_df)} models)")
        
        # Print top models
        print("\n   Top 5 models by pairwise accuracy:")
        if "pairwise_accuracy" in cliff_summary_df.columns:
            top5 = cliff_summary_df.head(5)[["feature", "pairwise_accuracy", 
                                              "both_correct_rate", "mean_delta"]]
            print(top5.to_string(index=False))
    else:
        print("   No activity cliff summaries found.")
    
    print()
    
    # 3. Combine activity cliff pairs
    print("3. Processing activity cliff pairs...")
    pairs_df = combine_activity_cliff_pairs(args.output_root)
    if not pairs_df.empty:
        pairs_path = os.path.join(save_dir, "combined_activity_cliff_pairs.csv")
        pairs_df.to_csv(pairs_path, index=False)
        print(f"   Saved: {pairs_path} ({len(pairs_df)} rows)")
        
        # Create pivot table
        pivot_df = create_pair_pivot_table(pairs_df)
        if not pivot_df.empty:
            pivot_path = os.path.join(save_dir, "activity_cliff_pairs_pivot.csv")
            pivot_df.to_csv(pivot_path)
            print(f"   Saved: {pivot_path}")
            
            # Print hardest pairs
            print("\n   Hardest pairs (lowest accuracy across models):")
            print(pivot_df[["pct_correct", "n_correct"]].head(5).to_string())
    else:
        print("   No activity cliff pair results found.")
    
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
