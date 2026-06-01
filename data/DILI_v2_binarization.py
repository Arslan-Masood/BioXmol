#!/usr/bin/env python3
"""
Binarize DILIRank v2.0 labels into binary classification tasks.
Creates both standard and conservative binary classification labels.
"""

import pandas as pd
import numpy as np
import argparse
import os

def create_binary_labels_standard(label):
    """
    DILI-Positive (1): vMost-DILI-concern + vLess-DILI-concern
    DILI-Negative (0): vNo-DILI-concern
    Ambiguous: NaN (to be excluded)
    """
    if pd.isna(label):
        return np.nan
    
    label_lower = str(label).lower().strip()
    
    if 'vmost-dili-concern' in label_lower or 'vless-dili-concern' in label_lower:
        return 1  # DILI-Positive
    elif 'vno-dili-concern' in label_lower:
        return 0  # DILI-Negative
    elif 'ambiguous' in label_lower:
        return np.nan  # Exclude
    else:
        return np.nan  # Unknown labels

def create_binary_labels_conservative(label):
    """
    DILI-Positive (1): vMost-DILI-concern ONLY
    DILI-Negative (0): vNo-DILI-concern ONLY
    Others (Less, Ambiguous): NaN (excluded)
    """
    if pd.isna(label):
        return np.nan
    
    label_lower = str(label).lower().strip()
    
    if 'vmost-dili-concern' in label_lower:
        return 1  # DILI-Positive
    elif 'vno-dili-concern' in label_lower:
        return 0  # DILI-Negative
    else:
        return np.nan  # Exclude Less and Ambiguous

def binarize_dilirank_labels(input_file: str, output_dir: str, label_column: str = "vDILI-Concern"):
    """
    Create binary labels for DILIRank dataset.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output files
        label_column: Name of the label column (default: "vDILI-Concern")
    """
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds")
    
    # Standardize labels
    df['vDILI-Concern_standardized'] = df[label_column].str.lower().str.strip()
    
    # Check original distribution
    print("\nOriginal label distribution:")
    print(df[label_column].value_counts())
    
    # Create standard binary labels
    df['binary_label'] = df['vDILI-Concern_standardized'].apply(create_binary_labels_standard)
    df_standard = df[df['binary_label'].notna()].copy()
    
    print("\n" + "="*60)
    print("STANDARD BINARY CLASSIFICATION")
    print("="*60)
    print(f"Dataset size after removing ambiguous: {len(df_standard)}")
    print(f"DILI-Positive (1): {(df_standard['binary_label'] == 1).sum()}")
    print(f"DILI-Negative (0): {(df_standard['binary_label'] == 0).sum()}")
    if (df_standard['binary_label'] == 0).sum() > 0:
        print(f"Class balance ratio: {(df_standard['binary_label'] == 1).sum() / (df_standard['binary_label'] == 0).sum():.2f}")
    
    # Create conservative binary labels
    df['binary_label_conservative'] = df['vDILI-Concern_standardized'].apply(create_binary_labels_conservative)
    df_conservative = df[df['binary_label_conservative'].notna()].copy()
    
    print("\n" + "="*60)
    print("CONSERVATIVE BINARY CLASSIFICATION")
    print("="*60)
    print(f"Dataset size after filtering: {len(df_conservative)}")
    print(f"DILI-Positive (1): {(df_conservative['binary_label_conservative'] == 1).sum()}")
    print(f"DILI-Negative (0): {(df_conservative['binary_label_conservative'] == 0).sum()}")
    if (df_conservative['binary_label_conservative'] == 0).sum() > 0:
        print(f"Class balance ratio: {(df_conservative['binary_label_conservative'] == 1).sum() / (df_conservative['binary_label_conservative'] == 0).sum():.2f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save standard approach (recommended)
    output_standard = os.path.join(output_dir, 'DILIrank_2.0_binary_standard.csv')
    df_standard.to_csv(output_standard, index=False)
    print(f"\n✓ Saved standard binary labels: {output_standard}")
    
    # Save conservative approach
    output_conservative = os.path.join(output_dir, 'DILIrank_2.0_binary_conservative.csv')
    df_conservative.to_csv(output_conservative, index=False)
    print(f"✓ Saved conservative binary labels: {output_conservative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize DILIRank v2.0 labels into binary classification tasks")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory for processed CSV files")
    parser.add_argument("--label_column", "-l", type=str, default="vDILI-Concern",
                       help="Name of the label column (default: vDILI-Concern)")
    
    args = parser.parse_args()
    
    try:
        binarize_dilirank_labels(args.input_file, args.output_dir, args.label_column)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
