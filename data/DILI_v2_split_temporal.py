#!/usr/bin/env python3
"""
Split DILIRank v2.0 data into train and test sets based on temporal criteria.
Train: ChEMBL_First_Approval < 2010
Test: ChEMBL_First_Approval >= 2010
"""

import pandas as pd
import argparse
import os

def split_dilirank_temporal(input_file: str, output_dir: str, 
                            smiles_col: str = "Normalized_SMILES_combined", 
                            label_col: str = "binary_label", 
                            year_col: str = "ChEMBL_First_Approval", 
                            train_cutoff_year: int = 2010):
    """
    Split DILIRank dataset into train/test based on approval year.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save train/test splits
        smiles_col: Name of the SMILES column
        label_col: Name of the label column
        year_col: Name of the year column
        train_cutoff_year: Year threshold for train/test split (default: 2010)
    """
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds")
    
    # Validate that required columns exist
    missing_cols = [col for col in [smiles_col, label_col, year_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing SMILES or labels
    df = df[df[smiles_col].notna()].copy()
    df = df[df[label_col].notna()].copy()
    df = df[df[year_col].notna()].copy()

    print(f"After filtering for valid SMILES, labels and year: {len(df)} compounds")

    # Split based on approval year
    train_df = df[df[year_col] < train_cutoff_year].copy()
    test_df = df[df[year_col] >= train_cutoff_year].copy()
    
    print(f"\nTemporal Split (cutoff year: {train_cutoff_year}):")
    print(f"  Train (< {train_cutoff_year}): {len(train_df)} compounds")
    print(f"  Test (>= {train_cutoff_year}): {len(test_df)} compounds")
    
    # Statistics
    if len(train_df) > 0:
        print(f"\nTrain set:")
        print(f"  DILI-Positive (1): {(train_df[label_col] == 1).sum()}")
        print(f"  DILI-Negative (0): {(train_df[label_col] == 0).sum()}")
        if (train_df[label_col] == 0).sum() > 0:
            print(f"  Class balance ratio: {(train_df[label_col] == 1).sum() / (train_df[label_col] == 0).sum():.2f}")
        print(f"  Approval year range: {train_df[year_col].min():.0f} - {train_df[year_col].max():.0f}")
    
    if len(test_df) > 0:
        print(f"\nTest set:")
        print(f"  DILI-Positive (1): {(test_df[label_col] == 1).sum()}")
        print(f"  DILI-Negative (0): {(test_df[label_col] == 0).sum()}")
        if (test_df[label_col] == 0).sum() > 0:
            print(f"  Class balance ratio: {(test_df[label_col] == 1).sum() / (test_df[label_col] == 0).sum():.2f}")
        print(f"  Approval year range: {test_df[year_col].min():.0f} - {test_df[year_col].max():.0f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train and test splits (full data files)
    train_file = os.path.join(output_dir, 'DILIrank_2.0_train.csv')
    test_file = os.path.join(output_dir, 'DILIrank_2.0_test.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✓ Saved train split: {train_file}")
    print(f"✓ Saved test split: {test_file}")
    
    # Also create split CSV files with just "SMILES" column (for dataloader split files)
    # The dataloader expects split files to have "SMILES" column
    if smiles_col in train_df.columns:
        train_split_file = os.path.join(output_dir, 'DILIrank_2.0_train_split.csv')
        test_split_file = os.path.join(output_dir, 'DILIrank_2.0_test_split.csv')
        
        train_split_df = pd.DataFrame({'SMILES': train_df[smiles_col]})
        test_split_df = pd.DataFrame({'SMILES': test_df[smiles_col]})
        
        train_split_df.to_csv(train_split_file, index=False)
        test_split_df.to_csv(test_split_file, index=False)
        
        print(f"\n✓ Saved train split file (SMILES only): {train_split_file}")
        print(f"✓ Saved test split file (SMILES only): {test_split_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split DILIRank v2.0 data into train/test based on approval year")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="Path to input CSV file (binary labeled data)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory for train/test splits")
    parser.add_argument("--cutoff_year", "-y", type=int, default=2010,
                       help="Year threshold for train/test split (default: 2010)")
    parser.add_argument("--smiles_col", "-s", type=str, default="Normalized_SMILES_combined",
                       help="Name of the SMILES column (default: Normalized_SMILES_combined)")
    parser.add_argument("--label_col", "-l", type=str, default="binary_label",
                       help="Name of the label column (default: binary_label)")
    parser.add_argument("--year_col", type=str, default="ChEMBL_First_Approval",
                       help="Name of the year column (default: ChEMBL_First_Approval)")
    args = parser.parse_args()
    
    try:
        split_dilirank_temporal(args.input_file, args.output_dir,
                                smiles_col=args.smiles_col, 
                                label_col=args.label_col,
                                year_col=args.year_col,
                                train_cutoff_year=args.cutoff_year)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

