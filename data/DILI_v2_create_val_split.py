#!/usr/bin/env python3
"""
Create validation split from DILIRank v2.0 train data.
This creates a train/val split from the temporal train set.
"""

import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def create_val_split(input_file: str, output_dir: str, val_size: float = 0.2, seed: int = 42, smiles_col: str = "ChEMBL_SMILES_Normalized"):
    """
    Create train/val split from the temporal train data.
    
    Args:
        input_file: Path to input train CSV file
        output_dir: Directory to save train/val split CSVs
        val_size: Fraction of data for validation (default: 0.2)
        seed: Random seed for splitting (default: 42)
        smiles_col: Name of the SMILES column (default: ChEMBL_SMILES_Normalized)
    """
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds from train set")
    
    # Check if SMILES column exists
    if smiles_col not in df.columns:
        raise ValueError(f"{smiles_col} column not found in input file")
    
    # Check if binary_label column exists
    if 'binary_label' not in df.columns:
        raise ValueError("binary_label column not found in input file")
    
    # Remove rows with missing SMILES or labels
    df = df[df[smiles_col].notna()].copy()
    df = df[df['binary_label'].notna()].copy()
    print(f"After filtering for valid SMILES and labels: {len(df)} compounds")
    
    # Create train/val split (stratified by binary_label)
    train_df, val_df = train_test_split(
        df, 
        test_size=val_size, 
        random_state=seed,
        stratify=df['binary_label']
    )
    
    print(f"\nTrain/Val Split (seed={seed}):")
    print(f"  Train: {len(train_df)} compounds ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} compounds ({len(val_df)/len(df)*100:.1f}%)")
    
    print(f"\nTrain set:")
    print(f"  DILI-Positive (1): {(train_df['binary_label'] == 1).sum()}")
    print(f"  DILI-Negative (0): {(train_df['binary_label'] == 0).sum()}")
    
    print(f"\nVal set:")
    print(f"  DILI-Positive (1): {(val_df['binary_label'] == 1).sum()}")
    print(f"  DILI-Negative (0): {(val_df['binary_label'] == 0).sum()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits - use "SMILES" column name (as expected by the dataloader)
    train_output = os.path.join(output_dir, f'DILIrank_2.0_train_seed{seed}_train.csv')
    val_output = os.path.join(output_dir, f'DILIrank_2.0_train_seed{seed}_val.csv')
    
    # Create split CSVs with "SMILES" column (as expected by the dataloader)
    # The dataloader looks for "SMILES" column in split files
    train_split_df = pd.DataFrame({'SMILES': train_df[smiles_col]})
    val_split_df = pd.DataFrame({'SMILES': val_df[smiles_col]})
    
    train_split_df.to_csv(train_output, index=False)
    val_split_df.to_csv(val_output, index=False)
    
    print(f"\n✓ Saved train split: {train_output}")
    print(f"✓ Saved val split: {val_output}")
    
    return train_output, val_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val split from DILIRank v2.0 train data")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="Path to input train CSV file")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory for train/val split CSVs")
    parser.add_argument("--val_size", "-v", type=float, default=0.2,
                       help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for splitting (default: 42)")
    parser.add_argument("--smiles_col", "-c", type=str, default="ChEMBL_SMILES_Normalized",
                       help="Name of the SMILES column (default: ChEMBL_SMILES_Normalized)")
    
    args = parser.parse_args()
    
    try:
        create_val_split(args.input_file, args.output_dir, args.val_size, args.seed, args.smiles_col)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

