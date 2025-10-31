#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import logging
import argparse
from typing import List, Tuple, Dict
import random
from rdkit import Chem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set seed for reproducibility.
    
    Args:
        seed: The seed to use for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)

def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is valid using RDKit.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if SMILES is valid, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def create_lincs_splits(genomic_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/val splits for LINCS data with 95/5 ratio.
    
    Args:
        genomic_df: DataFrame containing LINCS data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Get unique SMILES from LINCS and filter invalid ones
    unique_smiles = genomic_df['Metadata_SMILES'].unique()
    valid_smiles = [s for s in unique_smiles if validate_smiles(s)]
    total_smiles = len(valid_smiles)
    
    logger.info(f"Filtered out {len(unique_smiles) - total_smiles} invalid SMILES strings")
    
    # Calculate split sizes for LINCS (96.5/3.5 split)
    train_size = int(0.965 * total_smiles)
    
    # Randomly shuffle and split LINCS data
    set_seed(seed)
    shuffled_smiles = np.random.permutation(valid_smiles)
    
    train_smiles = shuffled_smiles[:train_size]
    val_smiles = shuffled_smiles[train_size:]
    
    # Create DataFrames
    train_df = pd.DataFrame({'SMILES': train_smiles})
    val_df = pd.DataFrame({'SMILES': val_smiles})
    
    # Log split statistics
    logger.info("LINCS Split Statistics:")
    logger.info(f"- Total unique SMILES: {total_smiles}")
    logger.info(f"- Training SMILES: {len(train_smiles)} ({len(train_smiles)/total_smiles:.2%})")
    logger.info(f"- Validation SMILES: {len(val_smiles)} ({len(val_smiles)/total_smiles:.2%})")
    
    return train_df, val_df

def concatenate_with_jump(lincs_splits: Tuple[pd.DataFrame, pd.DataFrame], 
                         jump_data_path: str, 
                         seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate LINCS splits with JUMP data ensuring no overlap between train and val sets.
    
    Args:
        lincs_splits: Tuple of (train_df, val_df) for LINCS data
        jump_data_path: Path to JUMP data directory
        seed: Split seed number
        
    Returns:
        Tuple of (combined_train_df, combined_val_df)
    """
    lincs_train_df, lincs_val_df = lincs_splits
    
    # Load JUMP data
    logger.info(f"Loading JUMP data from {jump_data_path}")
    jump_train = pd.read_csv(os.path.join(jump_data_path, f'jump-compound-split-{seed}-train.csv'))
    jump_val = pd.read_csv(os.path.join(jump_data_path, f'jump-compound-split-{seed}-val.csv'))
    
    # Check for compounds that appear in both train and val sets
    train_smiles = set(jump_train['SMILES']).union(set(lincs_train_df['SMILES']))
    val_smiles = set(jump_val['SMILES']).union(set(lincs_val_df['SMILES']))
    overlapping_smiles = train_smiles.intersection(val_smiles)
    
    if overlapping_smiles:
        logger.warning(f"Found {len(overlapping_smiles)} compounds that appear in both train and val sets")
        
        # Remove overlapping compounds from validation sets (prioritizing training set)
        jump_val = jump_val[~jump_val['SMILES'].isin(overlapping_smiles)]
        lincs_val_df = lincs_val_df[~lincs_val_df['SMILES'].isin(overlapping_smiles)]
    
    # Create combined datasets
    combined_train = pd.concat([
        jump_train,
        lincs_train_df
    ]).drop_duplicates(subset=['SMILES'])
    
    combined_val = pd.concat([
        jump_val,
        lincs_val_df
    ]).drop_duplicates(subset=['SMILES'])
    
    # Verify no overlap between final train and val sets
    assert len(set(combined_train['SMILES']).intersection(set(combined_val['SMILES']))) == 0, \
        "Found overlapping compounds between train and val sets after combining"
    
    # Log concatenation statistics
    logger.info("\nConcatenation Statistics:")
    logger.info(f"JUMP Training SMILES: {len(jump_train)}")
    logger.info(f"JUMP Validation SMILES: {len(jump_val)}")
    logger.info(f"LINCS Training SMILES: {len(lincs_train_df)}")
    logger.info(f"LINCS Validation SMILES: {len(lincs_val_df)}")
    logger.info(f"Combined Training SMILES: {len(combined_train)}")
    logger.info(f"Combined Validation SMILES: {len(combined_val)}")
    
    return combined_train, combined_val

def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str, seed: int):
    """
    Save all splits to files.
    
    Args:
        splits: Dictionary containing all splits to save
        output_dir: Directory to save splits
        seed: Split seed number
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    file_paths = {
        'lincs_train': f"LINCS-compound-split-{seed}-train.csv",
        'lincs_val': f"LINCS-compound-split-{seed}-val.csv",
        'combined_train': f"JUMP-LINCS-compound-split-{seed}-train.csv",
        'combined_val': f"JUMP-LINCS-compound-split-{seed}-val.csv"
    }
    
    # Save each split
    for split_name, df in splits.items():
        file_path = os.path.join(output_dir, file_paths[split_name])
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {split_name} to: {file_path}")

def process_splits(genomic_data_path: str, jump_data_path: str, output_dir: str, seed: int):
    """
    Main function to process one split: create LINCS splits, concatenate with JUMP, and save.
    
    Args:
        genomic_data_path: Path to processed LINCS data
        jump_data_path: Path to JUMP data directory
        output_dir: Directory to save splits
        seed: Split seed number
    """
    logger.info(f"\nProcessing split with seed {seed}")
    
    # Load LINCS data
    logger.info(f"Loading LINCS data from {genomic_data_path}")
    genomic_df = pd.read_parquet(genomic_data_path)
    
    # Create LINCS splits
    lincs_train_df, lincs_val_df = create_lincs_splits(genomic_df, seed)
    
    # Concatenate with JUMP data
    combined_train_df, combined_val_df = concatenate_with_jump(
        (lincs_train_df, lincs_val_df),
        jump_data_path,
        seed
    )
    
    # Save all splits
    splits = {
        'lincs_train': lincs_train_df,
        'lincs_val': lincs_val_df,
        'combined_train': combined_train_df,
        'combined_val': combined_val_df
    }
    save_splits(splits, output_dir, seed)

def main():
    parser = argparse.ArgumentParser(description="Create LINCS data splits for training")
    
    parser.add_argument("--genomic_data_path", type=str, required=True,
                       help="Path to processed LINCS genomic data (parquet file)")
    parser.add_argument("--jump_data_path", type=str, required=True,
                       help="Path to directory containing JUMP data splits")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save LINCS and combined splits")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2],
                       help="Random seeds for splits (default: 0 1 2)")
    
    args = parser.parse_args()
    
    logger.info(f"Creating LINCS splits with the following parameters:")
    logger.info(f"  Genomic data path: {args.genomic_data_path}")
    logger.info(f"  JUMP data path: {args.jump_data_path}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Seeds: {args.seeds}")
    
    # Process each split
    for seed in args.seeds:
        process_splits(args.genomic_data_path, args.jump_data_path, args.output_dir, seed)
        
    logger.info(f"\nCompleted all splits. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()