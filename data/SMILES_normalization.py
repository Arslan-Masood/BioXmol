#!/usr/bin/env python3
"""
SMILES normalization script using RDKit's MolStandardize.
Normalizes SMILES strings in a CSV file and saves the results.
"""

import numpy as np
import pandas as pd
import argparse
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

# Disable RDKit info logging
RDLogger.DisableLog('rdApp.info')

def standardize_smiles(smiles: str, remover=SaltRemover()) -> str:
    """
    Standardize a SMILES string using RDKit's MolStandardize.
    
    Args:
        smiles (str): Input SMILES string
        remover (SaltRemover): RDKit SaltRemover instance
        
    Returns:
        str: Standardized SMILES string or np.nan if standardization fails
    """
    config = {
        "StandardizeSmiles": True,
        "FragmentParent": True,
        "SaltRemover": True,
        "isomericSmiles": True,
        "kekuleSmiles": True,
        "canonical": True
    }
    
    try:
        if pd.isna(smiles) or smiles is None or smiles == "":
            return np.nan
            
        if config["StandardizeSmiles"]:
            smiles = rdMolStandardize.StandardizeSmiles(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        
        if config["SaltRemover"]:
            mol = remover.StripMol(mol, dontRemoveEverything=False)

        if config["FragmentParent"]:
            mol = rdMolStandardize.FragmentParent(mol)

        if config["kekuleSmiles"]:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            
        normalized_smiles = Chem.MolToSmiles(
            mol,
            isomericSmiles=config["isomericSmiles"],
            kekuleSmiles=config["kekuleSmiles"],
            canonical=config["canonical"],
            allHsExplicit=False
        )
        
        return normalized_smiles if normalized_smiles else np.nan
        
    except:
        return np.nan

def normalize_smiles_parallel(smiles_list: list, num_workers: int = None) -> list:
    """
    Process a list of SMILES strings in parallel using multiprocessing.
    
    Args:
        smiles_list (list): List of SMILES strings to normalize
        num_workers (int): Number of parallel workers (default: cpu_count())
        
    Returns:
        list: List of normalized SMILES strings
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(num_workers) as pool:
        results = []
        with tqdm(total=len(smiles_list), desc="Normalizing SMILES") as pbar:
            # Call standardize_smiles directly - it uses default SaltRemover() argument
            for normalized_smiles in pool.imap(standardize_smiles, smiles_list):
                results.append(normalized_smiles)
                pbar.update(1)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize SMILES strings in a CSV file")
    parser.add_argument(
        "--input_file", "-i", type=str, required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_file", "-o", type=str, required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--smiles_col", "-s", type=str, default="SMILES",
        help="Name of the column that contains SMILES strings (default: 'SMILES')"
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=None,
        help=f"Number of parallel workers (default: {cpu_count()})"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows from {args.input_file}")

        # Keep only rows with non-null labels (binary_label)
        if "binary_label" not in df.columns:
            raise ValueError("Input file must contain a 'binary_label' column to filter labeled rows.")
        initial_total = len(df)
        df = df[df["binary_label"].notna()].reset_index(drop=True)
        filtered_out = initial_total - len(df)
        print(f"Filtered to rows with non-null binary_label: {len(df)} remaining (removed {filtered_out})")

        if args.smiles_col not in df.columns:
            raise ValueError(f"Input file must contain a '{args.smiles_col}' column.")

        # Normalize SMILES column
        print(f"\nNormalizing '{args.smiles_col}' column...")
        smiles_list = df[args.smiles_col].tolist()
        normalized_smiles = normalize_smiles_parallel(smiles_list, args.num_workers)
        normalized_col_name = f"{args.smiles_col}_Normalized"
        df[normalized_col_name] = normalized_smiles

        # Statistics
        successful = pd.notna(normalized_smiles).sum()
        failed = len(normalized_smiles) - successful
        print(f"  Successful: {successful}/{len(normalized_smiles)} ({successful/len(normalized_smiles)*100:.1f}%)")
        print(f"  Failed: {failed}/{len(normalized_smiles)} ({failed/len(normalized_smiles)*100:.1f}%)")

        # For compatibility with downstream code, also create Normalized_SMILES_combined
        print(f"\nCreating Normalized_SMILES_combined column (from {normalized_col_name})...")
        df["Normalized_SMILES_combined"] = df[normalized_col_name]

        # Drop rows without a valid normalized SMILES
        initial_count = len(df)
        df = df[df["Normalized_SMILES_combined"].notna()].reset_index(drop=True)
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with failed SMILES normalization")
        print(f"Final dataset: {len(df)} rows")

        # Save results
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
        df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to: {args.output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
