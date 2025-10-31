#!/usr/bin/env python3
"""
Script to check for invalid SMILES in genomic and morphological datasets.
Generates a CSV report of failed molecules that can be used to filter them out.

Usage:
    python check_invalid_smiles.py --cell_data <path> --genomic_data <path> --output <path>
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import traceback

# Add mocop to path
sys.path.insert(0, str(Path(__file__).parent / "mocop"))

from featurizer.smiles_transformation import smiles2graph


def check_smiles_validity(smiles: str) -> Tuple[bool, str, str]:
    """
    Check if a SMILES string can be converted to a valid molecular graph.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Tuple of (is_valid, error_type, error_message)
    """
    if pd.isna(smiles) or smiles is None or smiles == "":
        return False, "EMPTY_SMILES", "SMILES is None, NaN, or empty"
    
    if smiles.lower() in ["restricted", "nan", "none"]:
        return False, "INVALID_SMILES", f"SMILES is '{smiles}' (restricted/invalid)"
    
    try:
        # Try to convert SMILES to graph
        result = smiles2graph(smiles)
        if result is None:
            return False, "INVALID_SMILES", "RDKit returned None"
        return True, "SUCCESS", "Valid SMILES"
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        return False, error_type, error_msg


def process_dataset(df: pd.DataFrame, dataset_name: str, smiles_col: str = "Metadata_SMILES") -> List[Dict]:
    """
    Process a dataset and check SMILES validity.
    
    Args:
        df: DataFrame containing the data
        dataset_name: Name of the dataset (e.g., "cell", "genomic")
        smiles_col: Name of the SMILES column
        
    Returns:
        List of dictionaries with failure information
    """
    failures = []
    
    # Get unique SMILES
    if smiles_col not in df.columns:
        print(f"Warning: {smiles_col} not found in {dataset_name} dataset. Available columns: {list(df.columns)}")
        return failures
    
    unique_smiles = df[smiles_col].dropna().unique()
    print(f"Checking {len(unique_smiles)} unique SMILES in {dataset_name} dataset...")
    
    for idx, smiles in enumerate(unique_smiles):
        is_valid, error_type, error_msg = check_smiles_validity(smiles)
        
        if not is_valid:
            failure_info = {
                "idx": idx,
                "smiles": smiles,
                "dataset": dataset_name,
                "error_type": error_type,
                "error_msg": error_msg,
                "success": False
            }
            failures.append(failure_info)
            
            if len(failures) % 10 == 0:
                print(f"  Found {len(failures)} invalid SMILES so far...")
    
    print(f"Found {len(failures)} invalid SMILES in {dataset_name} dataset")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Check for invalid SMILES in datasets")
    parser.add_argument("--cell_data", required=True, help="Path to cell/morphological data file (CSV/Parquet)")
    parser.add_argument("--genomic_data", required=True, help="Path to genomic data file (Parquet)")
    parser.add_argument("--output", required=True, help="Output CSV file path for failed molecules report")
    parser.add_argument("--smiles_col", default="Metadata_SMILES", help="Name of SMILES column (default: Metadata_SMILES)")
    
    args = parser.parse_args()
    
    # Load datasets
    print("Loading datasets...")
    
    # Load cell data
    cell_path = Path(args.cell_data)
    if cell_path.suffix == ".parquet":
        cell_df = pd.read_parquet(cell_path)
    else:
        cell_df = pd.read_csv(cell_path)
    print(f"Loaded cell data: {len(cell_df)} rows")
    
    # Load genomic data
    genomic_path = Path(args.genomic_data)
    genomic_df = pd.read_parquet(genomic_path)
    print(f"Loaded genomic data: {len(genomic_df)} rows")
    
    # Check if SMILES column exists
    for df, name in [(cell_df, "cell"), (genomic_df, "genomic")]:
        if args.smiles_col not in df.columns:
            print(f"Error: {args.smiles_col} column not found in {name} dataset")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Check SMILES validity
    all_failures = []
    
    # Check cell data
    cell_failures = process_dataset(cell_df, "cell", args.smiles_col)
    all_failures.extend(cell_failures)
    
    # Check genomic data
    genomic_failures = process_dataset(genomic_df, "genomic", args.smiles_col)
    all_failures.extend(genomic_failures)
    
    # Create summary
    if all_failures:
        failures_df = pd.DataFrame(all_failures)
        
        # Save to CSV
        output_path = Path(args.output)
        failures_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(all_failures)} failed molecules to {output_path}")
        
        # Print summary
        print("\nFailure Summary:")
        print(f"Total invalid SMILES: {len(all_failures)}")
        print(f"Cell dataset failures: {len(cell_failures)}")
        print(f"Genomic dataset failures: {len(genomic_failures)}")
        
        print("\nError types:")
        error_counts = failures_df["error_type"].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
        
        print("\nSample failures:")
        for _, row in failures_df.head(5).iterrows():
            print(f"  {row['smiles']} ({row['dataset']}): {row['error_type']} - {row['error_msg']}")
    else:
        print("\nNo invalid SMILES found! All molecules are valid.")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=["idx", "smiles", "dataset", "error_type", "error_msg", "success"])
        empty_df.to_csv(args.output, index=False)
        print(f"Created empty report file: {args.output}")


if __name__ == "__main__":
    main()
