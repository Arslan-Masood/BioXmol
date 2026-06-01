#!/usr/bin/env python3
"""Run the PyaiVS pipeline for a supplied DILIRank SMILES/label CSV."""

import argparse
import os
from pathlib import Path

def validate_labels_csv(labels_csv: Path) -> None:
    """Ensure the provided CSV has the columns required by PyaiVS."""
    import pandas as pd

    df = pd.read_csv(labels_csv)
    required_cols = {"Smiles", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(
            f"Input CSV must contain columns 'Smiles' and 'label'; missing: {', '.join(sorted(missing))}."
        )


def run_pyaivs(labels_csv: Path, out_dir: Path, cpus: int) -> None:
    """Run PyaiVS parameter optimisation and result generation."""
    from PyaiVS import model_bulid  # type: ignore
    from rdkit import RDLogger

    # Silence RDKit warnings for cleaner logs
    RDLogger.DisableLog("rdApp.*")

    splits = ["scaffold"]
    fingerprints = ["ECFP4"]
    models = [
        "SVM",
        "KNN",
        "RF",
        "XGB",
        "gcn",
        "gat",
        "attentivefp",
        "mpnn",
    ]

    model_bulid.running(
        str(labels_csv),
        out_dir=str(out_dir),
        split=splits,
        model=models,
        FP=fingerprints,
        run_type="param",
        cpus=cpus,
    )

    model_bulid.running(
        str(labels_csv),
        out_dir=str(out_dir),
        split=splits,
        model=models,
        FP=fingerprints,
        run_type="result",
        cpus=cpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyaiVS pipeline for DILIRank data")
    parser.add_argument("--input_csv", type=Path, required=True, help="Path to DILIRank SMILES/label CSV")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store PyaiVS outputs")
    parser.add_argument("--cpus", type=int, default=2, help="Number of CPUs to use (default: 2)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = args.input_csv.resolve()
    print(f"Validating supplied labels file: {labels_csv}")
    validate_labels_csv(labels_csv)

    print(f"Running PyaiVS pipeline (outputs in {output_dir})")
    run_pyaivs(labels_csv, output_dir, args.cpus)
    print("PyaiVS pipeline completed successfully")


if __name__ == "__main__":
    main()

