#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def load_invalid_smiles(report_path: Path) -> set:
    if not report_path.exists():
        raise FileNotFoundError(f"Invalid SMILES report not found: {report_path}")
    df = pd.read_csv(report_path)
    if "smiles" not in df.columns:
        raise ValueError("Invalid SMILES report must contain a 'smiles' column")
    return set(df["smiles"].dropna().astype(str).tolist())


def is_split_csv(p: Path) -> bool:
    if p.suffix.lower() != ".csv":
        return False
    name = p.name
    if name == "chembl20.csv":
        return False
    # Heuristics: include common split filenames used in this repo
    return (
        "split" in name or
        name.endswith("-train.csv") or
        name.endswith("-val.csv") or
        name.endswith("-test.csv")
    )


def filter_split_file(src: Path, dst: Path, invalid_smiles: set, master_df: pd.DataFrame, master_smiles_col: str) -> int:
    df = pd.read_csv(src)
    before = len(df)

    # Case 1: split CSV includes SMILES directly
    for cand in ["SMILES", "smiles"]:
        if cand in df.columns:
            smiles_series = df[cand].astype(str)
            if not invalid_smiles:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Ensure output has a canonical 'SMILES' column
                out_df = df.copy()
                if 'SMILES' not in out_df.columns and 'smiles' in out_df.columns:
                    out_df = out_df.rename(columns={'smiles': 'SMILES'})
                out_df.to_csv(dst, index=False)
                return 0
            df_filtered = df[~smiles_series.isin(invalid_smiles)].copy()
            # Ensure output has a canonical 'SMILES' column
            if 'SMILES' not in df_filtered.columns and 'smiles' in df_filtered.columns:
                df_filtered = df_filtered.rename(columns={'smiles': 'SMILES'})
            dst.parent.mkdir(parents=True, exist_ok=True)
            df_filtered.to_csv(dst, index=False)
            return before - len(df_filtered)

    # Case 2: legacy split with 'index' column referencing master CSV row positions
    if "index" in df.columns:
        indices = df["index"].astype(int)
        # Guard against out of range
        valid_mask = (indices >= 0) & (indices < len(master_df))
        if not valid_mask.all():
            dropped = (~valid_mask).sum()
            print(f"Warning: {src.name}: dropped {dropped} out-of-range indices relative to master file")
        smiles_series = master_df.iloc[indices[valid_mask]][master_smiles_col].astype(str).reset_index(drop=True)
        # Keep only rows whose mapped SMILES are not invalid
        keep_mask = ~smiles_series.isin(invalid_smiles)
        base_df = df[valid_mask].reset_index(drop=True)
        df_filtered = base_df[keep_mask].copy()
        # Attach canonical SMILES column mapped from master
        df_filtered['SMILES'] = smiles_series[keep_mask].values
        dst.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(dst, index=False)
        return before - len(df_filtered)

    raise ValueError(f"Split file {src} must contain 'SMILES'/'smiles' or 'index' column")


def main():
    parser = argparse.ArgumentParser(description="Filter ChEMBL20 split CSVs by removing invalid SMILES")
    parser.add_argument("--splits_dir", required=True, help="Directory containing original split CSVs (e.g., data/chembl20)")
    parser.add_argument("--invalid_report", required=True, help="CSV with 'smiles' column listing invalid SMILES")
    parser.add_argument("--out_dir", required=True, help="Output directory for filtered split CSVs")
    parser.add_argument("--master_file", required=False, default=None, help="Path to master chembl20.csv for index mapping")
    parser.add_argument("--master_smiles_col", required=False, default=None, help="SMILES column name in master file; auto-detect if not set")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    invalid_report = Path(args.invalid_report)
    out_dir = Path(args.out_dir)

    invalid_smiles = load_invalid_smiles(invalid_report)

    # Load master file for index-based splits
    master_file = Path(args.master_file) if args.master_file else (splits_dir / "chembl20.csv")
    if not master_file.exists():
        raise FileNotFoundError(f"Master file not found for index mapping: {master_file}")
    master_df = pd.read_csv(master_file)
    # Auto-detect master SMILES column
    master_smiles_col = args.master_smiles_col
    if master_smiles_col is None:
        for cand in ["smiles", "SMILES"]:
            if cand in master_df.columns:
                master_smiles_col = cand
                break
    if master_smiles_col is None:
        raise ValueError(f"Cannot find SMILES column in master file {master_file}; set --master_smiles_col")

    split_files = sorted([p for p in splits_dir.iterdir() if p.is_file() and is_split_csv(p)])
    if not split_files:
        print(f"No split CSVs found in {splits_dir}")
        return

    total_removed = 0
    for src in split_files:
        dst = out_dir / src.name
        removed = filter_split_file(src, dst, invalid_smiles, master_df, master_smiles_col)
        total_removed += removed
        print(f"Filtered {src.name}: removed {removed} rows -> {dst}")

    print(f"Done. Total rows removed across splits: {total_removed}")


if __name__ == "__main__":
    main()


