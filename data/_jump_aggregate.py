import os
import pandas as pd
import argparse
from tqdm import tqdm

def drop_bad_columns(df):
    cols = [c for c in df.columns if "Metadata" not in c]
    stdev = [df[c].std() for c in cols]

    cols_to_drop = []
    #cols_to_drop.extend([cols[i] for i, s in enumerate(stdev) if s < 0.1 or s > 5])
    # Drop only low variance columns
    cols_to_drop.extend([cols[i] for i, s in enumerate(stdev) if s < 0.1])
    #cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_RWC" in c])
    #cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_Manders" in c])
    #cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_14" in c])
    #cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_15" in c])
    #cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_16" in c])

    df = df[[c for c in df.columns if c not in cols_to_drop]]
    return df

def clean_dataframe(df, metadata_keyword="Metadata"):
    """
    Clean DataFrame by dropping feature columns with NaN values while preserving metadata columns,
    then remove rows with NaN values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing both feature and metadata columns
    metadata_keyword : str, default="Metadata"
        Keyword used to identify metadata columns
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with:
        - Feature columns containing NaN values removed
        - Metadata columns preserved
        - Rows containing NaN values removed
    """
    # Get feature and metadata columns
    feature_cols = [c for c in df.columns if metadata_keyword not in c]
    metadata_cols = [c for c in df.columns if metadata_keyword in c]
    
    # Drop feature columns with NaN values
    feature_df = df[feature_cols].dropna(axis=1)
    
    # Combine back with metadata columns
    cleaned_df = pd.concat([df[metadata_cols], feature_df], axis=1)
    
    # Drop rows with NaN values
    cleaned_df = cleaned_df.dropna(axis=0)
    
    return cleaned_df


def main(args):
    print("Starting aggregation...")
    data_dir = args.data_dir
    output_dir = args.output_dir
    is_centered = args.is_centered
    
    # Get list of files to process
    files_to_process = []
    for f in os.listdir(data_dir):
        if not f.endswith('.parquet'):
            continue
        if 'filtered' in f:  # Skip already processed files
            continue
            
        if is_centered:
            # Only process centered files when flag is True
            if "centered" in f:
                files_to_process.append(f)
        else:
            # Only process original files when flag is False (exclude centered files)
            if "centered" not in f:
        files_to_process.append(f)
    
    print(f"Found {len(files_to_process)} files to process")
    
    dfs = []
    # Add tqdm progress bar for file processing
    for f in tqdm(files_to_process, desc="Processing files"):
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)

    print("Concatenating dataframes...")
    df = pd.concat(dfs)
    
    print("Dropping bad columns...")
    df = drop_bad_columns(df)

    total_smiles = df.shape[0]
    print(f"Total SMILES: {total_smiles}")
    
    print("Cleaning dataframe...")
    df = clean_dataframe(df)
    
    dropped_smiles = total_smiles - df.shape[0]
    print(f"Dropped SMILES: {dropped_smiles}")
    
    print("Final cleanup...")
    df = df.dropna(axis=1)
    
    filename = f"{'centered.' if is_centered else ''}filtered.parquet"
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath)
    print(f"File saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("--is_centered", default=False, action="store_true")
    args = parser.parse_args()
    main(args)