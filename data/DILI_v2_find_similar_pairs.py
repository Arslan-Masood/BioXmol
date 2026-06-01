#!/usr/bin/env python3
"""
Identify compound pairs with:
- Tanimoto similarity >= 0.6 using MACCS fingerprints
- Same ATC Level 4 subgroup
- Opposite binary labels (one has label 0, the other has label 1)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from tqdm import tqdm
import itertools

def calculate_maccs_fingerprint(smiles):
    """Calculate MACCS fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except:
        return None

def tanimoto_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return None
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def find_similar_pairs_all_atc(df, similarity_threshold=0.6):
    """
    Alternative version that keeps ALL shared ATC codes for each pair.
    Returns one row per unique compound pair with a list of shared ATC codes.
    """
    # Filter out rows with missing SMILES, ATC_Level_4, or binary_label
    df_clean = df.dropna(subset=['SMILES', 'ATC_Level_4', 'binary_label']).copy()
    df_clean = df_clean[df_clean['ATC_Level_4'] != ''].copy()
    
    # Convert binary_label to numeric and filter for valid 0 or 1 values
    df_clean['binary_label'] = pd.to_numeric(df_clean['binary_label'], errors='coerce')
    df_clean = df_clean.dropna(subset=['binary_label']).copy()
    df_clean = df_clean[df_clean['binary_label'].isin([0.0, 1.0])].copy()
    
    print(f"Processing {len(df_clean)} compounds...")
    
    # Generate MACCS fingerprints
    print("Generating MACCS fingerprints...")
    records = []
    fingerprints = []
    
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Fingerprinting"):
        fp = calculate_maccs_fingerprint(row['SMILES'])
        if fp is not None:
            records.append({
                'Name': row['Name'],
                'CID': row['CID'],
                'SMILES': row['SMILES'],
                'binary_label': row['binary_label'],
                'ATC_Level_4': row['ATC_Level_4']
            })
            fingerprints.append(fp)
    
    df_valid = pd.DataFrame(records).reset_index(drop=True)
    print(f"Successfully generated fingerprints for {len(df_valid)} compounds")
    
    # Expand multi-valued ATC_Level_4 codes
    print("Expanding multi-valued ATC_Level_4 and grouping by individual codes...")
    
    expanded_records = []
    for idx, row in df_valid.iterrows():
        atc_codes = str(row['ATC_Level_4']).split('|')
        for atc_code in atc_codes:
            atc_code = atc_code.strip()
            if atc_code:
                expanded_records.append({
                    'compound_idx': idx,
                    'ATC_Level_4_single': atc_code
                })
    
    df_expanded = pd.DataFrame(expanded_records)
    atc_groups = df_expanded.groupby('ATC_Level_4_single')
    
    # Dictionary to collect all ATC codes for each pair
    pair_data = {}  # key: (cid1, cid2) -> data dict with list of ATC codes
    
    for atc_code, group_df in tqdm(atc_groups, desc="Finding similar pairs"):
        compound_indices = group_df['compound_idx'].unique().tolist()
        
        if len(compound_indices) < 2:
            continue
        
        for i, j in itertools.combinations(compound_indices, 2):
            cid1 = df_valid.loc[i, 'CID']
            cid2 = df_valid.loc[j, 'CID']
            
            if cid1 == cid2:
                continue
            
            # Canonical pair key
            if cid1 < cid2:
                pair_key = (cid1, cid2)
                idx1, idx2 = i, j
            else:
                pair_key = (cid2, cid1)
                idx1, idx2 = j, i
            
            label1 = df_valid.loc[idx1, 'binary_label']
            label2 = df_valid.loc[idx2, 'binary_label']
            
            if label1 == label2:
                continue
            
            # Check if this pair was already found
            if pair_key in pair_data:
                # Just add the new ATC code
                pair_data[pair_key]['ATC_codes'].add(atc_code)
            else:
                # Calculate similarity and create new entry
                fp1 = fingerprints[idx1]
                fp2 = fingerprints[idx2]
                similarity = tanimoto_similarity(fp1, fp2)
                
                if similarity is not None and similarity >= similarity_threshold:
                    pair_data[pair_key] = {
                        'Compound1_Name': df_valid.loc[idx1, 'Name'],
                        'Compound1_CID': cid1 if cid1 < cid2 else cid2,
                        'Compound1_SMILES': df_valid.loc[idx1, 'SMILES'],
                        'Compound1_binary_label': int(label1),
                        'Compound2_Name': df_valid.loc[idx2, 'Name'],
                        'Compound2_CID': cid2 if cid1 < cid2 else cid1,
                        'Compound2_SMILES': df_valid.loc[idx2, 'SMILES'],
                        'Compound2_binary_label': int(label2),
                        'ATC_codes': {atc_code},
                        'Tanimoto_Similarity': similarity
                    }
    
    # Convert to list and join ATC codes
    similar_pairs = []
    for pair_key, data in pair_data.items():
        data['Shared_ATC_Level_4_codes'] = '|'.join(sorted(data['ATC_codes']))
        data['Num_shared_ATC_codes'] = len(data['ATC_codes'])
        del data['ATC_codes']
        similar_pairs.append(data)

    return pd.DataFrame(similar_pairs)
def main():
    # Read the CSV file
    input_file = '/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/Master_DILIRank_Final_Cleaned_with_labels.csv'
    output_file = '/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/similar_pairs_maccs_0.6_opposite_labels.csv'
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Find similar pairs
    similar_pairs_df = find_similar_pairs_all_atc(df, similarity_threshold=0.6)
    
    # Sort by similarity (descending)
    similar_pairs_df = similar_pairs_df.sort_values('Tanimoto_Similarity', ascending=False)
    
    # Save results
    print(f"\nFound {len(similar_pairs_df)} similar pairs")
    print(f"Saving results to {output_file}...")
    similar_pairs_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total pairs found: {len(similar_pairs_df)}")
    print(f"Unique compounds involved: {len(set(similar_pairs_df['Compound1_CID'].tolist() + similar_pairs_df['Compound2_CID'].tolist()))}")
    print(f"ATC Level 4 subgroups: {similar_pairs_df['Shared_ATC_Level_4_codes'].nunique()}")
    print(f"\nBinary label distribution:")
    print(f"  Pairs with Compound1_label=0, Compound2_label=1: {len(similar_pairs_df[similar_pairs_df['Compound1_binary_label'] == 0])}")
    print(f"  Pairs with Compound1_label=1, Compound2_label=0: {len(similar_pairs_df[similar_pairs_df['Compound1_binary_label'] == 1])}")
    print(f"\nSimilarity statistics:")
    print(similar_pairs_df['Tanimoto_Similarity'].describe())

    # Group by ATC Level 4
    print("\n=== Pairs per ATC Level 4 Subgroup ===")
    atc_counts = similar_pairs_df['Shared_ATC_Level_4_codes'].value_counts()
    print(atc_counts.head(20))

if __name__ == '__main__':
    main()

