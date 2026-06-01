import pandas as pd
import numpy as np
import os
import re  # Added regex module to handle the '|' separator



work_dir = '/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data'

# 1. Load the file you just downloaded from PubChem Website
pubchem_map = pd.read_csv(os.path.join(work_dir, 'result_drugbank_ids.txt'), 
                          sep='\t', names=['CID', 'DrugBank_ID'], dtype=str)

# 2. Load your local DrugBank file
# (Ensure columns match your actual file headers)
drugbank = pd.read_csv("/scratch/work/masooda1/datasets/downstream_datasets/drugbank.tsv", sep='\t', dtype=str)

# 3. Merge them
# We match PubChem's 'DrugBank_ID' to DrugBank's 'DrugBank ID' column
final = pubchem_map.merge(drugbank, left_on='DrugBank_ID', right_on='drugbank_id', how='left')
final = final[['CID', 'DrugBank_ID', 'atc_codes']]
final.columns = ['CID', 'DrugBank_ID', 'ATC_Code']
# 4. Save

final.to_csv(os.path.join(work_dir, 'Final_DILIRank_with_DrugBank.csv'), index=False)
print("Done! ATC codes retrieved from local DrugBank file.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
work_dir = '/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data'
output_file = os.path.join(work_dir, 'Master_DILIRank_Final_Cleaned.csv')

print(f"📂 Working Directory: {work_dir}")

# ==========================================
# 2. LOAD ALL FILES
# ==========================================
print("\n⬇️  Loading files...")

# A. Base Map
df_base = pd.read_csv(os.path.join(work_dir, 'step1_name_cid_map.csv'), dtype=str)

# B. SMILES
df_smiles = pd.read_csv(os.path.join(work_dir, 'result_smiles.txt'), 
                        sep='\t', names=['CID', 'SMILES'], dtype=str)

# C. Parent CIDs
df_parents = pd.read_csv(os.path.join(work_dir, 'result_parents.txt'), 
                         sep='\t', names=['CID', 'Parent_CID'], dtype=str)

# D. Parent Names
df_parent_names = pd.read_csv(os.path.join(work_dir, 'result_parent_names.txt'), 
                              sep='\t', names=['Parent_CID', 'Parent_Name'], dtype=str)

# E. DrugBank IDs
df_db_ids = pd.read_csv(os.path.join(work_dir, 'result_drugbank_ids.txt'), 
                        sep='\t', names=['CID', 'DrugBank_ID'], dtype=str)
                   

try:
    atc_path = os.path.join(work_dir, 'Final_DILIRank_with_DrugBank.csv')
    df_atc_source = pd.read_csv(atc_path, dtype=str)
    
    if 'CID' in df_atc_source.columns and 'ATC_Code' in df_atc_source.columns:
        df_atc = df_atc_source[['CID', 'ATC_Code']].drop_duplicates(subset=['CID'])
    else:
        df_atc = pd.DataFrame(columns=['CID', 'ATC_Code'])
except FileNotFoundError:
    df_atc = pd.DataFrame(columns=['CID', 'ATC_Code'])

# ==========================================
# 3. MERGE DATA
# ==========================================
print("\n🔄 Merging datasets...")
master = df_base.merge(df_smiles, on='CID', how='left')
master = master.merge(df_parents, on='CID', how='left')
master = master.merge(df_parent_names, on='Parent_CID', how='left')
master = master.merge(df_db_ids, on='CID', how='left')
master = master.merge(df_atc, on='CID', how='left')

# Extract ATC codes based on Parent Name (only for rows where ATC_Code is missing)
null_ATC_codes = master[master.ATC_Code.isnull()].copy()

if not null_ATC_codes.empty:
    # Look up ATC codes in DrugBank using Parent_Name -> name
    merging_two = pd.merge(
        null_ATC_codes,
        drugbank,
        left_on="Parent_Name",
        right_on="name",
        how="left"
    )

    # Keep a simple CID → ATC mapping from this parent-name match
    if "atc_codes" in merging_two.columns:
        parent_atc = (
            merging_two[["CID", "atc_codes"]]
            .dropna(subset=["atc_codes"])
            .drop_duplicates(subset=["CID"])
            .rename(columns={"atc_codes": "ATC_from_parent"})
        )

        # Merge this back into the full master table
        master = master.merge(parent_atc, on="CID", how="left")

        # Fill missing ATC_Code from ATC_from_parent
        master["ATC_Code"] = master["ATC_Code"].fillna(master["ATC_from_parent"])

# ==========================================
# 4. CLEAN & DEDUPLICATE
# ==========================================
print("\n🧹 Cleaning and Deduplicating...")

master['CID_Numeric'] = pd.to_numeric(master['CID'], errors='coerce')
master['Has_DB'] = master['DrugBank_ID'].notna()
master['Has_ATC'] = master['ATC_Code'].notna()

# Sort so that, within each Name:
#   1) rows with ATC_Code are first
#   2) then rows with DrugBank_ID
#   3) then by numeric CID
master = master.sort_values(
    by=['Name', 'Has_ATC', 'Has_DB', 'CID_Numeric'],
    ascending=[True, False, False, True]
)

# Now drop duplicates on Name, keeping the first row (which should have ATC_Code if available)
master_clean = master.drop_duplicates(subset=['Name'], keep='first').copy()

# ==========================================
# 5. NEW: GENERATE MULTIPLE ATC LEVEL 4 (FIXED)
# ==========================================
print("\n🧪 Generating Multiple ATC Level 4 codes...")

def extract_level_4_multi(atc_str):
    if pd.isna(atc_str):
        return None
    
    # FIX: Regex split allows splitting by '|' OR ';' OR ','
    # This handles "A07AA02|D01AA01" correctly
    raw_codes = re.split(r'[|;,]', str(atc_str))
    
    level_4_set = set()
    for code in raw_codes:
        clean_code = code.strip()
        if len(clean_code) >= 5:
            level_4_set.add(clean_code[:5])
        elif clean_code: 
            level_4_set.add(clean_code)
            
    if not level_4_set:
        return None
    
    # Return separated by pipe '|' to match your source style, or use '; '
    return "|".join(sorted(level_4_set))

master_clean['ATC_Level_4'] = master_clean['ATC_Code'].apply(extract_level_4_multi)

# ==========================================
# 6. SAVE
# ==========================================
cols_order = ['Name', 'CID', 'SMILES', 'Parent_CID', 'Parent_Name', 'DrugBank_ID', 'ATC_Code', 'ATC_Level_4']
valid_cols = [c for c in cols_order if c in master_clean.columns]
master_clean = master_clean[valid_cols]

master_clean.to_csv(output_file, index=False)
print(f"\n✅ SUCCESS! Final clean file saved to:\n   {output_file}")

# ==========================================
# 7. Create binary labels
# ==========================================

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

dili_path = "/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/DILIRank_raw.xlsx"
dili = pd.read_excel(dili_path, header = 1)

df = pd.merge(master_clean, dili, left_on = "Name", right_on = "CompoundName")

# Standardize labels
label_column = "vDILI-Concern"
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

df.to_csv("/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/Master_DILIRank_Final_Cleaned_with_labels.csv", index=False)