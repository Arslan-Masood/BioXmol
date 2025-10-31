import pandas as pd

data_path = "/scratch/project_462000766/Multi_Modal_Contrastive/data/jump_data/cell_fetures_with_smiles.parquet"
genomic_data_path = "/scratch/cs/pml/AI_drug/molecular_representation_learning/LINCS/landmark_cmp_data_min1000compounds_all_measurements.parquet"

#JUMP-LINCS
print("Checking JUMP-LINCS splits")
train = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/JUMP-LINCS-compound-split-0-train.csv"
val = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/JUMP-LINCS-compound-split-0-val.csv"
test = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/JUMP-LINCS-compound-split-0-val.csv"

#JUMP
#print("Checking JUMP splits")
#train = "/scratch/project_462000766/Multi_Modal_Contrastive/data/jump/jump-compound-split-0-train.csv"
#val = "/scratch/project_462000766/Multi_Modal_Contrastive/data/jump/jump-compound-split-0-val.csv"
#test = "/scratch/project_462000766/Multi_Modal_Contrastive/data/jump/jump-compound-split-0-val.csv"

# LINCS
#print("Checking LINCS splits")
#train = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/LINCS-compound-split-0-train.csv"
#val = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/LINCS-compound-split-0-val.csv"
#test = "/scratch/project_462000766/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/LINCS-compound-split-0-val.csv"

# read datafile
cell_data = pd.read_parquet(data_path)
genomic_data = pd.read_parquet(genomic_data_path)

SMILES_col = "Metadata_SMILES"
genomic_smiles = genomic_data[SMILES_col].unique().tolist()
cell_smiles = cell_data[SMILES_col].unique().tolist()

unique_smiles = (
            list(genomic_smiles) +  # First: all SMILES with genomic data
            list(set(cell_smiles) - set(genomic_smiles))  # Then: SMILES with only morphological data
        )

print(f"genomic SMILES {len(genomic_smiles)}, cell SMILES {len(cell_smiles)}, total unique SMILES {len(unique_smiles)}")

# train split
train_SMILES = pd.read_csv(train)
print("SMILES in train split", train_SMILES.SMILES.nunique())

# train SMILES in cell data, and genomic data
print("How many train split SMILES in cellular data", len(train_SMILES[train_SMILES.SMILES.isin(cell_smiles)].SMILES))
print("How many train split SMILES in genomic data",len(train_SMILES[train_SMILES.SMILES.isin(genomic_smiles)].SMILES))

# val split
val_SMILES = pd.read_csv(val)
print("SMILES in val split", val_SMILES.SMILES.nunique())

# train SMILES in cell data, and genomic data
print("How many val split SMILES in cellular data", len(val_SMILES[val_SMILES.SMILES.isin(cell_smiles)].SMILES))
print("How many val split SMILES in genomic data",len(val_SMILES[val_SMILES.SMILES.isin(genomic_smiles)].SMILES))