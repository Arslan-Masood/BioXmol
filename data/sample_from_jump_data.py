import pandas as pd
import numpy as np

# Read the original parquet file
print("Reading the original parquet file")
cell_fetures_with_smiles = pd.read_parquet("/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/cell_fetures_with_smiles.parquet")

# Set random seed for reproducibility
np.random.seed(42)

# Sample 2000 rows randomly
print("Sampling 3000 rows randomly")
sampled_df = cell_fetures_with_smiles.sample(n=3000, random_state=42)

# Save the sampled data to a new parquet file
output_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/cell_fetures_with_smiles_3000.parquet"
print(f"Saving the sampled data to {output_path}")
sampled_df.to_parquet(output_path)
