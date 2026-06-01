import sys 
sys.path.insert(1, '/scratch/project_462000766/Multi_Modal_Contrastive/mocop')
from pathlib import Path

from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from featurizer.smiles_transformation import (inchi2smiles, smiles2fp,
                                              smiles2graph)
import deepchem as dc

# Minimal feature dataset wrapper for linear probing / splitting on precomputed features
class FeatureDataset:
    """
    Minimal wrapper to reuse _split_data for precomputed feature matrices.
    Exposes:
      - unique_smiles: list of SMILES (ids)
      - df: pandas DataFrame containing the label column
    """
    def __init__(self, smiles: List[str], labels, label_col: str):
        self.unique_smiles = list(smiles)
        self.df = pd.DataFrame({label_col: labels})

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, idx):
        return idx

class SupervisedGraphDataset(Dataset):
    def __init__(
        self, data_path, cmpd_col="smiles", label_col=None, cmpd_col_is_inchikey=False, pad_length=0, invalid_smiles_report=None
    ):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.df = self.df.set_index(cmpd_col)
        if cmpd_col_is_inchikey:
            self.df.index = [inchi2smiles(s) for s in self.df.index]
        if label_col is None:
            self.df = self.df[[c for c in self.df.columns if not c.startswith("Metadata")]]
        else:
            # Keep only the specified label column(s)
            if isinstance(label_col, str):
                label_col = [label_col]  # Convert to list for consistent handling
            self.df = self.df[label_col]
            
            #print(f"Dataset loaded with specific label columns: {label_col}")
            #for col in label_col:
                #print(f"{col} distribution:")
                #print(self.df[col].value_counts().sort_index())
        
        # Optionally filter out invalid SMILES using a precomputed report
        if invalid_smiles_report is not None:
            report_path = Path(invalid_smiles_report)
            if report_path.exists():
                try:
                    failed_df = pd.read_csv(report_path)
                    failed_smiles = set(failed_df.get("smiles", []))
                    if failed_smiles:
                        before = len(self.df)
                        self.df = self.df[~self.df.index.isin(failed_smiles)]
                        removed = before - len(self.df)
                        if removed:
                            print(f"Filtered out {removed} invalid SMILES using {report_path.name}")
                except Exception as e:
                    print(f"Warning: failed to read invalid SMILES report at {report_path}: {e}")

        self.unique_smiles = self.df.index
        # Build a lookup from SMILES to the current (post-filter) integer index
        # for robust and fast mapping of external split files
        self._smiles_to_idx = {s: i for i, s in enumerate(self.unique_smiles)}
        self.pad_length = pad_length
        print(f"Total compounds with valid SMILES: {len(self.unique_smiles)}")


    def __len__(self):
        return len(self.unique_smiles)

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        labels = self.df.loc[smiles]

        if len(labels.shape) > 1 and len(labels) > 1:
            labels = labels.sample(1).iloc[0]

        labels = torch.FloatTensor(labels.values)
        return {
            "inputs": {"x_a": [torch.FloatTensor(f) for f in cmpd_feat]},
            "labels": labels,
        }


class SupervisedGraphDatasetJUMP(SupervisedGraphDataset):
    def __init__(self, *args, **kwargs):
        super(SupervisedGraphDataset, self).__init__(*args, **kwargs)
        self.unique_smiles = self.df.index.unique()


class DualInputDatasetJUMP(Dataset):
    def __init__(self, data_path):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col not in self.df.columns:
            self.df[self.smiles_col] = [
                inchi2smiles(s) if s is not None else None
                for s in self.df["Metadata_InChI"]
            ]

        self.unique_smiles = [
            s for s in self.df[self.smiles_col].unique() if s is not None
        ]

        self.morph_col = [c for c in self.df.columns if not c.startswith("Metadata_")]
        self.smiles2mask = {}

    def _create_index(self):
        smiles = self.df[self.smiles_col].values
        return {s: np.argwhere(smiles == s).reshape(-1) for s in smiles}

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        cmpd_feat = smiles2fp(smiles)

        df = self.df[self.df[self.smiles_col] == smiles]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()

        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": torch.FloatTensor(cmpd_feat),
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


class DualInputGraphDatasetJUMP(DualInputDatasetJUMP):
    def __init__(self, pad_length, *args, **kwargs):
        super(DualInputGraphDatasetJUMP, self).__init__(*args, **kwargs)
        self.pad_length = pad_length

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        try:
            mask = self.smiles2mask[smiles]
        except KeyError:
            mask = self.df[self.smiles_col] == smiles
            self.smiles2mask[smiles] = mask
        df = self.df[mask]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()
        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


class TripleInputGraphDatasetJUMP(DualInputGraphDatasetJUMP):
    def __init__(self, data_path, genomic_data_path, pad_length, *args, **kwargs):
        # Define SMILES column name first
        self.smiles_col = "Metadata_SMILES"
        
        # Load cell data
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        # Load genomic data
        self.genomic_df = pd.read_parquet(genomic_data_path)

        # Ensure SMILES column exists in both datasets
        for df in [self.df, self.genomic_df]:
            if self.smiles_col not in df.columns:
                df[self.smiles_col] = [
                    inchi2smiles(s) if s is not None else None
                    for s in df["Metadata_InChI"]
                ]
        
        # Get unique SMILES from each dataset (excluding None)
        cell_smiles = set([s for s in self.df[self.smiles_col].unique() if s is not None])
        genomic_smiles = set([s for s in self.genomic_df[self.smiles_col].unique() if s is not None])
        
        # Print initial statistics
        print("\nSMILES Statistics:")
        print(f"Cell data unique SMILES: {len(cell_smiles)}")
        print(f"Genomic data unique SMILES: {len(genomic_smiles)}")
        print(f"Common SMILES (intersection): {len(cell_smiles.intersection(genomic_smiles))}")
        
        # Combine all unique SMILES (excluding None)
        self.unique_smiles = list(cell_smiles.union(genomic_smiles))
        
        # Print final dataset composition
        print(f"\nFinal Dataset:")
        print(f"Total unique SMILES: {len(self.unique_smiles)}")
        
        # Set remaining attributes
        self.pad_length = pad_length
        self.genomic_cols = [c for c in self.genomic_df.columns if not c.startswith("Metadata_")]
        self.morph_cols = [c for c in self.df.columns if not c.startswith("Metadata_")]

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        # Get graph features (x_a)
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)
        
        # Get morphological features (x_b)
        morph_mask = self.df[self.smiles_col] == smiles
        if morph_mask.any():
            morph_feat = self.df[morph_mask].sample(1)[self.morph_cols].values.astype(float).flatten()
        else:
            morph_feat = -1 * np.ones(len(self.morph_cols))
            
        # Get genomic features (x_c)
        genomic_mask = self.genomic_df[self.smiles_col] == smiles
        if genomic_mask.any():
            genomic_feat = self.genomic_df[genomic_mask].sample(1)[self.genomic_cols].values.astype(float).flatten()
        else:
            genomic_feat = -1 * np.ones(len(self.genomic_cols))
        
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
                "x_c": torch.FloatTensor(genomic_feat)
            },
            "labels": torch.Tensor([-1])
        }


class CellLineTripleInputGraphDatasetJUMP(DualInputGraphDatasetJUMP):
    """Dataset class for handling molecular data with cell line-specific genomic features.
    
    This class processes three types of data:
    1. Molecular graph features (x_a): Structural information about compounds
    2. Morphological features (x_b): Cell morphology measurements
    3. Genomic features (x_c): Gene expression data for different cell lines
    
    Each compound (SMILES) can have:
    - Morphological data only
    - Genomic data only
    - Both morphological and genomic data
    - Genomic data for multiple cell lines
    """

    def __init__(self, data_path, genomic_data_path, pad_length, invalid_smiles_report = None, *args, **kwargs):
        """Initialize the dataset.
        
        Args:
            data_path (str): Path to morphological data file (CSV/Parquet)
            genomic_data_path (str): Path to genomic data file (Parquet)
            pad_length (int): Maximum length for padding molecular graphs
            invalid_smiles_report (str): Path to invalid SMILES report file (CSV)
        """
        # Define column names for data identification
        self.smiles_col = "Metadata_SMILES"
        self.cell_line_col = "Metadata_cell_iname"
        
        # Load data files
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        self.genomic_df = pd.read_parquet(genomic_data_path)
        
        # Convert InChI to SMILES if needed
        for df in [self.df, self.genomic_df]:
            if self.smiles_col not in df.columns:
                df[self.smiles_col] = [
                    inchi2smiles(s) if s is not None else None
                    for s in df["Metadata_InChI"]
                ]
        
        # Get unique SMILES from each dataset (excluding None)
        cell_smiles = set([s for s in self.df[self.smiles_col].unique() if s is not None])
        genomic_smiles = set([s for s in self.genomic_df[self.smiles_col].unique() if s is not None])
        
        # Create ordered list: genomic SMILES first, then morphological-only SMILES
        self.unique_smiles = (
            list(genomic_smiles) +  # First: all SMILES with genomic data
            list(cell_smiles - genomic_smiles)  # Then: SMILES with only morphological data
        )

        failed_report = Path(invalid_smiles_report)
        if failed_report.exists():
            failed_df = pd.read_csv(failed_report)
            failed_smiles = set(failed_df.get('smiles', []))
            if failed_smiles:
                before = len(self.unique_smiles)
                self.unique_smiles = [s for s in self.unique_smiles if s not in failed_smiles]
                self.df = self.df[~self.df[self.smiles_col].isin(failed_smiles)].reset_index(drop=True)
                self.genomic_df = self.genomic_df[~self.genomic_df[self.smiles_col].isin(failed_smiles)].reset_index(drop=True)
                removed = before - len(self.unique_smiles)
                if removed:
                    print(f"Filtered out {removed} invalid SMILES using {failed_report.name}")
        
        # Print dataset composition statistics
        print("\nDataset Statistics:")
        print(f"Cell data unique SMILES: {len(cell_smiles)}")
        print(f"Genomic data unique SMILES: {len(genomic_smiles)}")
        print(f"Common SMILES (intersection): {len(cell_smiles.intersection(genomic_smiles))}")
        print(f"Morphological-only SMILES: {len(cell_smiles - genomic_smiles)}")
        print(f"Total unique SMILES: {len(self.unique_smiles)}")
        print(f"\nSMILES order guarantee:")
        print(f"- First {len(genomic_smiles)} SMILES have genomic data")
        print(f"- Last {len(cell_smiles - genomic_smiles)} SMILES have only morphological data")
        
        # Modify cell line indexing to start from 1 (0 will be padding)
        self.unique_cell_lines = sorted(list(self.genomic_df[self.cell_line_col].unique()))
        self.cell_line_to_idx = {cell: idx + 1 for idx, cell in enumerate(self.unique_cell_lines)}
        print(f"Number of unique cell lines: {len(self.unique_cell_lines)}")

         # Create mappings for dose levels and time points (1-based indexing, 0 for padding)
        self.unique_doses = sorted(list(self.genomic_df['Metadata_Dose_Level'].unique()))
        self.dose_to_idx = {dose: idx + 1 for idx, dose in enumerate(self.unique_doses)}
        print(f"Dose levels: {self.unique_doses}")
        print(f"Dose mapping: {self.dose_to_idx}")  # e.g., {2:1, 5:2, 7:3, 8:4}

        self.unique_times = sorted(list(self.genomic_df['Metadata_pert_time'].unique()))
        self.time_to_idx = {time: idx + 1 for idx, time in enumerate(self.unique_times)}
        print(f"Time points: {self.unique_times}")
        print(f"Time mapping: {self.time_to_idx}")  # e.g., {6:1, 24:2}
        
        # Set parameters for feature extraction
        self.pad_length = pad_length
        self.genomic_cols = [c for c in self.genomic_df.columns if not c.startswith("Metadata_")]
        self.morph_cols = [c for c in self.df.columns if not c.startswith("Metadata_")]
        
        # Print feature dimensions
        print(f"\nFeature Dimensions:")
        print(f"Morphological features: {len(self.morph_cols)}")
        print(f"Genomic features: {len(self.genomic_cols)}")


    def __getitem__(self, index):
        # Create a new, isolated random number generator for this specific item fetch.
        # ensuring samples are different each time.
        rng = np.random.default_rng()
        
        smiles = self.unique_smiles[index]
        
        # Get graph and morph features
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)
        
        # sample 1 replicate for each drug perturbation, passing the dedicated generator
        morph_mask = self.df[self.smiles_col] == smiles
        morph_feat = (self.df[morph_mask].sample(1, random_state=rng)[self.morph_cols].values.astype(float).flatten() 
                     if morph_mask.any() else -1 * np.ones(len(self.morph_cols)))
        
        # Get only valid genomic conditions for this SMILES
        valid_genomic_data = []
        valid_cell_indices = []
        valid_doses = []
        valid_times = []
        
        smiles_mask = self.genomic_df[self.smiles_col] == smiles
        if smiles_mask.any():
            # Convert to list for pickling compatibility with multiprocessing
            for cell_line in list(self.genomic_df[smiles_mask][self.cell_line_col].unique()):
                cell_idx = self.cell_line_to_idx[cell_line]
                mask = smiles_mask & (self.genomic_df[self.cell_line_col] == cell_line)
                if mask.any():
                    # Pass the dedicated generator to the sampling method here as well
                    sampled_row = self.genomic_df[mask].sample(1, random_state=rng)
                    
                    # Only include valid data
                    valid_genomic_data.append(sampled_row[self.genomic_cols].values.flatten())
                    valid_cell_indices.append(cell_idx)
                    
                    # Convert actual values to indices
                    dose_value = sampled_row['Metadata_Dose_Level'].values[0]
                    time_value = sampled_row['Metadata_pert_time'].values[0]
                    valid_doses.append(self.dose_to_idx[dose_value])
                    valid_times.append(self.time_to_idx[time_value])
        
        # Convert to tensors (empty if no valid data)
        if valid_genomic_data:
            genomic_features = torch.FloatTensor(np.array(valid_genomic_data))  # [n_valid_conditions, n_features]
            cell_indices = torch.LongTensor(valid_cell_indices)  # [n_valid_conditions]
            doses = torch.FloatTensor(valid_doses)  # [n_valid_conditions]
            times = torch.FloatTensor(valid_times)  # [n_valid_conditions]
        else:
            # No valid genomic data - return empty tensors
            genomic_features = torch.empty(0, len(self.genomic_cols))
            cell_indices = torch.empty(0, dtype=torch.long)
            doses = torch.empty(0)
            times = torch.empty(0)
        
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
                "x_c": genomic_features,  # [n_valid_conditions, n_features] - variable length
                "cell_indices": cell_indices,  # [n_valid_conditions] - variable length
                "doses": doses,  # [n_valid_conditions] - variable length
                "times": times   # [n_valid_conditions] - variable length
            },
            "labels": torch.Tensor([-1])
        }

    def collate_fn(self, batch):
        """Collate function for DataLoader.
        
        Handles batching of variable-length genomic data by:
        1. Concatenating all valid genomic conditions across the batch
        2. Creating batch indices to track which conditions belong to which sample
        
        Args:
            batch (list): List of items from __getitem__
            
        Returns:
            dict: Contains:
                - inputs: Dict with batched x_a, x_b, genomic data, and batch indices
                - labels: Batched labels
        """
        # Stack molecular features
        adj_mats = torch.stack([item["inputs"]["x_a"][0] for item in batch])
        node_feats = torch.stack([item["inputs"]["x_a"][1] for item in batch])
        atom_vecs = torch.stack([item["inputs"]["x_a"][2] for item in batch])
        x_a_batch = [adj_mats, node_feats, atom_vecs]
        
        # Stack morphological features and labels
        x_b_batch = torch.stack([item["inputs"]["x_b"] for item in batch])
        labels_batch = torch.stack([item["labels"] for item in batch])
        
        # Handle variable-length genomic data
        all_genomic_features = []
        all_cell_indices = []
        all_doses = []
        all_times = []
        batch_indices = []  # Track which sample each condition belongs to
        
        for batch_idx, item in enumerate(batch):
            genomic_features = item["inputs"]["x_c"]
            if len(genomic_features) > 0:  # If this sample has valid genomic data
                all_genomic_features.append(genomic_features)
                all_cell_indices.append(item["inputs"]["cell_indices"])
                all_doses.append(item["inputs"]["doses"])
                all_times.append(item["inputs"]["times"])
                
                # Create batch indices for this sample's conditions
                n_conditions = len(genomic_features)
                batch_indices.extend([batch_idx] * n_conditions)
        
        # Concatenate all genomic data if any exists
        if all_genomic_features:
            x_c_batch = torch.cat(all_genomic_features, dim=0)  # [total_conditions, n_features]
            cell_indices_batch = torch.cat(all_cell_indices, dim=0)  # [total_conditions]
            doses_batch = torch.cat(all_doses, dim=0)  # [total_conditions]
            times_batch = torch.cat(all_times, dim=0)  # [total_conditions]
            batch_indices = torch.LongTensor(batch_indices)  # [total_conditions]
        else:
            # No genomic data in this batch
            x_c_batch = torch.empty(0, len(self.genomic_cols))
            cell_indices_batch = torch.empty(0, dtype=torch.long)
            doses_batch = torch.empty(0)
            times_batch = torch.empty(0)
            batch_indices = torch.empty(0, dtype=torch.long)
        
        return {
            "inputs": {
                "x_a": x_a_batch,
                "x_b": x_b_batch,
                "x_c": x_c_batch,  # [total_conditions, n_features]
                "cell_indices": cell_indices_batch,  # [total_conditions]
                "doses": doses_batch,  # [total_conditions]
                "times": times_batch,  # [total_conditions]
                "batch_indices": batch_indices  # [total_conditions] - which sample each condition belongs to
            },
            "labels": labels_batch
        }

def _deepchem_split(dataset, train_size=0.8, val_size=0.2, test_size=0.0, 
                   split_method="butina", butina_threshold=0.7, seed=42, **kwargs):
    """
    Perform train/val/test split using DeepChem splitters.
    Similar to the user's example but returns integer indices for PyTorch compatibility.
    """
    print(f"Using DeepChem {split_method} splitter")
    
    # Get SMILES and activities from dataset
    smiles = dataset.unique_smiles
    activities = dataset.df.values
    
    # Create DeepChem dataset (similar to user's example)
    dc_dataset = dc.data.NumpyDataset(X=activities, ids=smiles)
    
    # Initialize the appropriate splitter
    if split_method == "butina":
        splitter = dc.splits.ButinaSplitter(cutoff=butina_threshold)
        print(f"Using Butina splitter with cutoff={butina_threshold}")
    elif split_method == "scaffold":
        splitter = dc.splits.ScaffoldSplitter()
        print("Using Scaffold splitter")
    else:
        raise ValueError(f"Unknown DeepChem split_method: {split_method}. Choose 'butina' or 'scaffold'")
    
    # Validate split ratios
    total_ratio = train_size + val_size + test_size
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Split ratios don't sum to 1.0 (sum={total_ratio:.3f}). Normalizing...")
        train_size = train_size / total_ratio
        val_size = val_size / total_ratio
        test_size = test_size / total_ratio
    
    # Perform the split (similar to user's example)
    if test_size > 0:
        train_ds, val_ds, test_ds = splitter.train_valid_test_split(
            dc_dataset, frac_train=train_size, frac_valid=val_size, frac_test=test_size, seed=seed)
        test_smiles = set(test_ds.ids) if len(test_ds) > 0 else set()
    else:
        train_ds, val_ds = splitter.train_test_split(
            dc_dataset, frac_train=train_size, seed=seed)
        test_smiles = set()
    
    # Get SMILES for each split (similar to user's example: ncv_smiles, heldouttest_smiles)
    train_smiles = set(train_ds.ids) if len(train_ds) > 0 else set()
    val_smiles = set(val_ds.ids) if len(val_ds) > 0 else set()
    
    # Convert to integer indices for PyTorch Dataset compatibility
    train_idx = [i for i, smile in enumerate(smiles) if smile in train_smiles]
    val_idx = [i for i, smile in enumerate(smiles) if smile in val_smiles]
    test_idx = [i for i, smile in enumerate(smiles) if smile in test_smiles]
    
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"DeepChem {split_method} split:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/total_samples:.3f})")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/total_samples:.3f})")
    if test_idx:
        print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/total_samples:.3f})")
    
    return train_idx, val_idx, test_idx

def _split_data(dataset: Dataset, splits: Dict[str, str], train_size=0.8, val_size=0.2, test_size=0.0,
                split_method="random", butina_threshold=0.7, seed=42, **kwargs) -> Dict[str, Dataset]:
    if splits is None:
        unique_smiles = dataset.unique_smiles
        total_smiles = len(unique_smiles)
        
        print(f"Using {split_method} split method")
        
        if split_method == "random":
            # Validate split ratios for random split
            total_ratio = train_size + val_size + test_size
            if abs(total_ratio - 1.0) > 1e-6:
                print(f"Warning: Split ratios don't sum to 1.0 (sum={total_ratio:.3f}). Normalizing...")
                train_size = train_size / total_ratio
                val_size = val_size / total_ratio
                test_size = test_size / total_ratio
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            indices = np.random.permutation(total_smiles)
            
            train_end = int(train_size * total_smiles)
            val_end = train_end + int(val_size * total_smiles)
            
            train_idx = indices[:train_end].tolist()
            val_idx = indices[train_end:val_end].tolist()
            test_idx = indices[val_end:].tolist()
            
            print(f"Random split:")
            print(f"  Train: {len(train_idx)} samples ({len(train_idx)/total_smiles:.3f})")
            print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/total_smiles:.3f})")
            print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/total_smiles:.3f})")
            
        elif split_method in ["butina", "scaffold"]:
            # Use DeepChem splitters
            train_idx, val_idx, test_idx = _deepchem_split(
                dataset=dataset,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                split_method=split_method,
                butina_threshold=butina_threshold,
                seed=seed,
                **kwargs
            )
        elif split_method == "murcko_sgkf":
            # Murcko scaffold groups + StratifiedGroupKFold (5 folds -> 3/1/1)
            smiles_list = list(unique_smiles)
            # Labels: assume binary stored in dataset.df first column
            label_col = dataset.df.columns[0]
            y = np.array([dataset.df.loc[s][label_col] for s in smiles_list])
            # Create scaffold groups
            def _scaffold(sm):
                m = Chem.MolFromSmiles(sm)
                return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
                
            groups = np.array([_scaffold(sm) for sm in smiles_list])
            # 5-fold split with stratification by y and grouping by scaffold
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            folds = list(sgkf.split(X=np.zeros(len(y)), y=y, groups=groups))
            # Assign first 3 folds to train, 4th to val, 5th to test
            train_indices = np.concatenate([folds[i][1] for i in [0,1,2]])
            val_indices = folds[3][1]
            test_indices = folds[4][1]
            train_idx = train_indices.tolist()
            val_idx = val_indices.tolist()
            test_idx = test_indices.tolist()
            # Scaffold statistics
            total_scaffolds = len(set(groups))
            train_scaffolds = len(set(groups[train_indices]))
            val_scaffolds = len(set(groups[val_indices]))
            test_scaffolds = len(set(groups[test_indices]))
            print("Murcko+SGKF split (5 folds -> 3/1/1):")
            print(f"  Train: {len(train_idx)} samples ({len(train_idx)/total_smiles:.3f})")
            print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/total_smiles:.3f})")
            print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/total_smiles:.3f})")
            print("Scaffold counts:")
            print(f"  Total unique scaffolds: {total_scaffolds}")
            print(f"  Train scaffolds: {train_scaffolds}")
            print(f"  Val scaffolds:   {val_scaffolds}")
            print(f"  Test scaffolds:  {test_scaffolds}")
        else:
            raise ValueError(f"Unknown split_method: {split_method}. Choose 'random', 'butina', or 'scaffold'")
        
        return {
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "test": Subset(dataset, test_idx),
        }

    assert "train" in splits and "val" in splits
    split_dataset = {}
    for k, v in splits.items():
        print(f"Split {k}: {v}")
        df_split = pd.read_csv(v)
        if "SMILES" in df_split.columns:
            # Use dict lookup for O(1) index mapping instead of O(n*m) enumeration
            split_smiles_list = df_split["SMILES"].astype(str).tolist()
            # Fast dict-based lookup: O(n) instead of O(n*m)
            idx = [dataset._smiles_to_idx[s] for s in split_smiles_list if s in dataset._smiles_to_idx]
            if len(idx) < len(split_smiles_list):
                dropped = len(split_smiles_list) - len(idx)
                if dropped > 0:
                    print(f"Warning: dropped {dropped} split entries not present after filtering invalid SMILES")
        else:
            idx = df_split["index"].values

        split_dataset[k] = Subset(dataset, idx)
    return split_dataset