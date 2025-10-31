"""
HDF5-based dataset classes for precomputed molecular features and data samples.

This module provides:
1. HDF5PrecomputedCellLineDataset: Production-ready HDF5 dataset class
2. precompute_dataset_to_hdf5: Function to precompute and save data to HDF5
3. Utility functions for HDF5 dataset management
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from dataset import CellLineTripleInputGraphDatasetJUMP
from featurizer.smiles_transformation import smiles2graph


def process_smiles_batch(smiles_batch):
    """Process a batch of SMILES to get graph features."""
    results = []
    for smiles in smiles_batch:
        graph_result = smiles2graph(smiles)
        if graph_result is not None:
            adj_mat, node_feat = graph_result
            results.append((smiles, adj_mat, node_feat))
        else:
            results.append((smiles, None, None))
    return results


def process_smiles_complete(args):
    """
    Process a single SMILES with validation and feature computation in one pass.
    This eliminates the need for two separate passes.
    
    Args:
        args: Tuple containing (smiles, original_dataset, pad_length, compression, compression_opts)
    
    Returns:
        Dictionary with all computed features or None if SMILES is invalid
    """
    smiles, original_dataset, pad_length, compression, compression_opts = args
    
    # Add process ID for debugging parallel execution
    import os
    process_id = os.getpid()
    
    # Try to compute molecular features (validation + computation in one step)
    graph_result = smiles2graph(smiles)
    if graph_result is None:
        return None  # Invalid SMILES
    
    adj_mat, node_feat = graph_result
    adj_mat = torch.FloatTensor(adj_mat)
    node_feat = torch.FloatTensor(node_feat)
    atom_vec = torch.ones(len(node_feat), 1)
    
    # Pad features
    p = pad_length - len(atom_vec)
    if p >= 0:
        adj_mat = torch.nn.functional.pad(adj_mat, (0, p, 0, p), "constant", 0)
        node_feat = torch.nn.functional.pad(node_feat, (0, 0, 0, p), "constant", 0)
        atom_vec = torch.nn.functional.pad(atom_vec, (0, 0, 0, p), "constant", 0)
    
    # Get morphological data
    morph_mask = original_dataset.df[original_dataset.smiles_col] == smiles
    if morph_mask.any():
        morph_data = original_dataset.df[morph_mask][original_dataset.morph_cols].values.astype(float)
    else:
        morph_data = np.empty((0, len(original_dataset.morph_cols)))
    
    # Get genomic data
    genomic_mask = original_dataset.genomic_df[original_dataset.smiles_col] == smiles
    if genomic_mask.any():
        genomic_data = []
        cell_indices = []
        doses = []
        times = []
        
        for cell_line in original_dataset.genomic_df[genomic_mask][original_dataset.cell_line_col].unique():
            cell_idx = original_dataset.cell_line_to_idx[cell_line]
            cell_mask = genomic_mask & (original_dataset.genomic_df[original_dataset.cell_line_col] == cell_line)
            
            if cell_mask.any():
                # Get all conditions for this cell line
                cell_data = original_dataset.genomic_df[cell_mask]
                genomic_data.append(cell_data[original_dataset.genomic_cols].values)
                
                # Get corresponding metadata
                cell_indices.extend([cell_idx] * len(cell_data))
                doses.extend([original_dataset.dose_to_idx[d] for d in cell_data['Metadata_Dose_Level'].values])
                times.extend([original_dataset.time_to_idx[t] for t in cell_data['Metadata_pert_time'].values])
        
        if genomic_data:
            # Concatenate all genomic data
            all_genomic = np.vstack(genomic_data)
            genomic_result = (all_genomic, np.array(cell_indices), np.array(doses), np.array(times))
        else:
            genomic_result = (np.empty((0, len(original_dataset.genomic_cols))), 
                            np.empty(0, dtype=int), np.empty(0), np.empty(0))
    else:
        genomic_result = (np.empty((0, len(original_dataset.genomic_cols))), 
                        np.empty(0, dtype=int), np.empty(0), np.empty(0))
    
    return {
        'smiles': smiles,
        'molecular': (adj_mat.numpy(), node_feat.numpy(), atom_vec.numpy()),
        'morphological': morph_data,
        'genomic': genomic_result,
        'process_id': process_id  # For debugging parallel execution
    }


class HDF5PrecomputedCellLineDataset:
    """
    HDF5-based dataset that precomputes and caches molecular features and data samples.
    
    This class provides the same interface as CellLineTripleInputGraphDatasetJUMP
    but with precomputed features stored in HDF5 format for faster loading.
    """
    
    def __init__(self, hdf5_path: str, pad_length: int = 250, read_only: bool = True, load_in_memory: bool = False):
        """
        Initialize HDF5 dataset.
        
        Args:
            hdf5_path: Path to the HDF5 file containing precomputed data
            pad_length: Maximum length for padding molecular graphs
            read_only: Whether to open HDF5 file in read-only mode
            load_in_memory: Whether to load all data into memory for faster access
        """
        self.hdf5_path = hdf5_path
        self.pad_length = pad_length
        self.read_only = read_only
        self.load_in_memory = load_in_memory
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Open HDF5 file
        mode = 'r' if read_only else 'r+'
        self.h5file = h5py.File(hdf5_path, mode)
        
        # Load metadata
        self._load_metadata()
        
        # Load all data into memory if requested
        if load_in_memory:
            self._load_all_data_into_memory()
        
        logging.info(f"HDF5 Dataset loaded from {hdf5_path}")
        logging.info(f"  Total SMILES: {len(self.unique_smiles)}")
        logging.info(f"  Cell lines: {len(self.unique_cell_lines)}")
        logging.info(f"  Dose levels: {len(self.unique_doses)}")
        logging.info(f"  Time points: {len(self.unique_times)}")
        logging.info(f"  Morphological features: {self.morph_dim}")
        logging.info(f"  Genomic features: {self.genomic_dim}")
        logging.info(f"  In-memory loading: {load_in_memory}")
    
    def _load_metadata(self):
        """Load metadata from HDF5 file."""
        # Get SMILES list
        self.unique_smiles = [s.decode('utf-8') for s in self.h5file['smiles'][:]]
        
        # Get cell lines, doses, and times
        self.unique_cell_lines = [s.decode('utf-8') for s in self.h5file['cell_lines'][:]]
        self.unique_doses = self.h5file['doses'][:]
        self.unique_times = self.h5file['times'][:]
        
        # Create mappings (1-based indexing, 0 for padding)
        self.cell_line_to_idx = {cell: idx + 1 for idx, cell in enumerate(self.unique_cell_lines)}
        self.dose_to_idx = {dose: idx + 1 for idx, dose in enumerate(self.unique_doses)}
        self.time_to_idx = {time: idx + 1 for idx, time in enumerate(self.unique_times)}
        
        # Get feature dimensions
        self.morph_dim = self.h5file.attrs['morph_dim']
        self.genomic_dim = self.h5file.attrs['genomic_dim']
        
        # Verify pad_length matches
        stored_pad_length = self.h5file.attrs['pad_length']
        if stored_pad_length != self.pad_length:
            logging.warning(f"Stored pad_length ({stored_pad_length}) != requested pad_length ({self.pad_length})")
    
    def _load_all_data_into_memory(self):
        """Load all data into memory for faster access during training."""
        logging.info("Loading all data into memory...")
        start_time = time.time()
        
        # Initialize in-memory storage
        self.molecular_features = {}
        self.morphological_data = {}
        self.genomic_data = {}
        self.genomic_cell_indices = {}
        self.genomic_doses = {}
        self.genomic_times = {}
        
        total_memory = 0
        
        # Single loop to load all data efficiently
        logging.info(f"Loading data for {len(self.unique_smiles)} SMILES...")
        for idx in range(len(self.unique_smiles)):
            # Load molecular features
            adj_mat = torch.FloatTensor(self.h5file[f'molecular_features/{idx}/adj_mat'][:])
            node_feat = torch.FloatTensor(self.h5file[f'molecular_features/{idx}/node_feat'][:])
            atom_vec = torch.FloatTensor(self.h5file[f'molecular_features/{idx}/atom_vec'][:])
            
            self.molecular_features[idx] = {
                'adj_mat': adj_mat,
                'node_feat': node_feat,
                'atom_vec': atom_vec
            }
            
            # Load morphological data
            morph_data = self.h5file[f'morphological_data/{idx}'][:]
            self.morphological_data[idx] = morph_data
            
            # Load genomic data
            genomic_data = self.h5file[f'genomic_data/{idx}'][:]
            cell_indices = self.h5file[f'genomic_cell_indices/{idx}'][:]
            doses = self.h5file[f'genomic_doses/{idx}'][:]
            times = self.h5file[f'genomic_times/{idx}'][:]
            
            self.genomic_data[idx] = genomic_data
            self.genomic_cell_indices[idx] = cell_indices
            self.genomic_doses[idx] = doses
            self.genomic_times[idx] = times
            
            # Calculate memory usage incrementally
            total_memory += adj_mat.numel() * 4  # float32
            total_memory += node_feat.numel() * 4
            total_memory += atom_vec.numel() * 4
            total_memory += morph_data.nbytes
            total_memory += genomic_data.nbytes
            total_memory += cell_indices.nbytes
            total_memory += doses.nbytes
            total_memory += times.nbytes
            
            # Progress logging for large datasets
            if (idx + 1) % 1000 == 0:
                logging.info(f"Loaded {idx + 1}/{len(self.unique_smiles)} SMILES...")
        
        # Close HDF5 file since we have everything in memory
        self.h5file.close()
        self.h5file = None
        
        load_time = time.time() - start_time
        memory_gb = total_memory / (1024**3)
        
        logging.info(f"Data loaded into memory in {load_time:.2f} seconds")
        logging.info(f"Estimated memory usage: {memory_gb:.2f} GB")
        logging.info(f"Loading rate: {len(self.unique_smiles) / load_time:.1f} SMILES/second")
    
    def __len__(self):
        return len(self.unique_smiles)
    
    def __getitem__(self, index):
        """Get item with precomputed features from HDF5 or memory."""
        # Create a new, isolated random number generator for this specific item fetch.
        # ensuring samples are different each time (same as original approach).
        rng = np.random.default_rng()
        
        smiles = self.unique_smiles[index]
        
        # Get precomputed molecular features (from memory or HDF5)
        if self.load_in_memory:
            adj_mat = self.molecular_features[index]['adj_mat']
            node_feat = self.molecular_features[index]['node_feat']
            atom_vec = self.molecular_features[index]['atom_vec']
        else:
            adj_mat = torch.FloatTensor(self.h5file[f'molecular_features/{index}/adj_mat'][:])
            node_feat = torch.FloatTensor(self.h5file[f'molecular_features/{index}/node_feat'][:])
            atom_vec = torch.FloatTensor(self.h5file[f'molecular_features/{index}/atom_vec'][:])
        
        cmpd_feat = [adj_mat, node_feat, atom_vec]
        
        # Get morphological features (from memory or HDF5)
        if self.load_in_memory:
            morph_data = self.morphological_data[index]
        else:
            morph_data = self.h5file[f'morphological_data/{index}'][:]
        
        if len(morph_data) > 0:
            # Random sampling using dedicated RNG (same as original approach)
            sample_idx = rng.integers(0, len(morph_data))
            morph_feat = torch.FloatTensor(morph_data[sample_idx])
        else:
            morph_feat = torch.FloatTensor(-1 * np.ones(self.morph_dim))
        
        # Get genomic features (from memory or HDF5)
        if self.load_in_memory:
            genomic_data = self.genomic_data[index]
            cell_indices_data = self.genomic_cell_indices[index]
            doses_data = self.genomic_doses[index]
            times_data = self.genomic_times[index]
        else:
            genomic_data = self.h5file[f'genomic_data/{index}'][:]
            cell_indices_data = self.h5file[f'genomic_cell_indices/{index}'][:]
            doses_data = self.h5file[f'genomic_doses/{index}'][:]
            times_data = self.h5file[f'genomic_times/{index}'][:]
        
        if len(genomic_data) > 0:
            # Sample one condition per unique cell line (same logic as original)
            unique_cells = np.unique(cell_indices_data)
            valid_genomic_data = []
            valid_cell_indices = []
            valid_doses = []
            valid_times = []
            
            for cell_idx in unique_cells:
                cell_mask = cell_indices_data == cell_idx
                if np.any(cell_mask):
                    # Random sampling from this cell line's conditions using dedicated RNG
                    cell_conditions = np.where(cell_mask)[0]
                    sample_idx = rng.choice(cell_conditions)
                    
                    valid_genomic_data.append(genomic_data[sample_idx])
                    valid_cell_indices.append(cell_indices_data[sample_idx])
                    valid_doses.append(doses_data[sample_idx])
                    valid_times.append(times_data[sample_idx])
            
            if valid_genomic_data:
                genomic_features = torch.FloatTensor(np.array(valid_genomic_data))
                cell_indices = torch.LongTensor(valid_cell_indices)
                doses = torch.FloatTensor(valid_doses)
                times = torch.FloatTensor(valid_times)
            else:
                genomic_features = torch.empty(0, self.genomic_dim)
                cell_indices = torch.empty(0, dtype=torch.long)
                doses = torch.empty(0)
                times = torch.empty(0)
        else:
            genomic_features = torch.empty(0, self.genomic_dim)
            cell_indices = torch.empty(0, dtype=torch.long)
            doses = torch.empty(0)
            times = torch.empty(0)
        
        return {
            "inputs": {
                "x_a": cmpd_feat,
                "x_b": morph_feat,
                "x_c": genomic_features,
                "cell_indices": cell_indices,
                "doses": doses,
                "times": times
            },
            "labels": torch.Tensor([-1])
        }
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        This is identical to the original collate_fn but works with HDF5 data.
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
            x_c_batch = torch.empty(0, self.genomic_dim)
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
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'h5file') and self.h5file is not None:
            self.h5file.close()
    
    def __del__(self):
        """Ensure HDF5 file is closed when object is destroyed."""
        self.close()


def precompute_dataset_to_hdf5(
    data_path: str,
    genomic_data_path: str,
    invalid_smiles_report: Optional[str],
    output_hdf5_path: str,
    pad_length: int = 250,
    max_smiles: Optional[int] = None,
    compression: str = 'gzip',
    compression_opts: int = 6,
    n_processes: Optional[int] = None
) -> Dict[str, int]:
    """
    Precompute molecular features and data samples, save to HDF5.
    
    Optimized version that uses single-pass parallel processing to eliminate
    redundant computation while maintaining identical output format.
    
    Args:
        data_path: Path to morphological data
        genomic_data_path: Path to genomic data
        invalid_smiles_report: Path to invalid SMILES report
        output_hdf5_path: Path to save HDF5 file
        pad_length: Maximum length for padding molecular graphs
        max_smiles: Maximum number of SMILES to process (for testing)
        compression: HDF5 compression algorithm ('gzip', 'lzf', 'szip', None)
        compression_opts: Compression level (for gzip: 0-9)
        n_processes: Number of processes for parallel processing (default: all available CPUs)
    
    Returns:
        Dictionary with statistics about the precomputed data
    """
    logging.info("Starting optimized HDF5 precomputation...")
    
    # Create original dataset to get the data structure
    original_dataset = CellLineTripleInputGraphDatasetJUMP(
        data_path=data_path,
        genomic_data_path=genomic_data_path,
        pad_length=pad_length,
        invalid_smiles_report=invalid_smiles_report
    )
    
    # Limit SMILES for testing if specified
    if max_smiles is not None:
        original_dataset.unique_smiles = original_dataset.unique_smiles[:max_smiles]
        logging.info(f"Limited to {max_smiles} SMILES for testing")
    
    logging.info(f"Processing {len(original_dataset.unique_smiles)} SMILES...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    
    # Set up parallel processing
    if n_processes is None:
        n_processes = cpu_count()
    logging.info(f"Using {n_processes} processes for parallel processing")
    
    # Prepare arguments for parallel processing
    process_args = [
        (smiles, original_dataset, pad_length, compression, compression_opts) 
        for smiles in original_dataset.unique_smiles
    ]
    
    # Process all SMILES in parallel (validation + feature computation in one pass)
    logging.info("Processing SMILES in parallel (validation + feature computation)...")
    logging.info(f"Using {n_processes} processes to process {len(process_args)} SMILES...")
    
    with Pool(processes=n_processes) as pool:
        # Use map for true parallel processing (no misleading progress bar)
        logging.info("Starting parallel processing...")
        start_time = time.time()
        
        all_results = pool.map(process_smiles_complete, process_args)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Filter out invalid SMILES
        results = [r for r in all_results if r is not None]
        
        logging.info(f"Parallel processing completed in {processing_time:.2f} seconds")
        logging.info(f"Processing rate: {len(process_args) / processing_time:.2f} SMILES/second")
    
    valid_smiles = [r['smiles'] for r in results]
    
    # Log process usage for debugging
    process_ids = set(r['process_id'] for r in results)
    logging.info(f"Valid SMILES: {len(valid_smiles)} out of {len(original_dataset.unique_smiles)}")
    logging.info(f"Used {len(process_ids)} different processes: {sorted(process_ids)}")
    
    # Create HDF5 file and write results
    with h5py.File(output_hdf5_path, 'w') as h5file:
        # Store metadata (only valid SMILES)
        h5file.create_dataset('smiles', data=[s.encode('utf-8') for s in valid_smiles])
        h5file.create_dataset('cell_lines', data=[s.encode('utf-8') for s in original_dataset.unique_cell_lines])
        h5file.create_dataset('doses', data=original_dataset.unique_doses)
        h5file.create_dataset('times', data=original_dataset.unique_times)
        
        # Store feature dimensions and parameters
        h5file.attrs['morph_dim'] = len(original_dataset.morph_cols)
        h5file.attrs['genomic_dim'] = len(original_dataset.genomic_cols)
        h5file.attrs['pad_length'] = pad_length
        h5file.attrs['compression'] = compression
        h5file.attrs['compression_opts'] = compression_opts
        
        # Create groups for different data types
        mol_group = h5file.create_group('molecular_features')
        morph_group = h5file.create_group('morphological_data')
        genomic_group = h5file.create_group('genomic_data')
        cell_indices_group = h5file.create_group('genomic_cell_indices')
        doses_group = h5file.create_group('genomic_doses')
        times_group = h5file.create_group('genomic_times')
        
        # Statistics
        stats = {
            'total_smiles': len(original_dataset.unique_smiles),
            'smiles_with_morph': 0,
            'smiles_with_genomic': 0,
            'total_morph_samples': 0,
            'total_genomic_samples': 0
        }
        
        # Write results to HDF5 (maintaining exact same structure as original)
        logging.info("Writing results to HDF5...")
        for idx, result in enumerate(tqdm(results, desc="Writing to HDF5")):
            # Write molecular features with compression (use original idx)
            mol_subgroup = mol_group.create_group(str(idx))
            mol_subgroup.create_dataset('adj_mat', data=result['molecular'][0], 
                                      compression=compression, compression_opts=compression_opts)
            mol_subgroup.create_dataset('node_feat', data=result['molecular'][1],
                                      compression=compression, compression_opts=compression_opts)
            mol_subgroup.create_dataset('atom_vec', data=result['molecular'][2],
                                      compression=compression, compression_opts=compression_opts)
            
            # Write morphological data
            morph_data = result['morphological']
            morph_group.create_dataset(str(idx), data=morph_data,
                                     compression=compression, compression_opts=compression_opts)
            if len(morph_data) > 0:
                stats['smiles_with_morph'] += 1
                stats['total_morph_samples'] += len(morph_data)
            
            # Write genomic data
            genomic_data, cell_indices, doses, times = result['genomic']
            genomic_group.create_dataset(str(idx), data=genomic_data,
                                       compression=compression, compression_opts=compression_opts)
            cell_indices_group.create_dataset(str(idx), data=cell_indices,
                                            compression=compression, compression_opts=compression_opts)
            doses_group.create_dataset(str(idx), data=doses,
                                     compression=compression, compression_opts=compression_opts)
            times_group.create_dataset(str(idx), data=times,
                                     compression=compression, compression_opts=compression_opts)
            
            if len(genomic_data) > 0:
                stats['smiles_with_genomic'] += 1
                stats['total_genomic_samples'] += len(genomic_data)
    
    # Get file size
    file_size = os.path.getsize(output_hdf5_path) / (1024**3)  # GB
    stats['file_size_gb'] = file_size
    
    logging.info(f"Optimized precomputation completed. HDF5 file saved to: {output_hdf5_path}")
    logging.info(f"HDF5 file size: {file_size:.2f} GB")
    logging.info(f"Statistics: {stats}")
    
    return stats


def get_hdf5_dataset_info(hdf5_path: str) -> Dict[str, any]:
    """
    Get information about an HDF5 dataset file.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary with dataset information
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as h5file:
        info = {
            'file_path': hdf5_path,
            'file_size_gb': os.path.getsize(hdf5_path) / (1024**3),
            'num_smiles': len(h5file['smiles']),
            'num_cell_lines': len(h5file['cell_lines']),
            'num_doses': len(h5file['doses']),
            'num_times': len(h5file['times']),
            'morph_dim': h5file.attrs['morph_dim'],
            'genomic_dim': h5file.attrs['genomic_dim'],
            'pad_length': h5file.attrs['pad_length'],
            'compression': h5file.attrs.get('compression', 'unknown'),
            'compression_opts': h5file.attrs.get('compression_opts', 'unknown')
        }
    
    return info


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    data_path = "/scratch/project_462000766/datasets/jump_preprocessed/dummy_data/cell_fetures_with_smiles_2000.parquet"
    genomic_data_path = "/scratch/project_462000766/datasets/LINCS_preprocessed/landmark_cmp_data_min1000compounds_all_measurements_test.parquet"
    invalid_smiles_report = "/scratch/project_462000766/Multi_Modal_Contrastive/failed_molecules_report.csv"
    output_hdf5_path = "/scratch/project_462000766/Multi_Modal_Contrastive/test_precomputed_data.h5"
    
    # Precompute dataset
    stats = precompute_dataset_to_hdf5(
        data_path=data_path,
        genomic_data_path=genomic_data_path,
        invalid_smiles_report=invalid_smiles_report,
        output_hdf5_path=output_hdf5_path,
        pad_length=250,
        max_smiles=50  # Small test
    )
    
    # Get dataset info
    info = get_hdf5_dataset_info(output_hdf5_path)
    print(f"Dataset info: {info}")
    
    # Test loading
    dataset = HDF5PrecomputedCellLineDataset(output_hdf5_path)
    sample = dataset[0]
    print(f"Sample keys: {sample['inputs'].keys()}")
    print(f"Molecular features shape: {[f.shape for f in sample['inputs']['x_a']]}")
    print(f"Morphological features shape: {sample['inputs']['x_b'].shape}")
    print(f"Genomic features shape: {sample['inputs']['x_c'].shape}")
    
    dataset.close()
