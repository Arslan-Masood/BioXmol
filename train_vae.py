#!/usr/bin/env python3
"""
Training script for JUMP Cell Painting VAE model.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
import yaml
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import JUMPVAE, create_vae_model

# Import existing functions from mocop
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocop'))
from training import build_dataloaders
from dataset import _split_data


def save_final_metrics(trainer, experiment_name: str, log_dir: str, hyperparameters: dict):
    """Save final validation metrics and hyperparameters to JSON file."""
    results_dir = os.path.join(log_dir, experiment_name, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get final metrics from wandb logger if available
    final_metrics = {}
    
    # Try to get metrics from wandb logger first
    if hasattr(trainer, 'logger') and trainer.logger is not None:
        # Check if it's a wandb logger or a list containing wandb logger
        wandb_logger = None
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'summary'):
            wandb_logger = trainer.logger
        elif isinstance(trainer.logger, list):
            # Find wandb logger in the list
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'summary'):
                    wandb_logger = logger
                    break
        
        if wandb_logger is not None:
            # Extract metrics from wandb summary
            summary = dict(wandb_logger.experiment.summary)
            
            # Extract validation metrics
            val_metrics = {k.replace('val/', ''): float(v) for k, v in summary.items() 
                          if k.startswith('val/')}
            if val_metrics:
                final_metrics['final_validation'] = val_metrics
            
            # Extract test metrics
            test_metrics = {k.replace('test/', ''): float(v) for k, v in summary.items() 
                           if k.startswith('test/')}
            if test_metrics:
                final_metrics['final_test'] = test_metrics
            
            # Extract train metrics for reference
            train_metrics = {k.replace('train/', ''): float(v) for k, v in summary.items() 
                            if k.startswith('train/')}
            if train_metrics:
                final_metrics['final_train'] = train_metrics
            
            print(f"Extracted metrics from wandb: {list(final_metrics.keys())}")
    
    # Fallback to trainer.callback_metrics if wandb not available
    if not final_metrics and trainer.callback_metrics:
        # Extract validation metrics (final epoch values)
        val_metrics = {k.replace('val/', ''): float(v) for k, v in trainer.callback_metrics.items() 
                      if k.startswith('val/')}
        if val_metrics:
            final_metrics['final_validation'] = val_metrics
        
        # Extract test metrics if available
        test_metrics = {k.replace('test/', ''): float(v) for k, v in trainer.callback_metrics.items() 
                       if k.startswith('test/')}
        if test_metrics:
            final_metrics['final_test'] = test_metrics
        
        print(f"Extracted metrics from callback_metrics: {list(final_metrics.keys())}")
    
    # Final fallback to logged_metrics
    if not final_metrics and trainer.logged_metrics:
        val_metrics = {k.replace('val/', ''): float(v) for k, v in trainer.logged_metrics.items() 
                      if k.startswith('val/')}
        if val_metrics:
            final_metrics['final_validation'] = val_metrics
        
        print(f"Extracted metrics from logged_metrics: {list(final_metrics.keys())}")
    
    # Create results JSON
    results = {
        'hyperparameters': hyperparameters,
        'final_metrics': final_metrics,
        'total_epochs': trainer.current_epoch + 1,
        'experiment_name': experiment_name,
        'saved_at': str(pd.Timestamp.now())
    }
    
    # Save to JSON file
    results_path = os.path.join(results_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Final results saved to: {results_path}")
    print(f"Metrics saved: {list(final_metrics.keys()) if final_metrics else 'No metrics found'}")


@dataclass
class Config:
    """Configuration class that handles all parameter extraction and validation."""
    
    # Raw config dictionary
    _config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required parameters after initialization."""
        # Validate required data path
        if not self._config.get('data', {}).get('data_path'):
            raise ValueError("data_path must be specified in config file under 'data' section")
    
    @property
    def data(self) -> Dict[str, Any]:
        """Data configuration - no defaults, everything from YAML."""
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Model configuration - no defaults, everything from YAML."""
        return self._config.get('model', {})
    
    @property
    def optimization(self) -> Dict[str, Any]:
        """Optimization configuration - no defaults, everything from YAML."""
        return self._config.get('optimization', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Training configuration - no defaults, everything from YAML."""
        return self._config.get('training', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Logging configuration - no defaults, everything from YAML."""
        return self._config.get('logging', {})
    
    @property
    def splits(self) -> Optional[Dict[str, str]]:
        """Get splits dictionary, returns None if no splits provided."""
        splits = {}
        data_config = self.data
        
        for split_name in ['train_split', 'val_split', 'test_split']:
            split_path = data_config.get(split_name)
            if split_path:
                splits[split_name.replace('_split', '')] = split_path
        
        return splits if splits else None
    
    @property
    def modality(self) -> str:
        """Get the modality to train on ('jump_cp' or 'genomic')."""
        return self.data.get('modality', 'jump_cp')  # Default to JUMP Cell Painting
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)
        print(f"Modality: {self.modality.upper()}")
        if self.modality == 'genomic':
            genomic_path = self.data.get('genomic_data_path', self.data['data_path'])
            print(f"Genomic data path: {genomic_path}")
            if self.data.get('target_cell_line'):
                print(f"Target cell line: {self.data['target_cell_line']}")
        else:
            print(f"Cell painting data path: {self.data['data_path']}")
        print(f"Model: {self.model['model_type'].upper()} with {self.model['architecture']} architecture")
        print(f"Latent dimension: {self.model['latent_dim']}")
        print(f"Batch size: {self.data['batch_size']}")
        print(f"Max epochs: {self.training['max_epochs']}")
        print(f"Save model weights: {self.training.get('save_model_weights', False)}")
        print(f"Experiment: {self.logging['experiment_name']}")
        if self.splits:
            print(f"Using predefined splits: {list(self.splits.keys())}")
        else:
            print(f"Using random splits")
        print("="*60)


class JUMPCellPaintingDataset(Dataset):
    """Dataset class for JUMP Cell Painting data that returns molecule-wise samples."""
    
    def __init__(self, config: Config):
        """Initialize dataset from config."""
        data_config = config.data
        
        self.data_path = data_config['data_path']
        self.normalize = data_config['normalize']
        self.random_seed = config.training['seed']
        
        # Molecule filtering (exclude controls/specific molecules)
        self.exclude_molecules = data_config.get('exclude_molecules', [])
        self.molecule_id_column = data_config.get('molecule_id_column', 'Metadata_InChIKey')
        
        # Load and process data
        self._load_data()
        
        # Set up for molecule-wise sampling
        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col in self.df.columns:
            self.unique_smiles = self.df[self.smiles_col].dropna().unique().tolist()
        else:
            raise ValueError("SMILES column not found in the dataset")
        
        # Pre-compute molecule indices for faster sampling
        self._precompute_indices()
        
        print(f"Dataset created with {len(self.unique_smiles)} unique molecules")
    
    def _load_data(self):
        """Load and preprocess the data."""
        print(f"Loading data from {self.data_path}")
        
        # Load data
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Data file must be .parquet or .csv")
        
        print(f"Loaded dataframe with shape: {self.df.shape}")
        
        # Filter out excluded molecules if specified
        if self.exclude_molecules:
            self._filter_molecules()
        
        # Get feature columns (non-metadata)
        self.feature_cols = [c for c in self.df.columns if 'Metadata' not in c]
        
        print(f"Feature columns: {len(self.feature_cols)}")
        self.feature_dim = len(self.feature_cols)
        
        # Extract features as numpy array for fast access
        self.features_array = self.df[self.feature_cols].values.astype(np.float32)
        
        # Handle NaN values once during initialization
        self.features_array = np.where(np.isnan(self.features_array), 0.0, self.features_array)
        
        # Store normalization parameters and pre-normalize if enabled
        if self.normalize:
            self.feature_mean = np.mean(self.features_array, axis=0, keepdims=True)
            self.feature_std = np.std(self.features_array, axis=0, keepdims=True)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
            
            # Pre-normalize all features for faster access
            self.features_array = (self.features_array - self.feature_mean) / self.feature_std
            print("Computed normalization statistics and pre-normalized all features")
        else:
            print("Features extracted as numpy array (no normalization)")
    
    def _filter_molecules(self):
        """Filter out excluded molecules from the dataset."""
        original_shape = self.df.shape
        
        # Check if the specified molecular ID column exists
        if self.molecule_id_column not in self.df.columns:
            available_cols = [col for col in self.df.columns if 'Metadata' in col]
            raise ValueError(
                f"Molecular ID column '{self.molecule_id_column}' not found in dataset. "
                f"Available metadata columns: {available_cols}"
            )
        
        before_count = len(self.df)
        
        # Filter out molecules in exclude list
        excluded_mask = self.df[self.molecule_id_column].isin(self.exclude_molecules)
        excluded_count = excluded_mask.sum()
        
        if excluded_count > 0:
            self.df = self.df[~excluded_mask].reset_index(drop=True)
            after_count = len(self.df)
            
            unique_excluded = len(set(self.exclude_molecules) & set(self.df[self.molecule_id_column].unique()))
            
            print(f"Filtered using {self.molecule_id_column}:")
            print(f"  - Excluded {excluded_count} rows containing {len(self.exclude_molecules)} control molecules")
            print(f"  - Dataset size: {before_count} → {after_count} rows")
        else:
            print(f"No molecules found to exclude using {self.molecule_id_column}")
        
        if self.df.shape[0] < original_shape[0] * 0.5:
            print(f"⚠️  WARNING: Filtering removed {((original_shape[0] - self.df.shape[0]) / original_shape[0] * 100):.1f}% of data")
    
    def _precompute_indices(self):
        """Pre-compute indices for each molecule to eliminate pandas operations in __getitem__."""
        print("Pre-computing molecule indices...")
        self.smiles_to_indices = {}
        for smiles in self.unique_smiles:
            indices = self.df[self.df[self.smiles_col] == smiles].index.tolist()
            self.smiles_to_indices[smiles] = indices
        
        # Initialize a single random number generator
        self.rng = np.random.default_rng(self.random_seed)
        print(f"Pre-computed indices for {len(self.unique_smiles)} molecules")
    
    def __len__(self):
        return len(self.unique_smiles)
    
    def __getitem__(self, index):
        """Get a molecule-wise sample - fully optimized with no pandas operations."""
        smiles = self.unique_smiles[index]
        
        # Get pre-computed row indices for this SMILES (no pandas filtering!)
        row_indices = self.smiles_to_indices[smiles]
        
        # Use your suggested approach for proper randomness with seed_everything()
        random_state = self.rng.integers(0, 2**32-1)
        temp_rng = np.random.default_rng(random_state)
        selected_row_idx = temp_rng.choice(row_indices)
        
        # Get features directly from pre-processed numpy array (no pandas!)
        features = self.features_array[selected_row_idx].copy()
        
        return torch.from_numpy(features)


class GenomicDataset(Dataset):
    """Dataset class for genomic (LINCS L1000) data that returns molecule-wise samples."""
    
    def __init__(self, config: Config):
        """Initialize genomic dataset from config."""
        data_config = config.data
        
        # Use genomic_data_path if specified, otherwise fallback to data_path
        self.data_path = data_config.get('genomic_data_path', data_config['data_path'])
        self.normalize = data_config['normalize']
        self.random_seed = config.training['seed']
        
        # Molecule filtering (exclude controls/specific molecules)
        self.exclude_molecules = data_config.get('exclude_molecules', [])
        self.molecule_id_column = data_config.get('molecule_id_column', 'Metadata_InChIKey')
        
        # Cell line filtering options
        self.target_cell_line = data_config.get('target_cell_line', None)  # Filter to specific cell line
        self.cell_line_col = "Metadata_cell_iname"
        
        # Load and process data
        self._load_data()
        
        # Set up for molecule-wise sampling
        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col in self.df.columns:
            self.unique_smiles = self.df[self.smiles_col].dropna().unique().tolist()
        else:
            raise ValueError("SMILES column not found in the genomic dataset")
        
        # Pre-compute molecule indices for faster sampling
        self._precompute_indices()
        
        print(f"Genomic dataset created with {len(self.unique_smiles)} unique molecules")
        if self.target_cell_line:
            print(f"Filtered to cell line: {self.target_cell_line}")
    
    def _load_data(self):
        """Load and preprocess the genomic data."""
        print(f"Loading genomic data from {self.data_path}")
        
        # Load data (genomic data is typically in parquet format)
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Genomic data file must be .parquet or .csv")
        
        print(f"Loaded genomic dataframe with shape: {self.df.shape}")
        
        # Filter to specific cell line if specified
        if self.target_cell_line:
            if self.cell_line_col in self.df.columns:
                original_shape = self.df.shape[0]
                self.df = self.df[self.df[self.cell_line_col] == self.target_cell_line].reset_index(drop=True)
                print(f"Filtered to {self.target_cell_line}: {original_shape} → {self.df.shape[0]} rows")
            else:
                print(f"Warning: Cell line column '{self.cell_line_col}' not found, skipping cell line filtering")
        
        # Filter out excluded molecules if specified
        if self.exclude_molecules:
            self._filter_molecules()
        
        # Get feature columns (non-metadata) - genomic features
        self.feature_cols = [c for c in self.df.columns if 'Metadata' not in c]
        
        print(f"Genomic feature columns: {len(self.feature_cols)}")
        self.feature_dim = len(self.feature_cols)
        
        # Extract features as numpy array for fast access
        self.features_array = self.df[self.feature_cols].values.astype(np.float32)
        
        # Handle NaN values once during initialization
        self.features_array = np.where(np.isnan(self.features_array), 0.0, self.features_array)
        
        # Store normalization parameters and pre-normalize if enabled
        if self.normalize:
            self.feature_mean = np.mean(self.features_array, axis=0, keepdims=True)
            self.feature_std = np.std(self.features_array, axis=0, keepdims=True)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
            
            # Pre-normalize all features for faster access
            self.features_array = (self.features_array - self.feature_mean) / self.feature_std
            print("Computed genomic normalization statistics and pre-normalized all features")
        else:
            print("Genomic features extracted as numpy array (no normalization)")
    
    def _filter_molecules(self):
        """Filter out excluded molecules from the genomic dataset."""
        original_shape = self.df.shape
        
        # Check if the specified molecular ID column exists
        if self.molecule_id_column not in self.df.columns:
            available_cols = [col for col in self.df.columns if 'Metadata' in col]
            raise ValueError(
                f"Molecular ID column '{self.molecule_id_column}' not found in genomic dataset. "
                f"Available metadata columns: {available_cols}"
            )
        
        before_count = len(self.df)
        
        # Filter out molecules in exclude list
        excluded_mask = self.df[self.molecule_id_column].isin(self.exclude_molecules)
        excluded_count = excluded_mask.sum()
        
        if excluded_count > 0:
            self.df = self.df[~excluded_mask].reset_index(drop=True)
            after_count = len(self.df)
            
            print(f"Filtered genomic data using {self.molecule_id_column}:")
            print(f"  - Excluded {excluded_count} rows containing {len(self.exclude_molecules)} control molecules")
            print(f"  - Dataset size: {before_count} → {after_count} rows")
        else:
            print(f"No molecules found to exclude from genomic data using {self.molecule_id_column}")
        
        if self.df.shape[0] < original_shape[0] * 0.5:
            print(f"⚠️  WARNING: Filtering removed {((original_shape[0] - self.df.shape[0]) / original_shape[0] * 100):.1f}% of genomic data")
    
    def _precompute_indices(self):
        """Pre-compute indices for each molecule to eliminate pandas operations in __getitem__."""
        print("Pre-computing genomic molecule indices...")
        self.smiles_to_indices = {}
        for smiles in self.unique_smiles:
            indices = self.df[self.df[self.smiles_col] == smiles].index.tolist()
            self.smiles_to_indices[smiles] = indices
        
        # Initialize a single random number generator
        self.rng = np.random.default_rng(self.random_seed)
        print(f"Pre-computed genomic indices for {len(self.unique_smiles)} molecules")
    
    def __len__(self):
        return len(self.unique_smiles)
    
    def __getitem__(self, index):
        """Get a molecule-wise sample from genomic data - fully optimized with no pandas operations."""
        smiles = self.unique_smiles[index]
        
        # Get pre-computed row indices for this SMILES (no pandas filtering!)
        row_indices = self.smiles_to_indices[smiles]
        
        # Use consistent approach for proper randomness with seed_everything()
        random_state = self.rng.integers(0, 2**32-1)
        temp_rng = np.random.default_rng(random_state)
        selected_row_idx = temp_rng.choice(row_indices)
        
        # Get features directly from pre-processed numpy array (no pandas!)
        features = self.features_array[selected_row_idx].copy()
        
        return torch.from_numpy(features)


class ConditionalGenomicDataset(Dataset):
    """Dataset class for conditional genomic (LINCS L1000) data with cell line, dose, and time embeddings."""
    
    def __init__(self, config: Config):
        """Initialize conditional genomic dataset from config."""
        data_config = config.data
        
        # Use genomic_data_path if specified, otherwise fallback to data_path
        self.data_path = data_config.get('genomic_data_path', data_config['data_path'])
        self.normalize = data_config['normalize']
        self.random_seed = config.training['seed']
        
        # Molecule filtering (exclude controls/specific molecules)
        self.exclude_molecules = data_config.get('exclude_molecules', [])
        self.molecule_id_column = data_config.get('molecule_id_column', 'Metadata_InChIKey')
        
        # Cell line filtering options
        self.target_cell_line = data_config.get('target_cell_line', None)  # Filter to specific cell line
        self.cell_line_col = "Metadata_cell_iname"
        
        # Load and process data
        self._load_data()
        
        # Set up for molecule-wise sampling
        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col in self.df.columns:
            self.unique_smiles = self.df[self.smiles_col].dropna().unique().tolist()
        else:
            raise ValueError("SMILES column not found in the genomic dataset")
        
        # Set up conditional embeddings (similar to CellLineTripleInputEncoder)
        self._setup_conditional_mappings()
        
        # Pre-compute molecule indices for faster sampling
        self._precompute_indices()
        
        print(f"Conditional genomic dataset created with {len(self.unique_smiles)} unique molecules")
        if self.target_cell_line:
            print(f"Filtered to cell line: {self.target_cell_line}")
        print(f"Conditional embedding dimensions: cell={len(self.cell_line_to_idx)}, dose={len(self.dose_to_idx)}, time={len(self.time_to_idx)}")
    
    def _load_data(self):
        """Load and preprocess the conditional genomic data."""
        print(f"Loading conditional genomic data from {self.data_path}")
        
        # Load data (genomic data is typically in parquet format)
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Genomic data file must be .parquet or .csv")
        
        print(f"Loaded conditional genomic dataframe with shape: {self.df.shape}")
        
        # Filter to specific cell line if specified
        if self.target_cell_line:
            if self.cell_line_col in self.df.columns:
                original_shape = self.df.shape[0]
                self.df = self.df[self.df[self.cell_line_col] == self.target_cell_line].reset_index(drop=True)
                print(f"Filtered to {self.target_cell_line}: {original_shape} → {self.df.shape[0]} rows")
            else:
                print(f"Warning: Cell line column '{self.cell_line_col}' not found, skipping cell line filtering")
        
        # Filter out excluded molecules if specified
        if self.exclude_molecules:
            self._filter_molecules()
        
        # Get feature columns (non-metadata) - genomic features
        self.feature_cols = [c for c in self.df.columns if 'Metadata' not in c]
        
        print(f"Genomic feature columns: {len(self.feature_cols)}")
        self.feature_dim = len(self.feature_cols)
        
        # Extract features as numpy array for fast access
        self.features_array = self.df[self.feature_cols].values.astype(np.float32)
        
        # Handle NaN values once during initialization
        self.features_array = np.where(np.isnan(self.features_array), 0.0, self.features_array)
        
        # Store normalization parameters and pre-normalize if enabled
        if self.normalize:
            self.feature_mean = np.mean(self.features_array, axis=0, keepdims=True)
            self.feature_std = np.std(self.features_array, axis=0, keepdims=True)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
            
            # Pre-normalize all features for faster access
            self.features_array = (self.features_array - self.feature_mean) / self.feature_std
            print("Computed conditional genomic normalization statistics and pre-normalized all features")
        else:
            print("Conditional genomic features extracted as numpy array (no normalization)")
    
    def _filter_molecules(self):
        """Filter out excluded molecules from the conditional genomic dataset."""
        original_shape = self.df.shape
        
        # Check if the specified molecular ID column exists
        if self.molecule_id_column not in self.df.columns:
            available_cols = [col for col in self.df.columns if 'Metadata' in col]
            raise ValueError(
                f"Molecular ID column '{self.molecule_id_column}' not found in conditional genomic dataset. "
                f"Available metadata columns: {available_cols}"
            )
        
        before_count = len(self.df)
        
        # Filter out molecules in exclude list
        excluded_mask = self.df[self.molecule_id_column].isin(self.exclude_molecules)
        excluded_count = excluded_mask.sum()
        
        if excluded_count > 0:
            self.df = self.df[~excluded_mask].reset_index(drop=True)
            after_count = len(self.df)
            
            print(f"Filtered conditional genomic data using {self.molecule_id_column}:")
            print(f"  - Excluded {excluded_count} rows containing {len(self.exclude_molecules)} control molecules")
            print(f"  - Dataset size: {before_count} → {after_count} rows")
        else:
            print(f"No molecules found to exclude from conditional genomic data using {self.molecule_id_column}")
        
        if self.df.shape[0] < original_shape[0] * 0.5:
            print(f"⚠️  WARNING: Filtering removed {((original_shape[0] - self.df.shape[0]) / original_shape[0] * 100):.1f}% of conditional genomic data")
    
    def _setup_conditional_mappings(self):
        """Set up mappings for cell lines, doses, and time points."""
        # Cell line mapping
        unique_cell_lines = sorted(self.df[self.cell_line_col].unique())
        self.cell_line_to_idx = {cell: idx + 1 for idx, cell in enumerate(unique_cell_lines)}  # Start from 1, 0 is padding
        
        # Dose mapping
        unique_doses = sorted(self.df['Metadata_Dose_Level'].unique())
        self.dose_to_idx = {dose: idx + 1 for idx, dose in enumerate(unique_doses)}  # Start from 1, 0 is padding
        
        # Time mapping
        unique_times = sorted(self.df['Metadata_pert_time'].unique())
        self.time_to_idx = {time: idx + 1 for idx, time in enumerate(unique_times)}  # Start from 1, 0 is padding
        
        print(f"Created conditional mappings:")
        print(f"  - Cell lines: {len(unique_cell_lines)} -> indices 1-{len(unique_cell_lines)}")
        print(f"  - Doses: {len(unique_doses)} -> indices 1-{len(unique_doses)}")
        print(f"  - Times: {len(unique_times)} -> indices 1-{len(unique_times)}")
    
    def _precompute_indices(self):
        """Pre-compute indices for each molecule to eliminate pandas operations in __getitem__."""
        print("Pre-computing conditional genomic molecule indices...")
        self.smiles_to_indices = {}
        for smiles in self.unique_smiles:
            indices = self.df[self.df[self.smiles_col] == smiles].index.tolist()
            self.smiles_to_indices[smiles] = indices
        
        # Initialize a single random number generator
        self.rng = np.random.default_rng(self.random_seed)
        print(f"Pre-computed conditional genomic indices for {len(self.unique_smiles)} molecules")
    
    def __len__(self):
        return len(self.unique_smiles)
    
    def __getitem__(self, index):
        """Get a molecule-wise sample from conditional genomic data with embeddings."""
        smiles = self.unique_smiles[index]
        
        # Get pre-computed row indices for this SMILES (no pandas filtering!)
        row_indices = self.smiles_to_indices[smiles]
        
        # Use consistent approach for proper randomness with seed_everything()
        random_state = self.rng.integers(0, 2**32-1)
        temp_rng = np.random.default_rng(random_state)
        selected_row_idx = temp_rng.choice(row_indices)
        
        # Get features directly from pre-processed numpy array (no pandas!)
        genomic_features = self.features_array[selected_row_idx].copy()
        
        # Get conditional information from the selected row
        row = self.df.iloc[selected_row_idx]
        cell_line = row[self.cell_line_col]
        dose = row['Metadata_Dose_Level']
        time = row['Metadata_pert_time']
        
        # Convert to embedding indices
        cell_idx = self.cell_line_to_idx[cell_line]
        dose_idx = self.dose_to_idx[dose]
        time_idx = self.time_to_idx[time]
        
        # Create conditional features tensor (we'll embed these in the model)
        conditional_features = torch.tensor([cell_idx, dose_idx, time_idx], dtype=torch.long)
        
        return {
            'genomic_features': torch.from_numpy(genomic_features),
            'conditional_features': conditional_features
        }
    
    def get_embedding_dims(self):
        """Get the dimensions for embedding layers."""
        return {
            'n_cell_lines': len(self.cell_line_to_idx),
            'n_dose_levels': len(self.dose_to_idx),
            'n_time_points': len(self.time_to_idx)
        }


class JUMPDataModule(pl.LightningDataModule):
    """Data module for JUMP Cell Painting dataset using build_dataloaders with predefined splits."""
    
    def __init__(self, config: Config):
        """Initialize data module from config."""
        super().__init__()
        self.config = config
        
        # Will be set during setup
        self.feature_dim = None
        self.dataloaders = None
        self.dataset = None
    
    def prepare_data(self):
        """Download or prepare data if needed."""
        data_path = self.config.data['data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Check split files if provided
        splits = self.config.splits
        if splits:
            for split_name, split_path in splits.items():
                if not os.path.exists(split_path):
                    raise FileNotFoundError(f"Split file not found: {split_path}")
    
    def setup(self, stage: str = None):
        """Load and split data using build_dataloaders."""
        # Skip setup if already done
        if self.dataset is not None and self.dataloaders is not None:
            print("Data module already set up, skipping...")
            return
            
        print("Setting up JUMP data module...")
        
        # Create dataset
        self.dataset = JUMPCellPaintingDataset(self.config)
        self.feature_dim = self.dataset.feature_dim
        
        # Build dataloaders using the existing build_dataloaders function
        self.dataloaders = build_dataloaders(
            dataset=self.dataset,
            batch_size=self.config.data['batch_size'],
            splits=self.config.splits,
            num_workers=self.config.data['num_workers'],
            pin_memory=True,
        )
        
        print(f"Data module setup complete. Feature dimension: {self.feature_dim}")
    
    def train_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing dataloaders")
        return self.dataloaders.get("train")
    
    def val_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing dataloaders")
        return self.dataloaders.get("val")
    
    def test_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing dataloaders")
        return self.dataloaders.get("test", self.dataloaders.get("val"))  # Use val as test if no test split


class GenomicDataModule(pl.LightningDataModule):
    """Data module for genomic (LINCS L1000) dataset using build_dataloaders with predefined splits."""
    
    def __init__(self, config: Config):
        """Initialize genomic data module from config."""
        super().__init__()
        self.config = config
        
        # Will be set during setup
        self.feature_dim = None
        self.dataloaders = None
        self.dataset = None
        
        # Check if conditional mode is enabled
        self.conditional_mode = config.model.get('conditional_mode', False)
        self.embedding_dims = None
    
    def prepare_data(self):
        """Download or prepare genomic data if needed."""
        # Check main genomic data path
        genomic_data_path = self.config.data.get('genomic_data_path', self.config.data['data_path'])
        if not os.path.exists(genomic_data_path):
            raise FileNotFoundError(f"Genomic data file not found: {genomic_data_path}")
        
        # Check split files if provided
        splits = self.config.splits
        if splits:
            for split_name, split_path in splits.items():
                if not os.path.exists(split_path):
                    raise FileNotFoundError(f"Split file not found: {split_path}")
    
    def setup(self, stage: str = None):
        """Load and split genomic data using build_dataloaders."""
        # Skip setup if already done
        if self.dataset is not None and self.dataloaders is not None:
            print("Genomic data module already set up, skipping...")
            return
            
        print("Setting up genomic data module...")
        
        # Create genomic dataset (conditional or regular)
        if self.conditional_mode:
            self.dataset = ConditionalGenomicDataset(self.config)
            self.embedding_dims = self.dataset.get_embedding_dims()
            print(f"Using conditional genomic dataset with embeddings: {self.embedding_dims}")
        else:
            self.dataset = GenomicDataset(self.config)
        
        self.feature_dim = self.dataset.feature_dim
        
        # Build dataloaders using the existing build_dataloaders function
        self.dataloaders = build_dataloaders(
            dataset=self.dataset,
            batch_size=self.config.data['batch_size'],
            splits=self.config.splits,
            num_workers=self.config.data['num_workers'],
            pin_memory=True,
        )
        
        print(f"Genomic data module setup complete. Feature dimension: {self.feature_dim}")
    
    def train_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing genomic dataloaders")
        return self.dataloaders.get("train")
    
    def val_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing genomic dataloaders")
        return self.dataloaders.get("val")
    
    def test_dataloader(self):
        if self.dataloaders is None:
            raise RuntimeError("setup() must be called before accessing genomic dataloaders")
        return self.dataloaders.get("test", self.dataloaders.get("val"))  # Use val as test if no test split


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(_config=config_dict)


def create_model(config: Config, input_dim: int, embedding_dims: Optional[Dict] = None):
    """Create model from config."""
    model_config = config.model
    opt_config = config.optimization
    
    # Handle norm_type
    norm_type = model_config['norm_type']
    if norm_type == "none":
        norm_type = None
    
    # Set up conditional parameters
    conditional_mode = model_config.get('conditional_mode', False)
    conditional_kwargs = {}
    
    if conditional_mode and embedding_dims is not None:
        # Add embedding parameters for conditional mode
        conditional_kwargs.update({
            'conditional_mode': True,
            'n_cell_lines': embedding_dims['n_cell_lines'],
            'n_dose_levels': embedding_dims['n_dose_levels'],
            'n_time_points': embedding_dims['n_time_points'],
            'cell_embedding_dim': model_config.get('cell_embedding_dim', 32),
            'dose_embedding_dim': model_config.get('dose_embedding_dim', 32),
            'time_embedding_dim': model_config.get('time_embedding_dim', 32),
        })
        print(f"Creating conditional VAE with embeddings: {embedding_dims}")
    elif conditional_mode:
        raise ValueError("Conditional mode enabled but no embedding dimensions provided")
    
    return create_vae_model(
        input_dim=input_dim,
        architecture=model_config['architecture'],
        latent_dim=model_config['latent_dim'],
        dropout=model_config['dropout'],
        norm_type=norm_type,
        learning_rate=opt_config['learning_rate'],
        weight_decay=opt_config['weight_decay'],
        beta=model_config['beta'],
        model_type=model_config['model_type'],
        scheduler_type=model_config['scheduler_type'],
        T_max=model_config['T_max'],
        eta_min=model_config['eta_min'],
        warmup_epochs=model_config['warmup_epochs'],
        **conditional_kwargs
    )


def setup_logging(config: Config):
    """Setup loggers from config."""
    log_config = config.logging
    
    # Create log directory if it doesn't exist
    log_dir = log_config['log_dir']
    experiment_name = log_config['experiment_name']
    full_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(full_log_dir, exist_ok=True)
    print(f"Created log directory: {full_log_dir}")
    
    loggers = []
    
    if log_config['use_wandb']:
        wandb_logger = WandbLogger(
            project=log_config['project_name'],
            name=log_config['experiment_name'],
            save_dir=log_config['log_dir'],
            entity=log_config['entity'],
            log_model=False,
        )
        loggers.append(wandb_logger)
        
        # Log configuration parameters
        wandb_logger.experiment.config.update(config._config)
    else:
        tb_logger = TensorBoardLogger(
            save_dir=log_config['log_dir'],
            name=log_config['experiment_name'],
        )
        loggers.append(tb_logger)
    
    return loggers


def setup_callbacks(config: Config):
    """Setup callbacks from config - with optional model checkpointing."""
    log_config = config.logging
    train_config = config.training
    
    callbacks = [
        EarlyStopping(
            monitor="val/total_loss",
            patience=train_config['patience'],
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Add model checkpointing if enabled in config
    save_model_weights = train_config.get('save_model_weights', False)
    if save_model_weights:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(log_config['log_dir'], log_config['experiment_name'], "checkpoints"),
                filename="best-epoch={epoch:02d}-val_total_loss={val/total_loss:.3f}",
                monitor="val/total_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,  # Prevents automatic metric insertion
            )
        )
        print("Model checkpointing enabled")
    else:
        print("Model checkpointing disabled - only JSON results will be saved")
    
    return callbacks


def create_trainer(config: Config, callbacks, loggers):
    """Create trainer from config."""
    train_config = config.training
    
    # Disable checkpointing if save_model_weights is False
    enable_checkpointing = train_config.get('save_model_weights', False)
    
    return pl.Trainer(
        max_epochs=train_config['max_epochs'],
        accelerator="gpu" if train_config['gpus'] > 0 else "cpu",
        devices=train_config['gpus'] if train_config['gpus'] > 0 else 1,
        precision=train_config['precision'],
        fast_dev_run=train_config['fast_dev_run'],
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=enable_checkpointing,
    )


def main():
    parser = argparse.ArgumentParser(description="Train VAE on JUMP Cell Painting data")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load and validate config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config loaded successfully")
    
    # Print configuration summary
    config.print_summary()
    
    # Set random seed
    pl.seed_everything(config.training['seed'])
    
    # Setup data module based on modality
    modality = config.modality
    print(f"Setting up data module for modality: {modality}")
    
    if modality == 'genomic':
        data_module = GenomicDataModule(config)
        print("Using GenomicDataModule for LINCS L1000 data")
    elif modality == 'jump_cp':
        data_module = JUMPDataModule(config)
        print("Using JUMPDataModule for Cell Painting data")
    else:
        raise ValueError(f"Unknown modality: {modality}. Supported modalities: 'jump_cp', 'genomic'")
    
    data_module.prepare_data()
    # Note: setup() will be called automatically by PyTorch Lightning during trainer.fit()
    
    # We need to setup once to get feature_dim for model creation
    data_module.setup()  # This will be called again by trainer.fit(), but that's okay
    
    # Create model
    print("Creating model...")
    embedding_dims = getattr(data_module, 'embedding_dims', None)
    model = create_model(config, data_module.feature_dim, embedding_dims)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training components
    loggers = setup_logging(config)
    callbacks = setup_callbacks(config)
    trainer = create_trainer(config, callbacks, loggers)
    
    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test model
    save_model_weights = config.training.get('save_model_weights', False)
    if save_model_weights:
        print("Testing best model...")
        trainer.test(model, data_module, ckpt_path="best")
        
        # Save final model
        log_config = config.logging
        final_model_path = os.path.join(log_config['log_dir'], log_config['experiment_name'], "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")
    else:
        print("Testing current model...")
        trainer.test(model, data_module)
    
    # Save final metrics to JSON
    save_final_metrics(
        trainer=trainer,
        experiment_name=config.logging['experiment_name'],
        log_dir=config.logging['log_dir'],
        hyperparameters=config._config
    )
    
    print("Training completed! Final results saved to JSON file.")
    
    if config.logging['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main() 