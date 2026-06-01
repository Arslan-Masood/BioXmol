#!/usr/bin/env python3
"""
Script to extract GNN embeddings from pretrained LightningGGNN model for molecular features.

This script extracts embeddings from specific layers of a pretrained GNN model and saves
them in CSV format compatible with the existing feature extraction pipeline.

Usage:
    python featurizer_GNN.py --input_file <input.csv> --checkpoint_path <checkpoint.ckpt> --layer_name <layer>
"""

import warnings
import os
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*please use MorganGenerator.*')
warnings.filterwarnings('ignore', message='.*Skipped loading.*')
warnings.filterwarnings('ignore', message='.*missing a dependency.*')

import logging
logging.getLogger('rdkit').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader

# Add the mocop directory to Python path
sys.path.insert(1, '/scratch/work/masooda1/Multi_Modal_Contrastive/mocop')

from model import LightningGGNN
from dataset import SupervisedGraphDataset


def extract_from_layer(model, x_a, layer_name):
    """Extract embeddings from specific layers of the GatedGraphNeuralNetwork.
    
    Args:
        model: LightningGGNN model
        x_a: Input tensor [adj_mat, node_feat, atom_vec]
        layer_name: One of 'GNN', 'first_fc', 'second_fc'
    
    Returns:
        torch.Tensor: Extracted embeddings
    """
    adj, node_feat, atom_vec = x_a
    
    # Forward through all conv layers (common for all options)
    for layer in model.model.conv_layers:
        node_feat = layer(adj, node_feat)
        node_feat = model.model.dropout(node_feat)
    
    # Apply atom_vec and sum to get graph-level representation
    output = torch.mul(node_feat, atom_vec)
    output = output.sum(1)
    
    if layer_name == 'GNN':
        # Return after conv layers (75-dim)
        return output
    elif layer_name == 'first_fc':
        # Continue through first FC layer (1024-dim)
        output = model.model.fc_layers[0](output)
        return output
    elif layer_name == 'second_fc':
        # Continue through first FC layer, then second FC layer (128-dim)
        output = model.model.fc_layers[0](output)
        output = model.model.dropout(output)
        output = model.model.fc_layers[1](output)
        return output
    else:
        raise ValueError(f"Unknown layer: {layer_name}. Choose from: GNN, first_fc, second_fc")


def simple_collate_fn(batch):
    """Simple collate function for graph data without multi-modal features.
    
    Handles batching of variable-length graph data by padding to the maximum size in the batch.
    """
    # Find maximum size in the batch
    max_size = max(item["inputs"]["x_a"][0].shape[0] for item in batch)
    
    # Pad and stack molecular features (x_a)
    adj_mats_list = []
    node_feats_list = []
    atom_vecs_list = []
    
    for item in batch:
        adj_mat, node_feat, atom_vec = item["inputs"]["x_a"]
        
        # Pad to max_size
        pad_size = max_size - adj_mat.shape[0]
        if pad_size > 0:
            adj_mat = torch.nn.functional.pad(adj_mat, (0, pad_size, 0, pad_size), "constant", 0)
            node_feat = torch.nn.functional.pad(node_feat, (0, 0, 0, pad_size), "constant", 0)
            atom_vec = torch.nn.functional.pad(atom_vec, (0, 0, 0, pad_size), "constant", 0)
        
        adj_mats_list.append(adj_mat)
        node_feats_list.append(node_feat)
        atom_vecs_list.append(atom_vec)
    
    # Stack all tensors
    adj_mats = torch.stack(adj_mats_list)
    node_feats = torch.stack(node_feats_list)
    atom_vecs = torch.stack(atom_vecs_list)
    x_a_batch = [adj_mats, node_feats, atom_vecs]
    
    # Stack labels if they exist (create dummy labels if not)
    if "labels" in batch[0] and batch[0]["labels"] is not None:
        labels_batch = torch.stack([item["labels"] for item in batch])
    else:
        # Create dummy labels (won't be used)
        labels_batch = torch.zeros(len(batch))
    
    return {
        "inputs": {"x_a": x_a_batch},
        "labels": labels_batch
    }


def extract_gnn_embeddings(model, dataset, batch_size=32, layer_name='GNN', device=None):
    """Extract embeddings from the specified layer of the model.
    
    Args:
        model: LightningGGNN model
        dataset: SupervisedGraphDataset instance
        batch_size: Batch size for processing
        layer_name: Layer to extract from ('GNN', 'first_fc', 'second_fc')
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        numpy.ndarray: Extracted embeddings of shape (n_samples, feature_dim)
    """
    model.eval()
    embeddings = []
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn
    )
    
    print(f"Extracting embeddings from layer: {layer_name}")
    print(f"  Using device: {device}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Computing GNN-{layer_name} features")):
            inputs = batch["inputs"]
            
            # Move to device - handle x_a which is a list of tensors
            if device == 'cuda' and torch.cuda.is_available():
                inputs = {}
                for k, v in batch["inputs"].items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.cuda()
                    elif isinstance(v, list):
                        # Handle list of tensors (e.g., x_a = [adj_mat, node_feat, atom_vec])
                        inputs[k] = [t.cuda() if isinstance(t, torch.Tensor) else t for t in v]
                    else:
                        inputs[k] = v
            
            # Extract embeddings from specified layer
            batch_embeddings = extract_from_layer(model, inputs['x_a'], layer_name)
            
            # Move back to CPU for numpy conversion
            embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    X = np.vstack(embeddings)
    
    print(f"Extracted embeddings shape: {X.shape}")
    
    return X


def compute_gnn_features_for_dataset(input_file: str, 
                                     output_file: Optional[str],
                                     checkpoint_path: str,
                                     smiles_col: str = "Normalized_SMILES_combined",
                                     layer_name: str = "GNN",
                                     batch_size: int = 32,
                                     device: Optional[str] = None):
    """
    Compute GNN embeddings for SMILES in a dataset and save with metadata columns.
    
    Args:
        input_file: Path to input CSV file containing SMILES
        output_file: Path to output CSV file (auto-generated if None)
        checkpoint_path: Path to LightningGGNN checkpoint file
        smiles_col: Name of the column containing SMILES strings
        layer_name: Layer to extract from ('GNN', 'first_fc', 'second_fc')
        batch_size: Batch size for processing (default: 32)
        device: Device to use for model ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        str: Path to the generated feature file
    """
    # Validate checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds from {input_file}")
    
    # Check required columns
    required_cols = ["LTKBID", "CompoundName", smiles_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Get SMILES list
    smiles_list = df[smiles_col].tolist()
    
    # Filter out None/NaN values for processing
    valid_indices = [i for i, s in enumerate(smiles_list) if pd.notna(s) and s is not None and str(s).strip() != '']
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    print(f"Computing features for {len(valid_smiles)} valid SMILES...")
    
    # Create a temporary CSV file for the dataset (SupervisedGraphDataset requires a file path)
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_df = df[[smiles_col]].copy()
    temp_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Create dataset
        dataset = SupervisedGraphDataset(
            data_path=temp_file.name,
            cmpd_col=smiles_col,
            label_col=None,  # We don't need labels for feature extraction
            cmpd_col_is_inchikey=False,
            pad_length=250
        )
        
        # Load pretrained model
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = LightningGGNN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=False,
            n_edge=1,
            in_dim=75,
            n_conv=6,
            fc_dims=[1024, 128, 1],
            p_dropout=0.1,
            freeze=False
        )
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        # Extract embeddings
        valid_features = extract_gnn_embeddings(
            model, dataset, batch_size, layer_name, device
        )
        
        # Get feature dimension
        feature_dim = valid_features.shape[1]
        
        # Create full feature array with zeros for invalid SMILES
        features_array = np.zeros((len(smiles_list), feature_dim))
        for idx, valid_idx in enumerate(valid_indices):
            features_array[valid_idx] = valid_features[idx]
        
        # Add feature columns
        feature_cols = [f'feature_{i}' for i in range(feature_dim)]
        feature_df = pd.DataFrame(features_array, columns=feature_cols)
        output_df = pd.concat([df, feature_df], axis=1)
        
        # Generate output filename if not provided
        if output_file is None:
            input_dir = os.path.dirname(input_file)
            input_basename = os.path.basename(input_file)
            input_name = os.path.splitext(input_basename)[0]
            output_file = os.path.join(input_dir, f"{input_name}_GNN_{layer_name}.csv")
        
        # Save features
        output_df.to_csv(output_file, index=False)
        print(f"\nFeatures saved to: {output_file}")
        print(f"  Shape: {output_df.shape}")
        
        # Statistics
        successful = len(valid_smiles)
        failed = len(smiles_list) - successful
        
        print(f"\nSummary:")
        print(f"  Successful: {successful}/{len(smiles_list)} ({successful/len(smiles_list)*100:.1f}%)")
        print(f"  Failed: {failed}/{len(smiles_list)} ({failed/len(smiles_list)*100:.1f}%)")
        
        return output_file
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GNN embeddings from pretrained LightningGGNN model"
    )
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="Path to input CSV file containing SMILES")
    parser.add_argument("--output_file", "-o", type=str, default=None,
                       help="Path to output CSV file (auto-generated if not provided)")
    parser.add_argument("--checkpoint_path", "-c", type=str, required=True,
                       help="Path to LightningGGNN checkpoint file")
    parser.add_argument("--smiles_col", "-s", type=str, default="Normalized_SMILES_combined",
                       help="Name of the column containing SMILES strings (default: Normalized_SMILES_combined)")
    parser.add_argument("--layer_name", "-l", type=str, required=True,
                       choices=["GNN", "first_fc", "second_fc"],
                       help="Layer to extract embeddings from: GNN, first_fc, or second_fc")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu", None],
                       help="Device to use ('cuda', 'cpu', or None for auto-detect)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    try:
        compute_gnn_features_for_dataset(
            input_file=args.input_file,
            output_file=args.output_file,
            checkpoint_path=args.checkpoint_path,
            smiles_col=args.smiles_col,
            layer_name=args.layer_name,
            batch_size=args.batch_size,
            device=args.device
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
