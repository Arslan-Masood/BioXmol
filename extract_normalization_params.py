#!/usr/bin/env python3
"""
Utility script to extract and save normalization parameters from existing VAE training setup.
This allows you to get the fast evaluation benefits without retraining.
"""

import os
import sys
import argparse
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_vae import load_config, JUMPDataModule, GenomicDataModule
import pytorch_lightning as pl


def extract_and_save_normalization_params(config_path: str, output_dir: str = None):
    """Extract normalization parameters and save them for fast evaluation."""
    
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Set up data module (this computes normalization parameters)
    print("Setting up data module to compute normalization parameters...")
    
    modality = config.modality
    print(f"Data modality: {modality}")
    
    if modality == 'genomic':
        data_module = GenomicDataModule(config)
    elif modality == 'jump_cp':
        data_module = JUMPDataModule(config)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    # Setup data (this computes the normalization parameters)
    data_module.setup()
    
    # Check if normalization is enabled
    dataset = data_module.dataset
    if not (hasattr(dataset, 'normalize') and dataset.normalize):
        print("❌ Normalization is not enabled in this config. No parameters to save.")
        return False
    
    # Determine save location from config or provided output_dir
    if output_dir is None:
        # Use config's log directory
        experiment_dir = os.path.join(config.logging['log_dir'], config.logging['experiment_name'])
        os.makedirs(experiment_dir, exist_ok=True)
        normalization_path = os.path.join(experiment_dir, 'normalization_params.npz')
    else:
        os.makedirs(output_dir, exist_ok=True)
        normalization_path = os.path.join(output_dir, 'normalization_params.npz')
    
    # Save normalization parameters
    print(f"💾 Saving normalization parameters to: {normalization_path}")
    
    np.savez(normalization_path,
             feature_mean=dataset.feature_mean,
             feature_std=dataset.feature_std,
             feature_dim=dataset.feature_dim,
             feature_cols=dataset.feature_cols)
    
    print(f"✅ Successfully saved normalization parameters:")
    print(f"   Feature mean shape: {dataset.feature_mean.shape}")
    print(f"   Feature std shape: {dataset.feature_std.shape}")
    print(f"   Feature dimension: {dataset.feature_dim}")
    print(f"   Saved to: {normalization_path}")
    
    # Verify the saved file
    if os.path.exists(normalization_path):
        file_size = os.path.getsize(normalization_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        
        # Quick verification by loading
        print("🔍 Verifying saved parameters...")
        norm_data = np.load(normalization_path)
        print(f"   Loaded keys: {list(norm_data.keys())}")
        print(f"   Feature mean range: {norm_data['feature_mean'].min():.6f} to {norm_data['feature_mean'].max():.6f}")
        print(f"   Feature std range: {norm_data['feature_std'].min():.6f} to {norm_data['feature_std'].max():.6f}")
        
        return True
    else:
        print("❌ Failed to save normalization parameters")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract and save normalization parameters from existing VAE setup")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file used for training")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save normalization params (default: use config's log_dir/experiment_name)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print("🚀 Extracting normalization parameters from existing setup...")
    print(f"Config: {args.config}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print("Output directory: Using config's log_dir/experiment_name")
    print("-" * 60)
    
    try:
        success = extract_and_save_normalization_params(args.config, args.output_dir)
        
        if success:
            print("\n🎉 Success! Your evaluation will now use fast normalization loading.")
            print("   Next time you run evaluation, it will be much faster!")
        else:
            print("\n❌ Failed to extract normalization parameters.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
