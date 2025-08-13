#!/usr/bin/env python3
"""
Simple example script demonstrating VAE vs AE training and comparison.
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import create_vae_model
from train_vae import JUMPDataModule

def simple_vae_ae_comparison():
    """Simple comparison between VAE and AE models."""
    
    print("=" * 60)
    print("VAE vs AE Comparison Example")
    print("=" * 60)
    
    # Example data paths (update these to your actual paths)
    data_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/cell_fetures_with_smiles_2000.parquet"
    
    # Example split files
    splits = {
        "train": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-0-train.csv",
        "val": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-0-val.csv",
    }
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path to point to your JUMP data file.")
        return
    
    missing_splits = [split for split, path in splits.items() if not os.path.exists(path)]
    if missing_splits:
        print(f"Split files not found for: {missing_splits}")
        print("Using random splitting instead...")
        splits = None
    
    # Create data module
    data_module = JUMPDataModule(
        data_path=data_path,
        splits=splits,
        batch_size=64,
        num_workers=2,
        normalize=True,
        random_seed=42,
    )
    
    # Setup data module
    data_module.prepare_data()
    data_module.setup()
    
    print(f"Feature dimension: {data_module.feature_dim}")
    
    # Create VAE model
    print("\n1. Creating VAE model...")
    vae_model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,
        dropout=0.1,
        norm_type="batchnorm",
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=1.0,
        model_type="vae",  # VAE mode
    )
    
    print(f"VAE parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    
    # Create AE model
    print("\n2. Creating AE model...")
    ae_model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,
        dropout=0.1,
        norm_type="batchnorm",
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=1.0,  # This will be ignored for AE
        model_type="ae",   # AE mode
    )
    
    print(f"AE parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    
    # Test on sample data
    train_loader = data_module.train_dataloader()
    if train_loader:
        sample_batch = next(iter(train_loader))
        print(f"\n3. Testing on batch of {sample_batch.shape[0]} samples...")
        
        with torch.no_grad():
            # VAE forward pass
            vae_recon, vae_mu, vae_logvar = vae_model(sample_batch)
            vae_total_loss, vae_recon_loss, vae_kl_loss = vae_model.loss_function(
                vae_recon, sample_batch, vae_mu, vae_logvar
            )
            
            # AE forward pass
            ae_recon, ae_latent, ae_zeros = ae_model(sample_batch)
            ae_total_loss, ae_recon_loss, ae_kl_loss = ae_model.loss_function(
                ae_recon, sample_batch, ae_latent, ae_zeros
            )
        
        print("\n4. Results comparison:")
        print(f"VAE - Reconstruction Loss: {vae_recon_loss:.4f}")
        print(f"VAE - KL Divergence:       {vae_kl_loss:.4f}")
        print(f"VAE - Total Loss:          {vae_total_loss:.4f}")
        print(f"")
        print(f"AE  - Reconstruction Loss: {ae_recon_loss:.4f}")
        print(f"AE  - KL Divergence:       {ae_kl_loss:.4f} (should be 0)")
        print(f"AE  - Total Loss:          {ae_total_loss:.4f} (same as recon)")
        
        # Latent space statistics
        vae_latent_mean = torch.mean(vae_mu).item()
        vae_latent_std = torch.std(vae_mu).item()
        ae_latent_mean = torch.mean(ae_latent).item()
        ae_latent_std = torch.std(ae_latent).item()
        
        print(f"\n5. Latent space comparison:")
        print(f"VAE latent - Mean: {vae_latent_mean:.4f}, Std: {vae_latent_std:.4f}")
        print(f"AE latent  - Mean: {ae_latent_mean:.4f}, Std: {ae_latent_std:.4f}")
        
        print(f"\n6. Key differences:")
        print(f"- VAE latent space is regularized to be close to N(0,1)")
        print(f"- AE latent space is unconstrained")
        print(f"- VAE has reconstruction + KL divergence loss")
        print(f"- AE has only reconstruction loss")
        print(f"- VAE can generate new samples, AE typically reconstructs better")

def usage_examples():
    """Show usage examples for different scenarios."""
    
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("\n1. Command line training:")
    print("   # Train VAE:")
    print("   python train_vae.py --data_path data.parquet --model_type vae --beta 1.0")
    print("   ")
    print("   # Train AE:")
    print("   python train_vae.py --data_path data.parquet --model_type ae")
    
    print("\n2. Programmatic usage:")
    print("   from models.vae import create_vae_model")
    print("   ")
    print("   # Create VAE")
    print("   vae = create_vae_model(input_dim=4000, model_type='vae', beta=1.0)")
    print("   ")
    print("   # Create AE")
    print("   ae = create_vae_model(input_dim=4000, model_type='ae')")
    
    print("\n3. When to use VAE vs AE:")
    print("   Use VAE when:")
    print("   - You want to generate new samples")
    print("   - You need regularized latent space")
    print("   - You want to interpolate between samples")
    print("   ")
    print("   Use AE when:")
    print("   - You want best reconstruction quality")
    print("   - You need deterministic latent codes")
    print("   - You want simpler training (no KL divergence)")

if __name__ == "__main__":
    simple_vae_ae_comparison()
    usage_examples() 