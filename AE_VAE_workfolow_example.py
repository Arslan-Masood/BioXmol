#!/usr/bin/env python3
"""
Example usage of JUMP Cell Painting VAE models.
This script demonstrates how to use the updated JUMPDataModule with predefined splits
and molecule-wise sampling similar to CellLineTripleInputGraphDatasetJUMP.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import create_vae_model
from train_vae import JUMPDataModule, JUMPCellPaintingDataset

def load_trained_model(checkpoint_path: str, input_dim: int):
    """Load a trained VAE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model (architecture should match the saved model)
    model = create_vae_model(
        input_dim=input_dim,
        architecture="vanilla",  # vanilla, medium, or large - Adjust as needed
        latent_dim=128,
        dropout=0.1,
        norm_type="batchnorm",
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print("Model loaded successfully!")
    return model

def example_with_predefined_splits():
    """Example using the new JUMPDataModule with predefined splits."""
    print("=" * 60)
    print("Example: Using JUMPDataModule with Predefined Splits")
    print("=" * 60)
    
    # Example paths (adjust these to your actual paths)
    data_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/cell_fetures_with_smiles_2000.parquet"
    
    # Example split files
    splits = {
            "train": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-0-train.csv",
            "val": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-0-val.csv",
            "test": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/jump-compound-split-0-val.csv",  # Using val as test
    }
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path to point to your JUMP data file.")
        return
    
    missing_splits = [split for split, path in splits.items() if not os.path.exists(path)]
    if missing_splits:
        print(f"Split files not found for: {missing_splits}")
        print("Please update the split paths or set splits=None for random splitting.")
        print("Falling back to random splitting...")
        splits = None
    
    # Create data module
    data_module = JUMPDataModule(
        data_path=data_path,
        splits=splits,
        batch_size=64,
        num_workers=2,
        normalize=False,
        random_seed=42,
    )
    
    # Setup data module
    data_module.prepare_data()
    data_module.setup()
    
    print(f"Feature dimension: {data_module.feature_dim}")
    print(f"Number of unique molecules: {len(data_module.dataset.unique_smiles)}")
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    if train_loader:
        print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Sample a batch to see the data
    if train_loader:
        sample_batch = next(iter(train_loader))
        print(f"Sample batch shape: {sample_batch.shape}")
        print(f"Sample batch stats: mean={sample_batch.mean():.3f}, std={sample_batch.std():.3f}")
    
    return data_module

def example_molecule_wise_dataset():
    """Example of directly using the molecule-wise dataset."""
    print("=" * 60)
    print("Example: Direct Usage of JUMPCellPaintingDataset")
    print("=" * 60)
    
    # Example data path
    data_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/cell_fetures_with_smiles_2000.parquet"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path to point to your JUMP data file.")
        return
    
    # Create dataset
    dataset = JUMPCellPaintingDataset(
        data_path=data_path,
        normalize=False,
        random_seed=42,
    )
    
    print(f"Dataset length (unique molecules): {len(dataset)}")
    print(f"Feature dimension: {dataset.feature_dim}")
    print(f"Number of unique SMILES: {len(dataset.unique_smiles)}")
    
    # Sample a few molecules
    print("\nSampling molecules:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Molecule {i}: shape={sample.shape}, mean={sample.mean():.3f}, std={sample.std():.3f}")
    
    # Show SMILES distribution if available
    if hasattr(dataset, 'df') and 'Metadata_SMILES' in dataset.df.columns:
        smiles_counts = dataset.df['Metadata_SMILES'].value_counts()
        print(f"\nSMILES with most replicates: {smiles_counts.head()}")
        print(f"Average replicates per SMILES: {smiles_counts.mean():.1f}")
    
    return dataset

def example_train_simple_vae():
    """Example of training a simple VAE model."""
    print("=" * 60)
    print("Example: Training a Simple VAE")
    print("=" * 60)
    
    # Get data module
    data_module = example_with_predefined_splits()
    if data_module is None:
        return
    
    # Create a simple VAE model
    model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,  # Smaller latent dim for demo
        dropout=0.1,
        norm_type="batchnorm",
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=1.0,
        model_type="vae",  # VAE mode
    )
    
    print(f"VAE Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo forward pass
    train_loader = data_module.train_dataloader()
    if train_loader:
        sample_batch = next(iter(train_loader))
        
        with torch.no_grad():
            reconstruction, mu, logvar = model(sample_batch)
            total_loss, recon_loss, kl_loss = model.loss_function(reconstruction, sample_batch, mu, logvar)
        
        print("\nDemo VAE forward pass:")
        print(f"Input shape: {sample_batch.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        print(f"Reconstruction loss: {recon_loss:.4f}")
        print(f"KL divergence: {kl_loss:.4f}")
        print(f"Total loss: {total_loss:.4f}")
    
    return model, data_module

def example_train_simple_ae():
    """Example of training a simple AE model."""
    print("=" * 60)
    print("Example: Training a Simple AE")
    print("=" * 60)
    
    # Get data module
    data_module = example_with_predefined_splits()
    if data_module is None:
        return
    
    # Create a simple AE model
    model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,  # Smaller latent dim for demo
        dropout=0.1,
        norm_type="batchnorm",
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=1.0,  # This will be ignored for AE
        model_type="ae",  # AE mode
    )
    
    print(f"AE Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo forward pass
    train_loader = data_module.train_dataloader()
    if train_loader:
        sample_batch = next(iter(train_loader))
        
        with torch.no_grad():
            reconstruction, latent, zeros = model(sample_batch)
            total_loss, recon_loss, kl_loss = model.loss_function(reconstruction, sample_batch, latent, zeros)
        
        print("\nDemo AE forward pass:")
        print(f"Input shape: {sample_batch.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Reconstruction loss: {recon_loss:.4f}")
        print(f"KL divergence (should be 0): {kl_loss:.4f}")
        print(f"Total loss (same as recon loss): {total_loss:.4f}")
    
    return model, data_module

def compare_vae_vs_ae():
    """Compare VAE and AE on the same data."""
    print("=" * 60)
    print("Example: Comparing VAE vs AE")
    print("=" * 60)
    
    # Get data module
    data_module = example_with_predefined_splits()
    if data_module is None:
        return
    
    # Create both models
    vae_model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,
        model_type="vae",
    )
    
    ae_model = create_vae_model(
        input_dim=data_module.feature_dim,
        architecture="vanilla",  # vanilla, medium, or large
        latent_dim=64,
        model_type="ae",
    )
    
    print(f"VAE parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    print(f"AE parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    
    # Demo comparison on same data
    val_loader = data_module.val_dataloader()
    if val_loader:
        sample_batch = next(iter(val_loader))
        
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
        
        print(f"\nComparison on {sample_batch.shape[0]} samples:")
        print(f"VAE - Recon Loss: {vae_recon_loss:.4f}, KL Loss: {vae_kl_loss:.4f}, Total: {vae_total_loss:.4f}")
        print(f"AE  - Recon Loss: {ae_recon_loss:.4f}, KL Loss: {ae_kl_loss:.4f}, Total: {ae_total_loss:.4f}")
        
        # Compare latent space properties
        vae_latent_std = torch.std(vae_mu, dim=0).mean()
        ae_latent_std = torch.std(ae_latent, dim=0).mean()
        
        print(f"\nLatent space properties:")
        print(f"VAE - Mean latent std: {vae_latent_std:.4f}")
        print(f"AE  - Mean latent std: {ae_latent_std:.4f}")
        
        # VAE should have more regularized (closer to unit gaussian) latent space
        print(f"VAE latent mean: {torch.mean(vae_mu):.4f} (should be close to 0)")
        print(f"AE latent mean: {torch.mean(ae_latent):.4f} (no constraint)")
    
    return vae_model, ae_model, data_module

def example_evaluate_model(model, data_module):
    """Example of evaluating a trained model."""
    print("=" * 60)
    print("Example: Model Evaluation")
    print("=" * 60)
    
    model.eval()
    
    # Evaluate on validation set
    val_loader = data_module.val_dataloader()
    if not val_loader:
        print("No validation loader available")
        return
    
    all_inputs = []
    all_reconstructions = []
    all_latents = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch in val_loader:
            reconstruction, mu, logvar = model(batch)
            
            all_inputs.append(batch.numpy())
            all_reconstructions.append(reconstruction.numpy())
            all_latents.append(mu.numpy())  # Use mean of latent distribution
    
    # Concatenate all batches
    inputs = np.concatenate(all_inputs, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    
    print(f"Evaluation data shape: {inputs.shape}")
    print(f"Latent representations shape: {latents.shape}")
    
    # Compute reconstruction metrics
    mae = mean_absolute_error(inputs.flatten(), reconstructions.flatten())
    mse = mean_squared_error(inputs.flatten(), reconstructions.flatten())
    
    print(f"\nReconstruction Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Correlation between input and reconstruction
    correlation = np.corrcoef(inputs.flatten(), reconstructions.flatten())[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    
    return inputs, reconstructions, latents

def example_latent_analysis(latents, data_module, model_name):
    """Example of analyzing latent representations."""
    print("=" * 60)
    print("Example: Latent Space Analysis")
    print("=" * 60)
    
    print(f"Latent space shape: {latents.shape}")
    print(f"Latent space stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
    
    # PCA analysis
    print("\nRunning PCA...")
    pca = PCA(n_components=min(10, latents.shape[1]))
    latents_pca = pca.fit_transform(latents)
    
    print(f"PCA explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative explained variance (10 components): {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}")
    
    # t-SNE visualization (on a subset for speed)
    if latents.shape[0] > 1000:
        subset_idx = np.random.choice(latents.shape[0], 1000, replace=False)
        latents_subset = latents[subset_idx]
    else:
        latents_subset = latents
    
    print(f"\nRunning t-SNE on {latents_subset.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_tsne = tsne.fit_transform(latents_subset)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6, s=20)
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("vae_analysis_plots", exist_ok=True)
    plt.savefig(f"vae_analysis_plots/{model_name}_latent_tsne.png", dpi=300, bbox_inches="tight")
    print(f"t-SNE plot saved to: vae_analysis_plots/{model_name}_latent_tsne.png")
    plt.close()
    
    return latents_pca, latents_tsne

def main():
    """Run all examples."""
    print("JUMP Cell Painting VAE/AE Examples")
    print("=" * 60)
    
    try:
        # Example 1: Data module with predefined splits
        data_module = example_with_predefined_splits()
        
        if data_module is not None:
            # Example 2: Direct dataset usage
            dataset = example_molecule_wise_dataset()
            
            # Example 3: Train simple VAE model
            vae_model, data_module = example_train_simple_vae()
            
            # Example 4: Train simple AE model
            ae_model, data_module = example_train_simple_ae()
            
            # Example 5: Compare VAE vs AE
            vae_model, ae_model, data_module = compare_vae_vs_ae()
            
            # Example 6: Evaluate VAE model
            print("\n" + "=" * 60)
            print("Evaluating VAE Model")
            print("=" * 60)
            inputs, reconstructions, latents = example_evaluate_model(vae_model, data_module)
            
            # Example 7: Evaluate AE model
            print("\n" + "=" * 60)
            print("Evaluating AE Model")
            print("=" * 60)
            ae_inputs, ae_reconstructions, ae_latents = example_evaluate_model(ae_model, data_module)
            
            # Example 8: Latent analysis for both models
            print("\n" + "=" * 60)
            print("VAE Latent Analysis")
            print("=" * 60)
            vae_latents_pca, vae_latents_tsne = example_latent_analysis(latents, data_module, model_name="vae")
            
            print("\n" + "=" * 60)
            print("AE Latent Analysis")
            print("=" * 60)
            ae_latents_pca, ae_latents_tsne = example_latent_analysis(ae_latents, data_module, model_name="ae")
            
            print("\n" + "=" * 60)
            print("All examples completed successfully!")
            print("Key differences observed:")
            print("1. VAE has KL divergence loss, AE does not")
            print("2. VAE latent space is regularized (closer to unit Gaussian)")
            print("3. AE latent space is unconstrained and typically has higher variance")
            print("4. VAE can generate new samples by sampling from latent space")
            print("5. AE reconstruction is typically better due to no KL constraint")
            print("Check the 'vae_analysis_plots' directory for visualization outputs.")
            
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print("This is likely due to missing data files. Please check the file paths.")
        print("You can modify the paths in this script to match your data location.")

if __name__ == "__main__":
    main() 