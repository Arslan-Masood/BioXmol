#!/usr/bin/env python3
"""
Evaluation script for trained JUMP Cell Painting VAE models.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import JUMPVAE
from train_vae import JUMPDataModule


def load_model_from_checkpoint(checkpoint_path: str) -> JUMPVAE:
    """Load VAE model from checkpoint."""
    model = JUMPVAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def compute_reconstruction_metrics(model: JUMPVAE, dataloader: DataLoader, device: str = "cuda"):
    """Compute reconstruction metrics in original space only with MAE and R² focus."""
    model.to(device)
    model.eval()
    
    all_original = []
    all_reconstructed = []
    all_original_norm = []
    all_reconstructed_norm = []
    all_latent = []
    
    # Get normalization parameters from dataset
    dataset = dataloader.dataset
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'feature_mean'):
        # Handle subset datasets
        feature_mean = dataset.dataset.feature_mean
        feature_std = dataset.dataset.feature_std
        has_normalization = True
    elif hasattr(dataset, 'feature_mean'):
        feature_mean = dataset.feature_mean
        feature_std = dataset.feature_std
        has_normalization = True
    else:
        has_normalization = False
        print("Error: Could not find normalization parameters. Cannot compute original space metrics.")
        return None
    
    with torch.no_grad():
        for batch in dataloader:
            x_norm = batch.to(device)
            x_recon_norm, mu, logvar = model(x_norm)
            
            # Store normalized versions
            all_original_norm.append(x_norm.cpu().numpy())
            all_reconstructed_norm.append(x_recon_norm.cpu().numpy())
            
            # Denormalize to original space
            x_orig = x_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
            x_recon_orig = x_recon_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
                
            all_original.append(x_orig)
            all_reconstructed.append(x_recon_orig)
            all_latent.append(mu.cpu().numpy())
    
    # Concatenate all batches
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    original_norm = np.concatenate(all_original_norm, axis=0)
    reconstructed_norm = np.concatenate(all_reconstructed_norm, axis=0)
    latent = np.concatenate(all_latent, axis=0)
    
    # Debug: Check data ranges
    print(f"Original data range: {original.min():.3f} to {original.max():.3f}")
    print(f"Reconstructed data range: {reconstructed.min():.3f} to {reconstructed.max():.3f}")
    print(f"Normalized original data range: {original_norm.min():.3f} to {original_norm.max():.3f}")
    print(f"Normalized reconstructed data range: {reconstructed_norm.min():.3f} to {reconstructed_norm.max():.3f}")
    print(f"Original shape: {original.shape}")
    
    # Compute sample-wise metrics (average across features for each sample)
    sample_mae = np.mean(np.abs(original - reconstructed), axis=1)  # Shape: (n_samples,)
    
    # Compute feature-wise R² using sklearn (for each feature across all samples)
    feature_r2 = []
    for j in range(original.shape[1]):
        y_true = original[:, j]  # True values for feature j across all samples
        y_pred = reconstructed[:, j]  # Predicted values for feature j across all samples
        
        # Check for valid data
        if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)) and np.var(y_true) > 1e-10:
            try:
                r2 = r2_score(y_true, y_pred)
                feature_r2.append(r2)
            except:
                feature_r2.append(np.nan)
        else:
            feature_r2.append(np.nan)
    
    feature_r2 = np.array(feature_r2)
    
    # Debug: Print computed statistics
    print(f"Sample MAE stats: mean={np.nanmean(sample_mae):.6f}, std={np.nanstd(sample_mae):.6f}")
    print(f"Feature R² stats: mean={np.nanmean(feature_r2):.6f}, std={np.nanstd(feature_r2):.6f}")
    print(f"Feature R² range: {np.nanmin(feature_r2):.3f} to {np.nanmax(feature_r2):.3f}")
    print(f"Sample MAE range: {np.nanmin(sample_mae):.3f} to {np.nanmax(sample_mae):.3f}")
    
    # Additional robust statistics for both MAE and R²
    sample_mae_clean = sample_mae[~np.isnan(sample_mae)]
    feature_r2_clean = feature_r2[~np.isnan(feature_r2)]
    
    # MAE outlier analysis
    mae_extreme_outliers = np.sum(sample_mae_clean > np.percentile(sample_mae_clean, 95) + 3 * (np.percentile(sample_mae_clean, 95) - np.percentile(sample_mae_clean, 5)))
    
    print(f"\n📊 Sample MAE Distribution Analysis:")
    print(f"   Median MAE: {np.median(sample_mae_clean):.6f}")
    print(f"   75th percentile MAE: {np.percentile(sample_mae_clean, 75):.6f}")
    print(f"   90th percentile MAE: {np.percentile(sample_mae_clean, 90):.6f}")
    print(f"   95th percentile MAE: {np.percentile(sample_mae_clean, 95):.6f}")
    print(f"   Samples with very high MAE (extreme outliers): {mae_extreme_outliers} ({mae_extreme_outliers/len(sample_mae_clean)*100:.1f}%)")
    print(f"   Samples with MAE < 1.0: {np.sum(sample_mae_clean < 1.0)} ({np.sum(sample_mae_clean < 1.0)/len(sample_mae_clean)*100:.1f}%)")
    print(f"   Samples with MAE < 0.5: {np.sum(sample_mae_clean < 0.5)} ({np.sum(sample_mae_clean < 0.5)/len(sample_mae_clean)*100:.1f}%)")
    
    # R² outlier analysis
    extreme_negative_count = np.sum(feature_r2_clean < -10)
    print(f"\n📊 Feature R² Distribution Analysis:")
    print(f"   Median R²: {np.median(feature_r2_clean):.3f}")
    print(f"   75th percentile R²: {np.percentile(feature_r2_clean, 75):.3f}")
    print(f"   90th percentile R²: {np.percentile(feature_r2_clean, 90):.3f}")
    print(f"   95th percentile R²: {np.percentile(feature_r2_clean, 95):.3f}")
    print(f"   Features with R² < -10: {extreme_negative_count} ({extreme_negative_count/len(feature_r2_clean)*100:.1f}%)")
    print(f"   Features with R² < -100: {np.sum(feature_r2_clean < -100)} ({np.sum(feature_r2_clean < -100)/len(feature_r2_clean)*100:.1f}%)")
    print(f"   Features with R² > 0.5: {np.sum(feature_r2_clean > 0.5)} ({np.sum(feature_r2_clean > 0.5)/len(feature_r2_clean)*100:.1f}%)")
    
    metrics = {
        'original': original,
        'reconstructed': reconstructed,
        'original_norm': original_norm,
        'reconstructed_norm': reconstructed_norm,
        'latent': latent,
        'sample_mae': sample_mae,
        'feature_r2': feature_r2,
        'mean_sample_mae': np.nanmean(sample_mae),
        'median_sample_mae': np.median(sample_mae_clean),
        'percentile_75_sample_mae': np.percentile(sample_mae_clean, 75),
        'percentile_95_sample_mae': np.percentile(sample_mae_clean, 95),
        'mae_extreme_outliers_count': mae_extreme_outliers,
        'mean_feature_r2': np.nanmean(feature_r2),
        'median_feature_r2': np.median(feature_r2_clean),
        'percentile_75_feature_r2': np.percentile(feature_r2_clean, 75),
        'std_sample_mae': np.nanstd(sample_mae),
        'std_feature_r2': np.nanstd(feature_r2),
        'extreme_negative_r2_count': extreme_negative_count,
    }
    
    return metrics


def plot_reconstruction_quality(metrics: dict, save_path: str = None):
    """Reconstruction quality plots: Sample MAE (zoomed) and Feature R² distribution (zoomed)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sample-wise MAE distribution (ZOOMED to 0-50 range with bin size 1)
    mae_in_range = metrics['sample_mae'][(metrics['sample_mae'] >= 0) & (metrics['sample_mae'] <= 50)]
    total_samples = len(metrics['sample_mae'])
    shown_samples = len(mae_in_range)
    
    axes[0, 0].hist(mae_in_range, bins=range(0, 51, 1), alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(metrics['median_sample_mae'], color='green', linestyle='--', linewidth=2,
                       label=f'Median: {metrics["median_sample_mae"]:.4f}')
    axes[0, 0].axvline(metrics['percentile_75_sample_mae'], color='orange', linestyle='--', linewidth=2,
                       label=f'75th %ile: {metrics["percentile_75_sample_mae"]:.4f}')
    axes[0, 0].set_xlabel('Sample MAE (Original Space)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Sample-wise MAE Distribution\n({shown_samples:,} of {total_samples:,} samples shown, range 0-50)')
    axes[0, 0].set_xlim(0, 50)  # Zoom to 0-50 range
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature-wise R² distribution (ZOOMED to -1 to 1 range)
    valid_feature_r2 = metrics['feature_r2'][~np.isnan(metrics['feature_r2'])]
    r2_in_range = valid_feature_r2[(valid_feature_r2 >= -1) & (valid_feature_r2 <= 1)]
    total_features = len(valid_feature_r2)
    shown_features = len(r2_in_range)
    
    axes[0, 1].hist(r2_in_range, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].axvline(np.median(r2_in_range), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(r2_in_range):.3f}')
    axes[0, 1].axvline(np.percentile(r2_in_range, 75), color='orange', linestyle='--', linewidth=2,
                       label=f'75th %ile: {np.percentile(r2_in_range, 75):.3f}')
    axes[0, 1].set_xlabel('Feature R² (Original Space)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Feature R² Distribution\n({shown_features:,} of {total_features:,} features shown, range -1 to 1)')
    axes[0, 1].set_xlim(-1, 1)  # Zoom to -1 to 1 range
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reconstruction density plot - using normalized features
    orig_norm_flat = metrics['original_norm'].flatten()
    recon_norm_flat = metrics['reconstructed_norm'].flatten()
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(orig_norm_flat) & np.isfinite(recon_norm_flat)
    orig_norm_flat = orig_norm_flat[valid_mask]
    recon_norm_flat = recon_norm_flat[valid_mask]
    
    # Use all points, not subsampled - create density plot with log scale
    hb = axes[1, 0].hexbin(orig_norm_flat, recon_norm_flat, gridsize=75, cmap='viridis', 
                           mincnt=1, norm=LogNorm())
    axes[1, 0].plot([orig_norm_flat.min(), orig_norm_flat.max()], 
                    [orig_norm_flat.min(), orig_norm_flat.max()], 
                    'r--', linewidth=2, label='Perfect Reconstruction')
    axes[1, 0].set_xlabel('Original Normalized Feature Values')
    axes[1, 0].set_ylabel('Reconstructed Normalized Feature Values')
    axes[1, 0].set_title(f'Reconstruction Density Plot (Normalized Space)\n({len(orig_norm_flat):,} points)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add colorbar with log scale
    cbar = plt.colorbar(hb, ax=axes[1, 0])
    cbar.set_label('Point Density (log scale)', rotation=270, labelpad=15)
    
    # 4. Remove the bottom-right plot (keep it empty)
    axes[1, 1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction quality plots saved to: {save_path}")
    
    plt.show()


def analyze_latent_space(metrics: dict, save_path: str = None):
    """Analyze the learned latent space with percentile-based reconstruction error coloring."""
    latent = metrics['latent']
    sample_mae = metrics['sample_mae']
    
    # Convert MAE to percentiles for better color distribution (handles outliers)
    mae_percentiles = (rankdata(sample_mae) - 1) / (len(sample_mae) - 1) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA on latent space colored by MAE percentiles
    pca = PCA(n_components=2, random_state=42)
    latent_pca = pca.fit_transform(latent)
    
    scatter1 = axes[0].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                              c=mae_percentiles, cmap='viridis', alpha=0.7, s=20)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[0].set_title('PCA - Colored by MAE Percentile')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('MAE Percentile (%)', rotation=270, labelpad=15)
    
    # t-SNE on latent space colored by MAE percentiles (subsample for efficiency)
    if len(latent) > 5000:
        idx = np.random.choice(len(latent), 5000, replace=False)
        latent_subset = latent[idx]
        mae_percentiles_subset = mae_percentiles[idx]
    else:
        latent_subset = latent
        mae_percentiles_subset = mae_percentiles
        idx = np.arange(len(latent))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_subset)-1))
    latent_tsne = tsne.fit_transform(latent_subset)
    
    scatter2 = axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                              c=mae_percentiles_subset, cmap='viridis', alpha=0.7, s=20)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE - Colored by MAE Percentile')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('MAE Percentile (%)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simplified latent space analysis plot saved to: {save_path}")
    
    plt.show()
    
    # Print percentile information for context
    print(f"\n📊 MAE Percentile Analysis:")
    print(f"   50th percentile (median): {np.percentile(sample_mae, 50):.6f}")
    print(f"   75th percentile: {np.percentile(sample_mae, 75):.6f}")
    print(f"   90th percentile: {np.percentile(sample_mae, 90):.6f}")
    print(f"   95th percentile: {np.percentile(sample_mae, 95):.6f}")
    print(f"   99th percentile: {np.percentile(sample_mae, 99):.6f}")
    
    return {
        'pca_explained_variance': pca.explained_variance_ratio_[:2],
        'mae_percentile_50': np.percentile(sample_mae, 50),
        'mae_percentile_75': np.percentile(sample_mae, 75),
        'mae_percentile_95': np.percentile(sample_mae, 95),
    }


def generate_report(model: JUMPVAE, metrics: dict, latent_analysis: dict, save_path: str = None):
    """Generate a simplified evaluation report focusing on meaningful metrics."""
    
    # Calculate R² quality categories
    feature_r2_clean = metrics['feature_r2'][~np.isnan(metrics['feature_r2'])]
    excellent_r2 = np.sum(feature_r2_clean > 0.9)
    good_r2 = np.sum(feature_r2_clean > 0.8)
    poor_r2 = np.sum(feature_r2_clean < 0.5)
    
    # Calculate robust statistics (less sensitive to outliers)
    median_r2 = np.median(feature_r2_clean)
    percentile_75_r2 = np.percentile(feature_r2_clean, 75)
    extreme_negative_count = np.sum(feature_r2_clean < -10)
    
    report = f"""
# Simplified VAE Model Evaluation Report

## Model Architecture
- Input Dimension: {model.hparams.input_dim}
- Latent Dimension: {model.hparams.latent_dim}
- Encoder Hidden Dims: {model.hparams.encoder_hidden_dims}
- Decoder Hidden Dims: {model.hparams.decoder_hidden_dims}
- Dropout: {model.hparams.dropout}
- Normalization: {model.hparams.norm_type}
- Beta (KL weight): {model.hparams.beta}

## Reconstruction Metrics (Original Space)

### Sample-wise Performance
- Mean MAE: {metrics['mean_sample_mae']:.6f} ± {metrics['std_sample_mae']:.6f}
- **Median MAE (robust)**: {metrics['median_sample_mae']:.6f}
- **75th percentile MAE**: {metrics['percentile_75_sample_mae']:.6f}
- Number of samples: {len(metrics['sample_mae']):,}

### Feature-wise Performance  
- Mean R²: {metrics['mean_feature_r2']:.6f} ± {metrics['std_feature_r2']:.6f}
- **Median R² (robust)**: {median_r2:.3f}
- **75th percentile R²**: {percentile_75_r2:.3f}
- Number of features: {len(metrics['feature_r2']):,}

## Feature Reconstruction Quality (R² Analysis)
- Features with excellent reconstruction (R² > 0.9): {excellent_r2} / {len(feature_r2_clean)} ({excellent_r2/len(feature_r2_clean)*100:.1f}%)
- Features with good reconstruction (R² > 0.8): {good_r2} / {len(feature_r2_clean)} ({good_r2/len(feature_r2_clean)*100:.1f}%)
- Features with poor reconstruction (R² < 0.5): {poor_r2} / {len(feature_r2_clean)} ({poor_r2/len(feature_r2_clean)*100:.1f}%)

### ⚠️ Mean vs. Median Discrepancy Analysis

#### MAE Outlier Analysis
- **Samples with extreme high MAE**: {metrics['mae_extreme_outliers_count']} / {len(metrics['sample_mae'])} ({metrics['mae_extreme_outliers_count']/len(metrics['sample_mae'])*100:.1f}%)
- **Recommendation**: Focus on median MAE ({metrics['median_sample_mae']:.6f}) as it's more representative of typical performance

#### R² Outlier Analysis
- **Features with extreme negative R² (< -10)**: {extreme_negative_count} / {len(feature_r2_clean)} ({extreme_negative_count/len(feature_r2_clean)*100:.1f}%)
- **Note**: The very negative mean R² is caused by {extreme_negative_count} features with extreme outlier values
- **Recommendation**: Focus on median R² ({median_r2:.3f}) as it's more representative of typical performance

## Latent Space Analysis
- PCA explained variance (PC1, PC2): {latent_analysis['pca_explained_variance'][0]:.4f}, {latent_analysis['pca_explained_variance'][1]:.4f}
- MAE percentiles for color scale:
  - 50th percentile (median): {latent_analysis['mae_percentile_50']:.6f}
  - 75th percentile: {latent_analysis['mae_percentile_75']:.6f}
  - 95th percentile: {latent_analysis['mae_percentile_95']:.6f}

## Model Parameters
- Total parameters: {sum(p.numel() for p in model.parameters()):,}
- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

## Performance Assessment
"""
    
    # Add performance-based recommendations (using robust median)
    median_mae = metrics['median_sample_mae']
    if median_mae < 1.0:
        report += f"- Low reconstruction error in original space (Median MAE = {median_mae:.6f} < 1.0)\n"
    elif median_mae < 5.0:
        report += f"- Moderate reconstruction error in original space (Median MAE = {median_mae:.6f} < 5.0)\n"
    else:
        report += f"- High reconstruction error - model may need more capacity or training (Median MAE = {median_mae:.6f})\n"
    
    # Use median R² for performance assessment (more robust than mean)
    if median_r2 > 0.8:
        report += f"- Excellent feature reconstruction quality (Median R² = {median_r2:.3f} > 0.8)\n"
    elif median_r2 > 0.6:
        report += f"- Good feature reconstruction quality (Median R² = {median_r2:.3f} > 0.6)\n"
    elif median_r2 > 0.4:
        report += f"- Moderate feature reconstruction quality (Median R² = {median_r2:.3f} > 0.4)\n"
    else:
        report += f"- Poor feature reconstruction (Median R² = {median_r2:.3f}) - consider model improvements\n"
    
    # Additional notes about mean vs median discrepancies
    if metrics['mae_extreme_outliers_count'] > 0:
        report += f"- **Note**: Mean MAE ({metrics['mean_sample_mae']:.6f}) is misleading due to {metrics['mae_extreme_outliers_count']} extreme outliers\n"
    
    if extreme_negative_count > 0:
        report += f"- **Note**: Mean R² ({metrics['mean_feature_r2']:.3f}) is misleading due to {extreme_negative_count} extreme outliers\n"
    
    if excellent_r2 / len(feature_r2_clean) > 0.8:
        report += "- Most features reconstructed excellently (>80% with R² > 0.9)\n"
    elif excellent_r2 / len(feature_r2_clean) > 0.6:
        report += "- Many features reconstructed well (>60% with R² > 0.9)\n"
    else:
        report += "- Feature reconstruction quality varies significantly\n"
    
    # Simplified latent space assessment based on percentile distribution
    mae_range = latent_analysis['mae_percentile_95'] - latent_analysis['mae_percentile_50']
    if mae_range < latent_analysis['mae_percentile_50']:
        report += "- Good reconstruction consistency across latent space (narrow MAE range)\n"
    else:
        report += "- Reconstruction quality varies significantly across latent space\n"
    
    # PCA variance assessment
    total_pca_variance = sum(latent_analysis['pca_explained_variance'])
    if total_pca_variance > 0.5:
        report += f"- First 2 PCA components capture {total_pca_variance:.1%} of latent variance\n"
    else:
        report += f"- Latent space is high-dimensional (first 2 PCs: {total_pca_variance:.1%})\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Simplified evaluation report saved to: {save_path}")
    
    print(report)


def load_saved_normalization_params(checkpoint_path: str, config_experiment_name: str, log_dir: str):
    """Load saved normalization parameters to avoid recomputing from huge dataset."""
    # Try to find normalization file in the same directory as checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    experiment_dir = os.path.dirname(checkpoint_dir)  # Go up from checkpoints/ to experiment/
    normalization_path = os.path.join(experiment_dir, 'normalization_params.npz')
    
    # Fallback: try constructing path from log_dir and experiment name
    if not os.path.exists(normalization_path):
        normalization_path = os.path.join(log_dir, config_experiment_name, 'normalization_params.npz')
    
    if os.path.exists(normalization_path):
        print(f"📂 Loading saved normalization parameters from: {normalization_path}")
        norm_data = np.load(normalization_path)
        
        params = {
            'feature_mean': norm_data['feature_mean'],
            'feature_std': norm_data['feature_std'],
            'feature_dim': int(norm_data['feature_dim']),
            'feature_cols': norm_data['feature_cols'].tolist()
        }
        
        print(f"✅ Loaded normalization parameters:")
        print(f"   Feature mean shape: {params['feature_mean'].shape}")
        print(f"   Feature std shape: {params['feature_std'].shape}")
        print(f"   Feature dimension: {params['feature_dim']}")
        
        return params
    else:
        print(f"⚠️  Normalization file not found at: {normalization_path}")
        print(f"   Will fall back to computing from dataset (slower)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE model")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file used for training")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from: {args.config}")
    from train_vae import load_config, JUMPDataModule
    config = load_config(args.config)
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint_path)
    print(f"Model loaded successfully")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Try to load saved normalization parameters (FAST PATH)
    saved_norm_params = load_saved_normalization_params(
        args.checkpoint_path, 
        config.logging['experiment_name'], 
        config.logging['log_dir']
    )
    
    if saved_norm_params is not None:
        # FAST PATH: Use saved normalization parameters without loading huge dataset
        print("🚀 Using saved normalization parameters (fast path)")
        
        # Create a minimal test-only data module
        print("Setting up test data only...")
        data_module = JUMPDataModule(config)
        data_module.setup()
        test_dataloader = data_module.test_dataloader()
        
        # Override normalization parameters in test dataset
        test_dataset = test_dataloader.dataset
        if hasattr(test_dataset, 'dataset'):
            # Handle subset datasets (likely case)
            test_dataset.dataset.feature_mean = saved_norm_params['feature_mean']
            test_dataset.dataset.feature_std = saved_norm_params['feature_std']
            print("✅ Applied saved normalization parameters to test subset dataset")
        elif hasattr(test_dataset, 'feature_mean'):
            # Direct dataset
            test_dataset.feature_mean = saved_norm_params['feature_mean']
            test_dataset.feature_std = saved_norm_params['feature_std']
            print("✅ Applied saved normalization parameters to test dataset")
        
        print(f"Test set size: {len(test_dataloader.dataset)}")
        
    else:
        # SLOW PATH: Fallback to computing normalization from full dataset
        print("⚠️  Using slow path: computing normalization from full dataset")
        print("Setting up data...")
        data_module = JUMPDataModule(config)
        data_module.setup()
        
        # Use test set for evaluation
        test_dataloader = data_module.test_dataloader()
        print(f"Test set size: {len(test_dataloader.dataset)}")
    
    # Compute metrics
    print("Computing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(model, test_dataloader, args.device)
    
    # Generate plots
    print("Generating reconstruction quality plots...")
    recon_plot_path = os.path.join(args.output_dir, "reconstruction_quality.png")
    plot_reconstruction_quality(metrics, recon_plot_path)
    
    print("Analyzing latent space...")
    latent_plot_path = os.path.join(args.output_dir, "latent_space_analysis.png")
    latent_analysis = analyze_latent_space(metrics, latent_plot_path)
    
    # Generate report
    print("Generating evaluation report...")
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    generate_report(model, metrics, latent_analysis, report_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.npz")
    save_dict = {
        'latent_representations': metrics['latent'],
        'sample_mae': metrics['sample_mae'],
        'feature_r2': metrics['feature_r2'],
    }
    
    np.savez(metrics_path, **save_dict)
    print(f"Simplified metrics saved to: {metrics_path}")
    
    print(f"Evaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 