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
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score
from sklearn.cluster import KMeans
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
            
            # Denormalize to original space
            x_orig = x_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
            x_recon_orig = x_recon_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
                
            all_original.append(x_orig)
            all_reconstructed.append(x_recon_orig)
            all_latent.append(mu.cpu().numpy())
    
    # Concatenate all batches
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    latent = np.concatenate(all_latent, axis=0)
    
    # Debug: Check data ranges
    print(f"Original data range: {original.min():.3f} to {original.max():.3f}")
    print(f"Reconstructed data range: {reconstructed.min():.3f} to {reconstructed.max():.3f}")
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
    
    # Additional R² statistics to understand the distribution
    feature_r2_clean = feature_r2[~np.isnan(feature_r2)]
    extreme_negative_count = np.sum(feature_r2_clean < -10)
    print(f"\n📊 R² Distribution Analysis:")
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
        'latent': latent,
        'sample_mae': sample_mae,
        'feature_r2': feature_r2,
        'mean_sample_mae': np.nanmean(sample_mae),
        'mean_feature_r2': np.nanmean(feature_r2),
        'median_feature_r2': np.median(feature_r2_clean),
        'percentile_75_feature_r2': np.percentile(feature_r2_clean, 75),
        'std_sample_mae': np.nanstd(sample_mae),
        'std_feature_r2': np.nanstd(feature_r2),
        'extreme_negative_r2_count': extreme_negative_count,
    }
    
    return metrics


def plot_reconstruction_quality(metrics: dict, save_path: str = None):
    """Reconstruction quality plots: Sample MAE (zoomed), Feature R² distribution (zoomed), and reconstruction scatterplot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sample-wise MAE distribution (ZOOMED to 0-50 range)
    axes[0, 0].hist(metrics['sample_mae'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(metrics['mean_sample_mae'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {metrics["mean_sample_mae"]:.4f}')
    axes[0, 0].set_xlabel('Sample MAE (Original Space)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sample-wise MAE Distribution (Zoomed 0-50)')
    axes[0, 0].set_xlim(0, 50)  # Zoom to 0-50 range
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature-wise R² distribution (ZOOMED to reasonable range) - moved from position [1,0]
    valid_feature_r2 = metrics['feature_r2'][~np.isnan(metrics['feature_r2'])]
    reasonable_r2 = valid_feature_r2[valid_feature_r2 > -10]  # Remove extreme outliers
    axes[0, 1].hist(reasonable_r2, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].axvline(np.median(reasonable_r2), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(reasonable_r2):.3f}')
    axes[0, 1].axvline(np.percentile(reasonable_r2, 75), color='orange', linestyle='--', linewidth=2,
                       label=f'75th %ile: {np.percentile(reasonable_r2, 75):.3f}')
    axes[0, 1].set_xlabel('Feature R² (Original Space)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Feature R² Distribution ({len(reasonable_r2)}/{len(valid_feature_r2)} features, R² > -10)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reconstruction scatterplot - subsample for visualization
    orig_flat = metrics['original'].flatten()
    recon_flat = metrics['reconstructed'].flatten()
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(orig_flat) & np.isfinite(recon_flat)
    orig_flat = orig_flat[valid_mask]
    recon_flat = recon_flat[valid_mask]
    
    # Subsample for better visualization (use every 100th point if dataset is large)
    if len(orig_flat) > 10000:
        subsample_idx = np.arange(0, len(orig_flat), len(orig_flat)//10000)
        orig_sample = orig_flat[subsample_idx]
        recon_sample = recon_flat[subsample_idx]
    else:
        orig_sample = orig_flat
        recon_sample = recon_flat
    
    axes[1, 0].scatter(orig_sample, recon_sample, alpha=0.5, s=1, color='blue')
    axes[1, 0].plot([orig_sample.min(), orig_sample.max()], [orig_sample.min(), orig_sample.max()], 
                    'r--', linewidth=2, label='Perfect Reconstruction')
    axes[1, 0].set_xlabel('Original Feature Values')
    axes[1, 0].set_ylabel('Reconstructed Feature Values')
    axes[1, 0].set_title(f'Reconstruction Scatterplot\n(Subsample: {len(orig_sample):,} points)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature correlation plot - show correlation between original and reconstructed per feature
    feature_corr = []
    for j in range(metrics['original'].shape[1]):
        orig_feature = metrics['original'][:, j]
        recon_feature = metrics['reconstructed'][:, j]
        if np.var(orig_feature) > 1e-10 and np.var(recon_feature) > 1e-10:
            corr = np.corrcoef(orig_feature, recon_feature)[0, 1]
            if np.isfinite(corr):
                feature_corr.append(corr)
    
    feature_corr = np.array(feature_corr)
    axes[1, 1].hist(feature_corr, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 1].axvline(np.mean(feature_corr), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(feature_corr):.3f}')
    axes[1, 1].axvline(np.median(feature_corr), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(feature_corr):.3f}')
    axes[1, 1].set_xlabel('Feature Correlation (Original vs Reconstructed)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Feature-wise Correlation Distribution\n({len(feature_corr)} features)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction quality plots saved to: {save_path}")
    
    plt.show()


def analyze_latent_space(metrics: dict, save_path: str = None):
    """Analyze the learned latent space with reconstruction error coloring."""
    latent = metrics['latent']
    sample_mae = metrics['sample_mae']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA on latent space colored by reconstruction error
    pca = PCA(n_components=2, random_state=42)
    latent_pca = pca.fit_transform(latent)
    
    scatter1 = axes[0, 0].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                 c=sample_mae, cmap='viridis', alpha=0.7, s=20)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].set_title('PCA - Colored by Reconstruction Error')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Sample MAE')
    
    # t-SNE on latent space colored by reconstruction error (subsample for efficiency)
    if len(latent) > 5000:
        idx = np.random.choice(len(latent), 5000, replace=False)
        latent_subset = latent[idx]
        mae_subset = sample_mae[idx]
    else:
        latent_subset = latent
        mae_subset = sample_mae
        idx = np.arange(len(latent))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_subset)-1))
    latent_tsne = tsne.fit_transform(latent_subset)
    
    scatter2 = axes[0, 1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                                 c=mae_subset, cmap='viridis', alpha=0.7, s=20)
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].set_title('t-SNE - Colored by Reconstruction Error')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Sample MAE')
    
    # Latent dimension statistics
    latent_means = np.mean(latent, axis=0)
    latent_stds = np.std(latent, axis=0)
    
    axes[0, 2].plot(latent_means, label='Mean', color='blue', linewidth=2)
    axes[0, 2].plot(latent_stds, label='Std', color='orange', linewidth=2)
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].set_title('Latent Dimension Statistics')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Reconstruction error vs latent magnitude
    latent_norms = np.linalg.norm(latent, axis=1)
    axes[1, 0].scatter(latent_norms, sample_mae, alpha=0.6, s=10)
    axes[1, 0].set_xlabel('Latent Vector Magnitude')
    axes[1, 0].set_ylabel('Sample MAE')
    axes[1, 0].set_title('Reconstruction Error vs Latent Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clustering analysis
    n_clusters_range = range(2, min(11, len(latent)//10 + 2))  # Ensure reasonable cluster range
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        if n_clusters < len(latent):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(latent)
            silhouette_avg = silhouette_score(latent, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    if silhouette_scores:
        axes[1, 1].plot(n_clusters_range, silhouette_scores, 'o-', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].set_title('Clustering Quality in Latent Space')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Best clustering result
        best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        best_silhouette_score = max(silhouette_scores)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for clustering', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        best_n_clusters = 2
        best_silhouette_score = 0
    
    # Latent space distribution (show actual values distribution)
    latent_flat = latent.flatten()
    axes[1, 2].hist(latent_flat, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 2].axvline(np.mean(latent_flat), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(latent_flat):.3f}')
    axes[1, 2].set_xlabel('Latent Values')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Latent Space Value Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clean latent space analysis plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'best_n_clusters': best_n_clusters,
        'best_silhouette_score': best_silhouette_score,
        'pca_explained_variance': pca.explained_variance_ratio_[:2],
        'latent_magnitude_correlation': np.corrcoef(latent_norms, sample_mae)[0, 1] if len(latent_norms) > 1 else 0
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
- **Features with extreme negative R² (< -10)**: {extreme_negative_count} / {len(feature_r2_clean)} ({extreme_negative_count/len(feature_r2_clean)*100:.1f}%)
- **Note**: The very negative mean R² is caused by {extreme_negative_count} features with extreme outlier values
- **Recommendation**: Focus on median R² ({median_r2:.3f}) as it's more representative of typical performance

## Latent Space Analysis
- Best number of clusters: {latent_analysis['best_n_clusters']}
- Best silhouette score: {latent_analysis['best_silhouette_score']:.4f}
- PCA explained variance (PC1, PC2): {latent_analysis['pca_explained_variance'][0]:.4f}, {latent_analysis['pca_explained_variance'][1]:.4f}
- Latent magnitude vs reconstruction error correlation: {latent_analysis['latent_magnitude_correlation']:.4f}

## Model Parameters
- Total parameters: {sum(p.numel() for p in model.parameters()):,}
- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

## Performance Assessment
"""
    
    # Add performance-based recommendations
    if metrics['mean_sample_mae'] < 1.0:
        report += "- Low reconstruction error in original space\n"
    elif metrics['mean_sample_mae'] < 5.0:
        report += "- Moderate reconstruction error in original space\n"
    else:
        report += "- High reconstruction error - model may need more capacity or training\n"
    
    # Use median R² for performance assessment (more robust than mean)
    if median_r2 > 0.8:
        report += f"- Excellent feature reconstruction quality (Median R² = {median_r2:.3f} > 0.8)\n"
    elif median_r2 > 0.6:
        report += f"- Good feature reconstruction quality (Median R² = {median_r2:.3f} > 0.6)\n"
    elif median_r2 > 0.4:
        report += f"- Moderate feature reconstruction quality (Median R² = {median_r2:.3f} > 0.4)\n"
    else:
        report += f"- Poor feature reconstruction (Median R² = {median_r2:.3f}) - consider model improvements\n"
    
    # Additional note about mean vs median
    if extreme_negative_count > 0:
        report += f"- **Note**: Mean R² ({metrics['mean_feature_r2']:.3f}) is misleading due to {extreme_negative_count} extreme outliers\n"
    
    if excellent_r2 / len(feature_r2_clean) > 0.8:
        report += "- Most features reconstructed excellently (>80% with R² > 0.9)\n"
    elif excellent_r2 / len(feature_r2_clean) > 0.6:
        report += "- Many features reconstructed well (>60% with R² > 0.9)\n"
    else:
        report += "- Feature reconstruction quality varies significantly\n"
    
    if latent_analysis['best_silhouette_score'] > 0.5:
        report += "- Latent space shows good clustering structure\n"
    elif latent_analysis['best_silhouette_score'] > 0.2:
        report += "- Latent space shows moderate clustering structure\n"
    else:
        report += "- Latent space shows poor clustering - consider different architectures\n"
    
    abs_correlation = abs(latent_analysis['latent_magnitude_correlation'])
    if abs_correlation > 0.5:
        report += f"- Strong correlation ({latent_analysis['latent_magnitude_correlation']:.3f}) between latent magnitude and reconstruction error\n"
    elif abs_correlation > 0.2:
        report += f"- Moderate correlation ({latent_analysis['latent_magnitude_correlation']:.3f}) between latent magnitude and reconstruction error\n"
    else:
        report += "- Weak correlation between latent magnitude and reconstruction error\n"
    
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