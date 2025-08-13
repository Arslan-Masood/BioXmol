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
from sklearn.metrics import silhouette_score
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
    """Compute reconstruction metrics for the entire dataset in both normalized and original space."""
    model.to(device)
    model.eval()
    
    all_mse_norm = []
    all_mae_norm = []
    all_mse_orig = []
    all_mae_orig = []
    all_original_norm = []
    all_reconstructed_norm = []
    all_original_orig = []
    all_reconstructed_orig = []
    all_latent = []
    
    # Get normalization parameters from dataset if available
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
        print("Warning: Could not find normalization parameters. Computing metrics only in model space.")
    
    with torch.no_grad():
        for batch in dataloader:
            x_norm = batch.to(device)  # Use the complete batch tensor
            x_recon_norm, mu, logvar = model(x_norm)  # Normalized reconstruction
            
            # Compute metrics in normalized space
            mse_norm = torch.mean((x_norm - x_recon_norm) ** 2, dim=1)
            mae_norm = torch.mean(torch.abs(x_norm - x_recon_norm), dim=1)
            
            all_mse_norm.extend(mse_norm.cpu().numpy())
            all_mae_norm.extend(mae_norm.cpu().numpy())
            all_original_norm.append(x_norm.cpu().numpy())
            all_reconstructed_norm.append(x_recon_norm.cpu().numpy())
            all_latent.append(mu.cpu().numpy())
            
            # Compute metrics in original space if normalization info available
            if has_normalization:
                # Denormalize to original space
                x_orig = x_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
                x_recon_orig = x_recon_norm.cpu().numpy() * feature_std.flatten() + feature_mean.flatten()
                
                # Compute metrics in original space
                mse_orig = np.mean((x_orig - x_recon_orig) ** 2, axis=1)
                mae_orig = np.mean(np.abs(x_orig - x_recon_orig), axis=1)
                
                all_mse_orig.extend(mse_orig)
                all_mae_orig.extend(mae_orig)
                all_original_orig.append(x_orig)
                all_reconstructed_orig.append(x_recon_orig)
    
    # Concatenate all batches
    all_original_norm = np.concatenate(all_original_norm, axis=0)
    all_reconstructed_norm = np.concatenate(all_reconstructed_norm, axis=0)
    all_latent = np.concatenate(all_latent, axis=0)
    
    metrics = {
        # Normalized space metrics
        'mse_per_sample_norm': np.array(all_mse_norm),
        'mae_per_sample_norm': np.array(all_mae_norm),
        'mean_mse_norm': np.mean(all_mse_norm),
        'mean_mae_norm': np.mean(all_mae_norm),
        'std_mse_norm': np.std(all_mse_norm),
        'std_mae_norm': np.std(all_mae_norm),
        'original_norm': all_original_norm,
        'reconstructed_norm': all_reconstructed_norm,
        'latent': all_latent,
        'has_original_space': has_normalization
    }
    
    # Add original space metrics if available
    if has_normalization:
        all_original_orig = np.concatenate(all_original_orig, axis=0)
        all_reconstructed_orig = np.concatenate(all_reconstructed_orig, axis=0)
        
        metrics.update({
            'mse_per_sample_orig': np.array(all_mse_orig),
            'mae_per_sample_orig': np.array(all_mae_orig),
            'mean_mse_orig': np.mean(all_mse_orig),
            'mean_mae_orig': np.mean(all_mae_orig),
            'std_mse_orig': np.std(all_mse_orig),
            'std_mae_orig': np.std(all_mae_orig),
            'original_orig': all_original_orig,
            'reconstructed_orig': all_reconstructed_orig,
        })
    
    return metrics


def plot_reconstruction_quality(metrics: dict, save_path: str = None):
    """Plot reconstruction quality metrics in both normalized and original space."""
    # Determine number of subplots based on available metrics
    if metrics['has_original_space']:
        fig, axes = plt.subplots(4, 3, figsize=(21, 24))  # 4 rows: norm metrics, norm features, orig metrics, orig features
        spaces = [('norm', 'Normalized Space'), ('orig', 'Original Space')]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows: metrics, features
        spaces = [('norm', 'Model Space')]
    
    for space_idx, (space_suffix, space_name) in enumerate(spaces):
        row_offset = space_idx * 2 if metrics['has_original_space'] else 0
        
        mae_key = f'mae_per_sample_{space_suffix}'
        mse_key = f'mse_per_sample_{space_suffix}'
        mean_mae_key = f'mean_mae_{space_suffix}'
        mean_mse_key = f'mean_mse_{space_suffix}'
        original_key = f'original_{space_suffix}'
        reconstructed_key = f'reconstructed_{space_suffix}'
    
    # MAE histogram
        axes[row_offset, 0].hist(metrics[mae_key], bins=50, alpha=0.7, edgecolor='black')
        axes[row_offset, 0].axvline(metrics[mean_mae_key], color='red', linestyle='--', 
                          label=f'Mean: {metrics[mean_mae_key]:.4f}')
        axes[row_offset, 0].set_xlabel('MAE per Sample')
        axes[row_offset, 0].set_ylabel('Frequency')
        axes[row_offset, 0].set_title(f'MAE Distribution - {space_name}')
        axes[row_offset, 0].legend()
        axes[row_offset, 0].grid(True, alpha=0.3)
    
    # MSE histogram
        axes[row_offset, 1].hist(metrics[mse_key], bins=50, alpha=0.7, edgecolor='black')
        axes[row_offset, 1].axvline(metrics[mean_mse_key], color='red', linestyle='--', 
                          label=f'Mean: {metrics[mean_mse_key]:.4f}')
        axes[row_offset, 1].set_xlabel('MSE per Sample')
        axes[row_offset, 1].set_ylabel('Frequency')
        axes[row_offset, 1].set_title(f'MSE Distribution - {space_name}')
        axes[row_offset, 1].legend()
        axes[row_offset, 1].grid(True, alpha=0.3)
    
    # Reconstruction scatter plot (first feature)
        sample_size = min(1000, len(metrics[original_key]))
        idx = np.random.choice(len(metrics[original_key]), sample_size, replace=False)
        orig_sample = metrics[original_key][idx, 0]
        recon_sample = metrics[reconstructed_key][idx, 0]
    
        axes[row_offset, 2].scatter(orig_sample, recon_sample, alpha=0.5)
    min_val, max_val = min(orig_sample.min(), recon_sample.min()), max(orig_sample.max(), recon_sample.max())
        axes[row_offset, 2].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[row_offset, 2].set_xlabel('Original Feature 0')
        axes[row_offset, 2].set_ylabel('Reconstructed Feature 0')
        axes[row_offset, 2].set_title(f'Reconstruction Quality - {space_name}')
        axes[row_offset, 2].grid(True, alpha=0.3)
    
        # Feature-wise reconstruction error (second row for each space)
        feature_mae = np.mean(np.abs(metrics[original_key] - metrics[reconstructed_key]), axis=0)
        row_idx = row_offset + 1
        
        axes[row_idx, 0].plot(feature_mae)
        axes[row_idx, 0].set_xlabel('Feature Index')
        axes[row_idx, 0].set_ylabel('Mean Absolute Error')
        axes[row_idx, 0].set_title(f'Per-Feature Reconstruction Error - {space_name}')
        axes[row_idx, 0].grid(True, alpha=0.3)
    
    # Correlation between original and reconstructed
    correlations = []
        for i in range(metrics[original_key].shape[1]):  # All features instead of limiting to 100
            corr = np.corrcoef(metrics[original_key][:, i], metrics[reconstructed_key][:, i])[0, 1]
            if not np.isnan(corr):  # Only add valid correlations
        correlations.append(corr)
    
        axes[row_idx, 1].hist(correlations, bins=30, alpha=0.7, edgecolor='black')
        axes[row_idx, 1].axvline(np.mean(correlations), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(correlations):.3f}')
        axes[row_idx, 1].set_xlabel('Correlation Coefficient')
        axes[row_idx, 1].set_ylabel('Frequency')
        axes[row_idx, 1].set_title(f'Feature-wise Correlation - {space_name}')
        axes[row_idx, 1].legend()
        axes[row_idx, 1].grid(True, alpha=0.3)
        
        # Latent space distribution (for first space only)
        if space_idx == 0:
            latent_sample = metrics['latent'][idx, :2] if len(metrics['latent'].shape) > 1 and metrics['latent'].shape[1] >= 2 else metrics['latent'][idx, :1]
            
            if len(latent_sample.shape) > 1 and latent_sample.shape[1] >= 2:
                axes[row_idx, 2].scatter(latent_sample[:, 0], latent_sample[:, 1], alpha=0.6)
                axes[row_idx, 2].set_xlabel('Latent Dimension 0')
                axes[row_idx, 2].set_ylabel('Latent Dimension 1')
                axes[row_idx, 2].set_title('Latent Space Distribution')
                axes[row_idx, 2].grid(True, alpha=0.3)
            else:
                # Handle 1D latent case
                axes[row_idx, 2].hist(latent_sample.flatten(), bins=30, alpha=0.7, edgecolor='black')
                axes[row_idx, 2].set_xlabel('Latent Value')
                axes[row_idx, 2].set_ylabel('Frequency')
                axes[row_idx, 2].set_title('Latent Distribution (1D)')
                axes[row_idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction quality plot saved to: {save_path}")
    
    plt.show()


def analyze_latent_space(metrics: dict, save_path: str = None):
    """Analyze the learned latent space."""
    latent = metrics['latent']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA on latent space
    pca = PCA(n_components=2, random_state=42)
    latent_pca = pca.fit_transform(latent)
    
    axes[0, 0].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].set_title('PCA of Latent Space')
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE on latent space (subsample for efficiency)
    if len(latent) > 5000:
        idx = np.random.choice(len(latent), 5000, replace=False)
        latent_subset = latent[idx]
    else:
        latent_subset = latent
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_subset)
    
    axes[0, 1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6)
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].set_title('t-SNE of Latent Space')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Latent dimension statistics
    latent_means = np.mean(latent, axis=0)
    latent_stds = np.std(latent, axis=0)
    
    axes[0, 2].plot(latent_means, label='Mean')
    axes[0, 2].plot(latent_stds, label='Std')
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].set_title('Latent Dimension Statistics')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Clustering analysis
    n_clusters_range = range(2, 11)
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent)
        silhouette_avg = silhouette_score(latent, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    axes[1, 0].plot(n_clusters_range, silhouette_scores, 'o-')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Clustering Quality in Latent Space')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best clustering visualization
    best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent)
    
    scatter = axes[1, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                c=cluster_labels, alpha=0.6, cmap='tab10')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[1, 1].set_title(f'Clusters in Latent Space (k={best_n_clusters})')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # Latent space interpolation
    # Sample two random points and interpolate between them
    idx1, idx2 = np.random.choice(len(latent), 2, replace=False)
    point1, point2 = latent[idx1], latent[idx2]
    
    # Create interpolation
    alphas = np.linspace(0, 1, 10)
    interpolated_points = []
    for alpha in alphas:
        point = (1 - alpha) * point1 + alpha * point2
        interpolated_points.append(point)
    
    interpolated_points = np.array(interpolated_points)
    interpolated_pca = pca.transform(interpolated_points)
    
    axes[1, 2].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.3, c='gray', s=1)
    axes[1, 2].plot(interpolated_pca[:, 0], interpolated_pca[:, 1], 'ro-', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[1, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[1, 2].set_title('Latent Space Interpolation')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space analysis plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'best_n_clusters': best_n_clusters,
        'best_silhouette_score': max(silhouette_scores),
        'pca_explained_variance': pca.explained_variance_ratio_[:2],
        'cluster_labels': cluster_labels
    }


def generate_report(model: JUMPVAE, metrics: dict, latent_analysis: dict, save_path: str = None):
    """Generate a comprehensive evaluation report."""
    report = f"""
# VAE Model Evaluation Report

## Model Architecture
- Input Dimension: {model.hparams.input_dim}
- Latent Dimension: {model.hparams.latent_dim}
- Encoder Hidden Dims: {model.hparams.encoder_hidden_dims}
- Decoder Hidden Dims: {model.hparams.decoder_hidden_dims}
- Dropout: {model.hparams.dropout}
- Normalization: {model.hparams.norm_type}
- Beta (KL weight): {model.hparams.beta}

## Reconstruction Metrics

### Normalized Space (Training Space)
- Mean MAE: {metrics['mean_mae_norm']:.6f} ± {metrics['std_mae_norm']:.6f}
- Mean MSE: {metrics['mean_mse_norm']:.6f} ± {metrics['std_mse_norm']:.6f}
- Number of samples: {len(metrics['mae_per_sample_norm']):,}
"""
    
    if metrics['has_original_space']:
        report += f"""
### Original Space (Interpretable Units)
- Mean MAE: {metrics['mean_mae_orig']:.6f} ± {metrics['std_mae_orig']:.6f}
- Mean MSE: {metrics['mean_mse_orig']:.6f} ± {metrics['std_mse_orig']:.6f}

### Interpretation Notes
- **Normalized Space**: Reflects what the model actually optimized for during training
- **Original Space**: Shows reconstruction quality in original measurement units
- **Ratio (Orig/Norm)**: MAE ratio = {metrics['mean_mae_orig']/metrics['mean_mae_norm']:.2f}, MSE ratio = {metrics['mean_mse_orig']/metrics['mean_mse_norm']:.2f}
"""
    else:
        report += """
### Note
- Only normalized/model space metrics available (no normalization parameters found)
"""
    
    # Compute feature correlations for the appropriate space
    original_key = 'original_orig' if metrics['has_original_space'] else 'original_norm'
    reconstructed_key = 'reconstructed_orig' if metrics['has_original_space'] else 'reconstructed_norm'
    
    correlations = []
    for i in range(metrics[original_key].shape[1]):  # All features instead of limiting to 100
        corr = np.corrcoef(metrics[original_key][:, i], metrics[reconstructed_key][:, i])[0, 1]
        if not np.isnan(corr):  # Only add valid correlations
            correlations.append(corr)
    
    report += f"""
## Latent Space Analysis
- Best number of clusters: {latent_analysis['best_n_clusters']}
- Best silhouette score: {latent_analysis['best_silhouette_score']:.4f}
- PCA explained variance (PC1, PC2): {latent_analysis['pca_explained_variance'][0]:.4f}, {latent_analysis['pca_explained_variance'][1]:.4f}

## Feature Reconstruction Quality"""
    
    if correlations:
        space_name = "Original" if metrics['has_original_space'] else "Normalized"
        report += f"""
- Features with excellent reconstruction (correlation > 0.9): {np.sum(np.array(correlations) > 0.9)} / {len(correlations)}
- Features with good reconstruction (correlation > 0.8): {np.sum(np.array(correlations) > 0.8)} / {len(correlations)}
- Features with poor reconstruction (correlation < 0.5): {np.sum(np.array(correlations) < 0.5)} / {len(correlations)}
- Mean correlation in {space_name} space: {np.mean(correlations):.4f}
"""
    else:
        report += """
- Could not compute feature correlations (insufficient data or NaN values)
"""
    
    report += f"""
## Model Parameters
- Total parameters: {sum(p.numel() for p in model.parameters()):,}
- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

## Recommendations
"""
    
    # Add interpretation-based recommendations
    if metrics['has_original_space']:
        mae_ratio = metrics['mean_mae_orig'] / metrics['mean_mae_norm']
        if mae_ratio > 10:
            report += "- High error ratio suggests normalization is crucial for this dataset\n"
        elif mae_ratio < 2:
            report += "- Low error ratio suggests features are naturally well-scaled\n"
        
        if metrics['mean_mae_orig'] < 0.1:
            report += "- Excellent reconstruction quality in original space\n"
        elif metrics['mean_mae_orig'] < 0.5:
            report += "- Good reconstruction quality in original space\n"
        else:
            report += "- Consider improving model architecture or increasing training\n"
    
    if latent_analysis['best_silhouette_score'] > 0.5:
        report += "- Latent space shows good clustering structure\n"
    elif latent_analysis['best_silhouette_score'] < 0.2:
        report += "- Latent space shows poor clustering - consider different architectures\n"
    
    if correlations and np.mean(correlations) > 0.8:
        report += "- High feature-wise correlations indicate good reconstruction fidelity\n"
    elif correlations and np.mean(correlations) < 0.5:
        report += "- Low feature correlations suggest model may need more capacity or training\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")
    
    print(report)


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
    
    # Setup data
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
        'cluster_labels': latent_analysis['cluster_labels']
    }
    
    # Save metrics for both spaces if available
    if metrics['has_original_space']:
        save_dict.update({
            'mae_per_sample_norm': metrics['mae_per_sample_norm'],
            'mse_per_sample_norm': metrics['mse_per_sample_norm'],
            'mae_per_sample_orig': metrics['mae_per_sample_orig'],
            'mse_per_sample_orig': metrics['mse_per_sample_orig'],
        })
    else:
        save_dict.update({
            'mae_per_sample': metrics['mae_per_sample_norm'],
            'mse_per_sample': metrics['mse_per_sample_norm'],
        })
    
    np.savez(metrics_path, **save_dict)
    print(f"Metrics saved to: {metrics_path}")
    
    print(f"Evaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 