#!/usr/bin/env python3
"""
UMAP Visualization for Aggregated JUMP Cell Painting Data
Works with filtered.parquet (original) and centered.filtered.parquet (normalized)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
try:
    from cuml import UMAP as cumlUMAP
    import cupy as cp
    USE_GPU = True
    print("cuML UMAP available - using GPU acceleration")
except ImportError:
    import umap
    USE_GPU = False
    print("cuML not available - using CPU UMAP")
from sklearn.preprocessing import LabelEncoder
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_aggregated_data(data_dir, max_plates=20):
    """
    Load the aggregated parquet files and sample by plates efficiently
    """
    print("Loading aggregated JUMP CP data...")
    
    # Load both files
    original_file = os.path.join(data_dir, "filtered.parquet")
    centered_file = os.path.join(data_dir, "centered.filtered.parquet")
    
    # First, read just the Metadata_Plate column to find common plates
    print("Reading plate metadata to find common plates...")
    orig_plates = set(pd.read_parquet(original_file, columns=['Metadata_Plate'])['Metadata_Plate'].unique())
    cent_plates = set(pd.read_parquet(centered_file, columns=['Metadata_Plate'])['Metadata_Plate'].unique())
    common_plates = list(orig_plates.intersection(cent_plates))
    
    print(f"Common plates: {len(common_plates)} (orig: {len(orig_plates)}, cent: {len(cent_plates)})")
    
    # Sample plates
    if len(common_plates) > max_plates:
        selected_plates = np.random.choice(common_plates, size=max_plates, replace=False)
    else:
        selected_plates = common_plates
    
    print(f"Using {len(selected_plates)} plates")
    
    # Now read only the data for selected plates using filters
    print("Loading filtered data...")
    df_orig_filtered = pd.read_parquet(
        original_file, 
        filters=[('Metadata_Plate', 'in', selected_plates)]
    )
    df_cent_filtered = pd.read_parquet(
        centered_file, 
        filters=[('Metadata_Plate', 'in', selected_plates)]
    )
    
    print(f"Original: {df_orig_filtered.shape}, Centered: {df_cent_filtered.shape}")
    
    return {
        'original': df_orig_filtered,
        'centered': df_cent_filtered
    }

def load_aggregated_data_by_drugs(data_dir, max_drugs=10):
    """
    Load the aggregated parquet files and sample by drugs (including all replicates)
    Simple approach: select drugs first, then keep common plates
    """
    print("Loading aggregated JUMP CP data by drugs...")
    
    # Load both files
    original_file = os.path.join(data_dir, "filtered.parquet")
    centered_file = os.path.join(data_dir, "centered.filtered.parquet")
    
    # First, get available drugs from both datasets
    print("Reading drug metadata...")
    orig_drugs = set(pd.read_parquet(original_file, columns=['Metadata_InChIKey'])['Metadata_InChIKey'].unique())
    cent_drugs = set(pd.read_parquet(centered_file, columns=['Metadata_InChIKey'])['Metadata_InChIKey'].unique())
    common_drugs = list(orig_drugs.intersection(cent_drugs))
    
    print(f"Common drugs: {len(common_drugs)}")
    
    # Sample drugs
    selected_drugs = np.random.choice(common_drugs, size=min(max_drugs, len(common_drugs)), replace=False)
    print(f"Using {len(selected_drugs)} drugs")
    
    # Load data for selected drugs
    print("Loading data for selected drugs...")
    df_orig = pd.read_parquet(original_file, filters=[('Metadata_InChIKey', 'in', selected_drugs)])
    df_cent = pd.read_parquet(centered_file, filters=[('Metadata_InChIKey', 'in', selected_drugs)])
    
    # Find common plates between the drug-filtered datasets
    orig_plates = set(df_orig['Metadata_Plate'].unique())
    cent_plates = set(df_cent['Metadata_Plate'].unique())
    common_plates = list(orig_plates.intersection(cent_plates))
    
    print(f"Common plates after drug filtering: {len(common_plates)} (orig: {len(orig_plates)}, cent: {len(cent_plates)})")
    
    # Filter to common plates
    df_orig_filtered = df_orig[df_orig['Metadata_Plate'].isin(common_plates)]
    df_cent_filtered = df_cent[df_cent['Metadata_Plate'].isin(common_plates)]
    
    print(f"Original: {df_orig_filtered.shape}, Centered: {df_cent_filtered.shape}")
    print(f"Original drugs: {df_orig_filtered['Metadata_InChIKey'].nunique()}, plates: {df_orig_filtered['Metadata_Plate'].nunique()}")
    print(f"Centered drugs: {df_cent_filtered['Metadata_InChIKey'].nunique()}, plates: {df_cent_filtered['Metadata_Plate'].nunique()}")
    
    return {
        'original': df_orig_filtered,
        'centered': df_cent_filtered
    }

def prepare_features(df):
    """
    Extract feature columns for UMAP
    """
    # Get feature columns (non-metadata)
    feature_cols = [col for col in df.columns 
                   if not col.startswith('Metadata_')]
    
    print(f"Using {len(feature_cols)} feature columns")
    
    # Extract features
    features = df[feature_cols].copy()
    
    # Handle missing/infinite values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    
    # Remove constant columns
    varying_cols = features.columns[features.nunique() > 1]
    features = features[varying_cols]
    
    print(f"Using {len(varying_cols)} varying features after filtering")
    
    return features.values, varying_cols.tolist()

def create_umap_plots(data_dict, output_dir='umap_plots', n_neighbors=15, min_dist=0.1):
    """
    Create UMAP plots for both datasets
    """
    print(f"\n🚀 Starting UMAP analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('UMAP: Original vs Normalized JUMP Cell Painting Data', fontsize=16, fontweight='bold')
    
    results = {}
    total_embedding_time = 0
    
    for idx, (data_type, df) in enumerate(data_dict.items()):
        print(f"\nProcessing {data_type} data for UMAP...")
        
        # Prepare features
        features, feature_names = prepare_features(df)
        
        if features.shape[0] == 0 or features.shape[1] == 0:
            print(f"No valid features for {data_type} data")
            continue
        
        # Run UMAP
        print("Computing UMAP embedding...")
        start_time = time.time()
        
        try:
            if USE_GPU:
                print("Using GPU-accelerated cuML UMAP...")
                reducer = cumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED)
                embedding = reducer.fit_transform(features)
                
                # Convert back to numpy if it's a cuML array
                if hasattr(embedding, 'to_numpy'):
                    embedding = embedding.to_numpy()
                elif hasattr(embedding, 'get'):
                    embedding = embedding.get()
            else:
                print("Using CPU UMAP...")
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED, verbose=False)
                embedding = reducer.fit_transform(features)
        except Exception as e:
            print(f"GPU UMAP failed: {e}")
            print("Falling back to CPU UMAP...")
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED, verbose=False)
            embedding = reducer.fit_transform(features)
        
        end_time = time.time()
        embedding_time = end_time - start_time
        
        print(f"UMAP embedding shape: {embedding.shape}")
        print(f"⏱️  UMAP computation time for {data_type} data: {embedding_time:.2f} seconds ({embedding_time/60:.2f} minutes)")
        
        total_embedding_time += embedding_time
        
        # Store results
        results[data_type] = {
            'embedding': embedding,
            'features': features,
            'df': df,
            'computation_time': embedding_time
        }
        
        # Plot colored by plate only
        ax = axes[idx]
        if 'Metadata_Plate' in df.columns and df['Metadata_Plate'].nunique() > 1:
            plate_ids = df['Metadata_Plate'].values
            le = LabelEncoder()
            colors = le.fit_transform(plate_ids)
            unique_plates = le.classes_
            
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=colors, cmap='tab20', alpha=0.7, s=20)
            
            # Add legend if not too many plates
            if len(unique_plates) <= 15:
                for i, plate in enumerate(unique_plates):
                    mask = colors == i
                    if np.any(mask):
                        ax.scatter([], [], c=plt.cm.tab20(i), label=str(plate), s=50)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Plate ID')
            
            ax.set_title(f'{data_type.title()} Data - Colored by Plate ID\n({len(unique_plates)} plates)')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
            ax.set_title(f'{data_type.title()} Data')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Standardize axis limits across both plots for better comparison
    if len(results) == 2:
        all_embeddings = [results[key]['embedding'] for key in results.keys()]
        x_min = min(emb[:, 0].min() for emb in all_embeddings)
        x_max = max(emb[:, 0].max() for emb in all_embeddings)
        y_min = min(emb[:, 1].min() for emb in all_embeddings)
        y_max = max(emb[:, 1].max() for emb in all_embeddings)
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        
        for ax in axes:
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Save plot
    plot_filename = os.path.join(output_dir, 'jump_umap_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    analysis_end_time = time.time()
    total_analysis_time = analysis_end_time - analysis_start_time
    
    # Print timing summary
    print(f"\n📊 Timing Summary:")
    print(f"  Total UMAP embedding time: {total_embedding_time:.2f} seconds ({total_embedding_time/60:.2f} minutes)")
    print(f"  Total analysis time: {total_analysis_time:.2f} seconds ({total_analysis_time/60:.2f} minutes)")
    print(f"  Overhead (plotting, I/O, etc.): {(total_analysis_time - total_embedding_time):.2f} seconds")
    print(f"\n🏁 UMAP analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def create_umap_plots_by_drugs(data_dict, output_dir='umap_plots', n_neighbors=15, min_dist=0.1):
    """
    Create UMAP plots for both datasets colored by drugs
    """
    print(f"\n🧬 Starting Drug-based UMAP analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('UMAP: Original vs Normalized JUMP Cell Painting Data (Colored by Drug)', fontsize=16, fontweight='bold')
    
    results = {}
    total_embedding_time = 0
    
    for idx, (data_type, df) in enumerate(data_dict.items()):
        print(f"\nProcessing {data_type} data for Drug UMAP...")
        
        # Prepare features
        features, feature_names = prepare_features(df)
        
        if features.shape[0] == 0 or features.shape[1] == 0:
            print(f"No valid features for {data_type} data")
            continue
        
        # Run UMAP
        print("Computing UMAP embedding...")
        start_time = time.time()
        
        try:
            if USE_GPU:
                print("Using GPU-accelerated cuML UMAP...")
                reducer = cumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED)
                embedding = reducer.fit_transform(features)
                
                # Convert back to numpy if it's a cuML array
                if hasattr(embedding, 'to_numpy'):
                    embedding = embedding.to_numpy()
                elif hasattr(embedding, 'get'):
                    embedding = embedding.get()
            else:
                print("Using CPU UMAP...")
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED, verbose=False)
                embedding = reducer.fit_transform(features)
        except Exception as e:
            print(f"GPU UMAP failed: {e}")
            print("Falling back to CPU UMAP...")
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED, verbose=False)
            embedding = reducer.fit_transform(features)
        
        end_time = time.time()
        embedding_time = end_time - start_time
        
        print(f"UMAP embedding shape: {embedding.shape}")
        print(f"⏱️  UMAP computation time for {data_type} data: {embedding_time:.2f} seconds ({embedding_time/60:.2f} minutes)")
        
        total_embedding_time += embedding_time
        
        # Store results
        results[data_type] = {
            'embedding': embedding,
            'features': features,
            'df': df,
            'computation_time': embedding_time
        }
        
        # Plot colored by drug
        ax = axes[idx]
        if 'Metadata_InChIKey' in df.columns and df['Metadata_InChIKey'].nunique() > 1:
            drug_ids = df['Metadata_InChIKey'].values
            le = LabelEncoder()
            colors = le.fit_transform(drug_ids)
            unique_drugs = le.classes_
            
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=colors, cmap='tab20', alpha=0.7, s=20)
            
            ax.set_title(f'{data_type.title()} Data - Colored by Drug\n({len(unique_drugs)} drugs, {df["Metadata_Plate"].nunique()} plates)')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
            ax.set_title(f'{data_type.title()} Data')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Standardize axis limits across both plots for better comparison
    if len(results) == 2:
        all_embeddings = [results[key]['embedding'] for key in results.keys()]
        x_min = min(emb[:, 0].min() for emb in all_embeddings)
        x_max = max(emb[:, 0].max() for emb in all_embeddings)
        y_min = min(emb[:, 1].min() for emb in all_embeddings)
        y_max = max(emb[:, 1].max() for emb in all_embeddings)
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        
        for ax in axes:
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Save plot
    plot_filename = os.path.join(output_dir, 'jump_umap_drugs_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nDrug UMAP plot saved as: {plot_filename}")
    
    analysis_end_time = time.time()
    total_analysis_time = analysis_end_time - analysis_start_time
    
    # Print timing summary
    print(f"\n📊 Drug UMAP Timing Summary:")
    print(f"  Total UMAP embedding time: {total_embedding_time:.2f} seconds ({total_embedding_time/60:.2f} minutes)")
    print(f"  Total analysis time: {total_analysis_time:.2f} seconds ({total_analysis_time/60:.2f} minutes)")
    print(f"  Overhead (plotting, I/O, etc.): {(total_analysis_time - total_embedding_time):.2f} seconds")
    print(f"\n🏁 Drug UMAP analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def main(args):
    """Main function"""
    # Set random seed for reproducibility
    global RANDOM_SEED
    RANDOM_SEED = args.random_seed
    np.random.seed(RANDOM_SEED)
    
    print("JUMP Cell Painting Aggregated Data UMAP Analysis")
    print("=" * 55)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plate sampling: max {args.max_plates} plates")
    print(f"Drug sampling: max {args.max_drugs} drugs")
    print(f"UMAP parameters: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    
    # === PLATE-BASED ANALYSIS ===
    print("\n" + "="*60)
    print("🔬 PLATE-BASED UMAP ANALYSIS")
    print("="*60)
    
    # Load aggregated data by plates
    plate_data_dict = load_aggregated_data(args.data_dir, max_plates=args.max_plates)
    
    if not plate_data_dict:
        print("No plate data loaded. Please check if aggregated files exist:")
        print(f"- {os.path.join(args.data_dir, 'filtered.parquet')}")
        print(f"- {os.path.join(args.data_dir, 'centered.filtered.parquet')}")
    else:
        # Create plate-based UMAP plots
        plate_results = create_umap_plots(plate_data_dict, args.output_dir, args.n_neighbors, args.min_dist)
        
        # Print plate summary
        print("\nPlate-based Summary:")
        print("-" * 30)
        for data_type, df in plate_data_dict.items():
            print(f"{data_type.title()} data:")
            print(f"  - Samples: {len(df)}")
            if 'Metadata_Plate' in df.columns:
                print(f"  - Unique plates: {df['Metadata_Plate'].nunique()}")
            if 'Metadata_Batch' in df.columns:
                print(f"  - Unique batches: {df['Metadata_Batch'].nunique()}")
            print()
    
    # === DRUG-BASED ANALYSIS ===
    print("\n" + "="*60)
    print("💊 DRUG-BASED UMAP ANALYSIS")
    print("="*60)
    
    # Load aggregated data by drugs
    drug_data_dict = load_aggregated_data_by_drugs(args.data_dir, max_drugs=args.max_drugs)
    
    if not drug_data_dict:
        print("No drug data loaded.")
    else:
        # Create drug-based UMAP plots
        drug_results = create_umap_plots_by_drugs(drug_data_dict, args.output_dir, args.n_neighbors, args.min_dist)
        
        # Print drug summary
        print("\nDrug-based Summary:")
        print("-" * 30)
        for data_type, df in drug_data_dict.items():
            print(f"{data_type.title()} data:")
            print(f"  - Samples: {len(df)}")
            if 'Metadata_InChIKey' in df.columns:
                print(f"  - Unique drugs: {df['Metadata_InChIKey'].nunique()}")
            if 'Metadata_Plate' in df.columns:
                print(f"  - Unique plates: {df['Metadata_Plate'].nunique()}")
            print()
    
    print(f"\n🎯 All results saved in: {args.output_dir}/")
    print("📊 Generated files:")
    print("  - jump_umap_comparison.png (plate-based)")
    print("  - jump_umap_drugs_comparison.png (drug-based)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create UMAP plots for aggregated JUMP Cell Painting data")
    parser.add_argument("-d", "--data-dir", required=True,
                       help="Directory containing aggregated parquet files")
    parser.add_argument("-o", "--output-dir", default="umap_plots",
                       help="Directory to save output plots and data")
    parser.add_argument("--max-plates", type=int, default=20,
                       help="Maximum number of plates to include for sampling")
    parser.add_argument("--max-drugs", type=int, default=10,
                       help="Maximum number of drugs to include for drug-based analysis")
    parser.add_argument("--n-neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1,
                       help="UMAP min_dist parameter")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible UMAP embeddings")
    
    args = parser.parse_args()
    main(args)
