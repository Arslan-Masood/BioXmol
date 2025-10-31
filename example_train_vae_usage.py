#!/usr/bin/env python3
"""
Example usage script for training VAE on different modalities.
This script demonstrates how to train VAE models on both JUMP Cell Painting 
and genomic (LINCS L1000) data using the modality selection feature.
"""

import os
import subprocess
import sys

def run_training(config_path, description):
    """Run VAE training with the specified config."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Config: {config_path}")
    print('='*60)
    
    cmd = [sys.executable, "train_vae.py", "--config", config_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        print("STDOUT:", e.stdout[-500:] if e.stdout else "No stdout")
        print("STDERR:", e.stderr[-500:] if e.stderr else "No stderr")

def main():
    """Main function to demonstrate different training configurations."""
    
    # Check if configs exist
    configs = [
        ("configs/train_vae_jump_cp_config.yaml", "JUMP Cell Painting VAE"),
        ("configs/train_vae_genomic_config.yaml", "Genomic (All Cell Lines) VAE"),
        ("configs/train_vae_genomic_u2os_config.yaml", "Genomic (U2OS Only) VAE"),
    ]
    
    print("VAE Training Examples for Different Modalities")
    print("=" * 60)
    
    for config_path, description in configs:
        if os.path.exists(config_path):
            print(f"✅ Found: {config_path}")
        else:
            print(f"❌ Missing: {config_path}")
            
    print("\nAvailable training options:")
    print("1. Train on JUMP Cell Painting data")
    print("2. Train on genomic data (all cell lines)")
    print("3. Train on genomic data (U2OS only)")
    print("4. Run all trainings sequentially")
    print("5. Show config examples")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        if os.path.exists(configs[0][0]):
            run_training(configs[0][0], configs[0][1])
        else:
            print(f"Config file not found: {configs[0][0]}")
            
    elif choice == "2":
        if os.path.exists(configs[1][0]):
            run_training(configs[1][0], configs[1][1])
        else:
            print(f"Config file not found: {configs[1][0]}")
            
    elif choice == "3":
        if os.path.exists(configs[2][0]):
            run_training(configs[2][0], configs[2][1])
        else:
            print(f"Config file not found: {configs[2][0]}")
            
    elif choice == "4":
        for config_path, description in configs:
            if os.path.exists(config_path):
                run_training(config_path, description)
            else:
                print(f"Skipping {description} - config not found: {config_path}")
                
    elif choice == "5":
        show_config_examples()
        
    else:
        print("Invalid choice. Exiting.")

def show_config_examples():
    """Show configuration examples for both modalities."""
    print("\n" + "="*60)
    print("Configuration Examples")
    print("="*60)
    
    print("\n1. JUMP Cell Painting Configuration (configs/train_vae_jump_cp_config.yaml):")
    print("-" * 40)
    print("""
data:
  modality: "jump_cp"  # Specify Cell Painting modality
  data_path: "/path/to/centered.filtered.parquet"
  batch_size: 512
  normalize: true

model:
  model_type: "vae"
  architecture: "vanilla"
  latent_dim: 128
  dropout: 0.1
    """)
    
    print("\n2. Genomic Configuration (configs/train_vae_genomic_config.yaml):")
    print("-" * 40)
    print("""
data:
  modality: "genomic"  # Specify genomic modality
  genomic_data_path: "/path/to/landmark_cmp_data.parquet"
  batch_size: 256   # Smaller batch size for genomic data
  normalize: true
  target_cell_line: null  # All cell lines

model:
  model_type: "vae"
  architecture: "medium"   # Larger model for genomic features
  latent_dim: 256          # Larger latent space
  dropout: 0.2             # Higher dropout
    """)
    
    print("\n3. U2OS-Specific Genomic Configuration (configs/train_vae_genomic_u2os_config.yaml):")
    print("-" * 40)
    print("""
data:
  modality: "genomic"
  genomic_data_path: "/path/to/landmark_cmp_data.parquet"
  batch_size: 512   # Can use larger batch with single cell line
  normalize: true
  target_cell_line: "U2OS"  # Filter to U2OS only

model:
  model_type: "vae"
  architecture: "vanilla"  # Smaller model for single cell line
  latent_dim: 128
  dropout: 0.15
    """)
    
    print("\n" + "="*60)
    print("Command Line Usage:")
    print("="*60)
    print("python train_vae.py --config configs/train_vae_jump_cp_config.yaml")
    print("python train_vae.py --config configs/train_vae_genomic_config.yaml")
    print("python train_vae.py --config configs/train_vae_genomic_u2os_config.yaml")
    print("="*60)

if __name__ == "__main__":
    main() 