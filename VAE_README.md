# JUMP Cell Painting VAE Implementation

A comprehensive implementation of Variational Autoencoders (VAE) for the JUMP Cell Painting dataset, featuring molecule-wise sampling, predefined splits integration, and multiple architecture configurations.

## Quick Start & Architecture Reference

### Available Architectures

Choose from three predefined architectures based on your computational resources and requirements:

```bash
# Compact model - fast training, lower memory usage
python train_vae.py --architecture vanilla --latent_dim 64

# Balanced model - good performance vs efficiency trade-off  
python train_vae.py --architecture medium --latent_dim 128

# High-capacity model - best performance, requires more resources
python train_vae.py --architecture large --latent_dim 256
```

### Architecture Specifications

| Architecture | Encoder Layers | Decoder Layers | Parameters | Memory | Training Time |
|--------------|---------------|----------------|------------|---------|--------------|
| **vanilla** | 4000→512→256 | 256→512→4000 | ~3M | Low | Fast |
| **medium** | 4000→1024→512→256 | 256→512→1024→4000 | ~11M | Medium | Medium |
| **large** | 4000→2048→1024→512→256 | 256→512→1024→2048→4000 | ~22M | High | Slow |

### Model Type Options

- **VAE** (`--model_type vae`): Variational Autoencoder with KL divergence regularization
- **AE** (`--model_type ae`): Standard Autoencoder without KL divergence

## Key Features

### Updated Data Module (v2.0)
- **Molecule-wise sampling**: Similar to `CellLineTripleInputGraphDatasetJUMP`, samples are grouped by molecule (SMILES)
- **Predefined splits support**: Uses `build_dataloaders` with existing split files
- **Random sampling fallback**: When no splits provided, falls back to random splitting
- **Improved normalization**: Per-molecule sampling with consistent normalization

### Model Architectures
- **Vanilla**: Compact encoder-decoder (4k → 512 → 256 → latent → 256 → 512 → 4k)
- **Medium**: Standard architecture with balanced capacity (4k → 1024 → 512 → 256 → latent → 256 → 512 → 1024 → 4k)  
- **Large**: Extended architecture with high capacity (4k → 2048 → 1024 → 512 → 256 → latent → 256 → 512 → 1024 → 2048 → 4k)

### Advanced Features
- Configurable hyperparameters (dropout, normalization, learning rate, L2 regularization)
- Comprehensive logging (WandB, TensorBoard)
- Extensive evaluation metrics (MAE, MSE, latent analysis, visualizations)
- SLURM cluster integration for Triton

## Updated Installation & Setup

```bash
# Clone repository
git clone https://github.com/Arslan-Masood/Multi_Modal_Contrastive.git
cd Multi_Modal_Contrastive

# Install dependencies
pip install torch pytorch-lightning pandas numpy scikit-learn
pip install wandb tensorboard matplotlib seaborn
pip install rdkit-pypi  # For molecular processing
```

## New Usage: Molecule-wise Dataset with Predefined Splits

### 1. Prepare Your Split Files

Create CSV files with SMILES column for train/val/test splits:

```csv
# train_split.csv
SMILES
CCO
C1=CC=CC=C1
...

# val_split.csv  
SMILES
CCC
C1=CC=C(C=C1)O
...
```

### 2. Basic Training with Predefined Splits

```python
from train_vae import JUMPDataModule, create_vae_model
import pytorch_lightning as pl

# Setup data module with predefined splits
splits = {
    "train": "/path/to/train_split.csv",
    "val": "/path/to/val_split.csv", 
    "test": "/path/to/test_split.csv"  # Optional
}

data_module = JUMPDataModule(
    data_path="/path/to/jump_data.parquet",
    splits=splits,  # New: predefined splits
    batch_size=256,
    normalize=True,
    random_seed=42
)

# Create model
model = create_vae_model(
    input_dim=data_module.feature_dim,
    architecture="vanilla",  # Options: vanilla, medium, large
    latent_dim=128,
    dropout=0.1,
    norm_type="batchnorm"
)

# Train
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, data_module)
```

### 3. Command Line Training

```bash
# With predefined splits
python train_vae.py \
    --data_path /path/to/jump_data.parquet \
    --train_split /path/to/train_split.csv \
    --val_split /path/to/val_split.csv \
    --test_split /path/to/test_split.csv \
    --architecture vanilla \
    --latent_dim 128 \
    --batch_size 256 \
    --max_epochs 100 \
    --gpus 1

# Without splits (random splitting)
python train_vae.py \
    --data_path /path/to/jump_data.parquet \
    --architecture medium \
    --latent_dim 128 \
    --batch_size 256 \
    --max_epochs 100 \
    --gpus 1
```

## Molecule-wise Dataset Features

### JUMPCellPaintingDataset Class

The new dataset class provides molecule-wise sampling:

```python
from train_vae import JUMPCellPaintingDataset

# Create dataset
dataset = JUMPCellPaintingDataset(
    data_path="/path/to/jump_data.parquet",
    normalize=True,
    random_seed=42
)

# Key features
print(f"Unique molecules: {len(dataset.unique_smiles)}")
print(f"Feature dimension: {dataset.feature_dim}")

# Sample a molecule (random replicate)
sample = dataset[0]  # Returns features for molecule 0
```

### Key Differences from Original

| Feature | Original | Updated (v2.0) |
|---------|----------|----------------|
| Sampling | Row-based | Molecule-based (SMILES) |
| Splits | Random percentages | Predefined CSV files |
| Integration | Custom splitting | Uses `build_dataloaders` |
| Fallback | None | Random splits if no files |
| Compatibility | Standalone | Integrates with existing codebase |

## Integration with Existing Codebase

The updated `JUMPDataModule` integrates seamlessly with the existing MoCoP codebase:

```python
# Uses existing functions from mocop/
from training import build_dataloaders  # Leverages existing dataloader logic
from dataset import _split_data        # Uses existing split handling

# Compatible with existing split files
splits = {
    "train": "/path/to/jump-compound-split-0-train.csv",
    "val": "/path/to/jump-compound-split-0-val.csv"
}
```

## Example Split File Paths

For the Multi_Modal_Contrastive project:

```python
# JUMP-only splits
splits = {
    "train": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-0-train.csv",
    "val": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-0-val.csv"
}

# JUMP + LINCS combined splits  
splits = {
    "train": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/JUMP-LINCS-compound-split-0-train.csv",
    "val": "/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS_All_cell_lines/JUMP-LINCS-compound-split-0-val.csv"
}
```

## Model Architectures

The VAE implementation supports three predefined architectures with increasing capacity:

| Architecture | Encoder Path | Decoder Path | Parameters | Use Case |
|--------------|-------------|--------------|------------|----------|
| **Vanilla** | 4000 → 512 → 256 → latent | latent → 256 → 512 → 4000 | ~3M | Compact, efficient training |
| **Medium** | 4000 → 1024 → 512 → 256 → latent | latent → 256 → 512 → 1024 → 4000 | ~11M | Balanced capacity & efficiency |
| **Large** | 4000 → 2048 → 1024 → 512 → 256 → latent | latent → 256 → 512 → 1024 → 2048 → 4000 | ~22M | High capacity, complex features |

### Vanilla (Default)
- **Architecture**: 4000 → 512 → 256 → latent → 256 → 512 → 4000
- **Parameters**: ~3M
- **Use case**: Compact model for efficient training and inference

### Medium
- **Architecture**: 4000 → 1024 → 512 → 256 → latent → 256 → 512 → 1024 → 4000
- **Parameters**: ~11M
- **Use case**: Standard reconstruction with good balance of capacity and efficiency

### Large
- **Architecture**: 4000 → 2048 → 1024 → 512 → 256 → latent → 256 → 512 → 1024 → 2048 → 4000
- **Parameters**: ~22M
- **Use case**: High capacity model for complex feature representations

## Hyperparameter Configuration

### Core Parameters
```python
# Model architecture
--architecture vanilla|medium|large
--latent_dim 64|128|256|512

# Regularization
--dropout 0.0-0.5
--norm_type batchnorm|layernorm|none
--beta 0.1-10.0  # KL divergence weight

# Optimization
--learning_rate 1e-5 to 1e-2
--weight_decay 1e-6 to 1e-2
--batch_size 128|256|512|1024
```

### Training Configuration
```python
# Training
--max_epochs 100
--patience 15
--gpus 1
--precision 32

# Data
--val_split 0.2
--test_split 0.1
--num_workers 8
```

## Training on Triton Cluster

### Updated SLURM Scripts

The SLURM scripts now support the new predefined splits:

```bash
# Modified train_vae_triton.sh
sbatch scripts/train_vae_triton.sh \
    --data_path /scratch/work/masooda1/datasets/jump_data/centered.filtered.parquet \
    --train_split /scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-0-train.csv \
    --val_split /scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data/jump-compound-split-0-val.csv
```

## Evaluation & Analysis

### Training Metrics
- Total loss (reconstruction + KL)
- Reconstruction loss (MSE)
- KL divergence loss
- MAE (Mean Absolute Error)
- Learning rate

### Evaluation Metrics
- **MAE Histogram**: Distribution of per-sample reconstruction errors
- **MSE Distribution**: Reconstruction quality statistics
- **Feature Correlations**: Original vs reconstructed feature correlations
- **Latent Space Analysis**: PCA, t-SNE, clustering analysis
- **Reconstruction Scatter**: Quality visualization for individual features

### Generated Outputs
- `reconstruction_quality.png`: Comprehensive reconstruction analysis
- `latent_space_analysis.png`: Latent space visualization and clustering
- `evaluation_report.md`: Detailed evaluation report
- `metrics.npz`: Raw metrics for further analysis

## Complete Example

See `vae_example.py` for comprehensive examples including:

1. **Predefined splits usage**: How to use existing split files
2. **Molecule-wise dataset**: Direct usage of the new dataset class  
3. **Model training**: Simple VAE training demo
4. **Evaluation**: Comprehensive model evaluation
5. **Latent analysis**: PCA, t-SNE visualization of latent space

```bash
# Run all examples
python vae_example.py
```

## Migration from Original VAE

To migrate from the original VAE implementation:

1. **Update import**: `from train_vae import JUMPDataModule` (unchanged)
2. **Add splits parameter**: Provide dictionary with split file paths
3. **Remove percentage splits**: No longer need `val_split`, `test_split` percentages
4. **Update paths**: Point to your actual split CSV files

```python
# Old usage (hypothetical)
data_module = JUMPDataModule(
    data_path="data.parquet",
    val_split=0.2,  # Remove
    test_split=0.1  # Remove
)

# New usage  
data_module = JUMPDataModule(
    data_path="data.parquet",
    splits={  # Add
        "train": "train_split.csv",
        "val": "val_split.csv"
    }
)

# Architecture examples
vanilla_model = create_vae_model(architecture="vanilla")   # Compact: 512→256→latent
medium_model = create_vae_model(architecture="medium")     # Balanced: 1024→512→256→latent  
large_model = create_vae_model(architecture="large")       # High-capacity: 2048→1024→512→256→latent
```

## Troubleshooting

### Common Issues

1. **Split files not found**: Check file paths and ensure CSV files exist
2. **SMILES column missing**: Dataset will fall back to row-based sampling
3. **Memory issues**: Reduce `batch_size` or use smaller splits for testing

### Debug Mode

```python
# Enable verbose output
data_module = JUMPDataModule(
    data_path="data.parquet", 
    splits=splits,
    batch_size=64  # Smaller for debugging
)

data_module.setup()
print(f"Feature dim: {data_module.feature_dim}")
print(f"Unique molecules: {len(data_module.dataset.unique_smiles)}")
```

## Citation

If you use this VAE implementation, please cite:

```bibtex
@inproceedings{masood2025multimodal,
  title={Multi-Modal Representation Learning for Molecules},
  author={Masood, Muhammad Arslan and Heinonen, Markus and Kaski, Samuel},
  booktitle={ICLR 2025 Workshop},
  year={2025},
  url={https://openreview.net/forum?id=WT7BpLvL6D}
}
``` 