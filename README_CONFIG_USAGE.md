# VAE Training with Config Files

The VAE training script now uses **config files only** - no command line arguments needed except for specifying the config file path.

## Quick Start

### 1. Submit a training job directly:

```bash
# Train with VAE config
sbatch scripts/train_vae_triton.sh configs/train_vae_config.yaml

# Train with AE config  
sbatch scripts/train_vae_triton.sh configs/train_ae_config.yaml
```

### 2. Local training (for testing):

```bash
# Train VAE locally
python train_vae.py --config configs/train_vae_config.yaml

# Train AE locally
python train_vae.py --config configs/train_ae_config.yaml
```

## Config File Structure

All configuration is done through YAML files. Here's the structure:

```yaml
# Data parameters
data:
  data_path: "/path/to/your/data.parquet"
  train_split: "/path/to/train_split.csv"  # Optional
  val_split: "/path/to/val_split.csv"      # Optional
  test_split: "/path/to/test_split.csv"    # Optional
  batch_size: 512
  num_workers: 8

# Model parameters
model:
  model_type: "vae"        # "vae" or "ae"
  architecture: "vanilla"  # "vanilla", "medium", or "large"
  latent_dim: 128
  dropout: 0.1
  norm_type: "batchnorm"   # "batchnorm", "layernorm", or "none"
  beta: 1.0                # KL divergence weight (ignored for AE)

# Optimization parameters
optimization:
  learning_rate: 1e-3
  weight_decay: 1e-4

# Training parameters
training:
  max_epochs: 100
  patience: 15
  gpus: 1
  precision: 32
  seed: 42

# Logging parameters
logging:
  experiment_name: "my_experiment"
  project_name: "jump-cell-painting"
  log_dir: "/path/to/logs"
  use_wandb: false
```

## Available Config Files

- `configs/train_vae_config.yaml` - Vanilla VAE with default dataset
- `configs/train_ae_config.yaml` - Vanilla AE with dummy data and splits

## Creating Custom Configs

1. Copy an existing config file:
```bash
cp configs/train_vae_config.yaml configs/my_config.yaml
```

2. Edit the parameters as needed

3. Submit the job:
```bash
sbatch scripts/train_vae_triton.sh configs/my_config.yaml
```

## Benefits of Config-Only Approach

- ✅ **Cleaner**: No long command lines
- ✅ **Reproducible**: Config files can be version controlled
- ✅ **Flexible**: Easy to create different experimental setups
- ✅ **Organized**: All parameters in one place
- ✅ **Shareable**: Send config files to collaborators

## Example Workflow

```bash
# 1. Create configs for different experiments
cp configs/train_vae_config.yaml configs/experiment_1.yaml
cp configs/train_vae_config.yaml configs/experiment_2.yaml

# 2. Edit configs (change architecture, latent_dim, etc.)
vim configs/experiment_1.yaml  # Set architecture: "medium"
vim configs/experiment_2.yaml  # Set architecture: "large"

# 3. Submit jobs
sbatch scripts/train_vae_triton.sh configs/experiment_1.yaml
sbatch scripts/train_vae_triton.sh configs/experiment_2.yaml

# 4. Monitor jobs
squeue -u $USER
```

This approach makes it much easier to manage multiple experiments and reproduce results! 