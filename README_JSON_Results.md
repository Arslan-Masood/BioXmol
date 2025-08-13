# VAE Training with Configurable Model Weight Saving

This system has been enhanced to give you flexible control over model weight saving. You can choose to save both JSON results and model weights, or just JSON results for efficient hyperparameter sweeps.

## Key Features

### 1. Configurable Model Weight Saving
- **Configurable**: Control via `save_model_weights` parameter in config files
- **Default**: Model weights are saved (`save_model_weights: true`)
- **For sweeps**: Set to `false` to save only JSON results for efficiency

### 2. JSON Results Logging (Always Enabled)
- **Always active**: Simple final metrics saving after training completion
- **Saves**: Final validation metrics and hyperparameters in JSON format
- **Location**: `{log_dir}/{experiment_name}/results/final_results.json`

### 3. Files Created Per Experiment

**When `save_model_weights: true` (default):**
```
vae_logs/
└── {experiment_name}/
    ├── checkpoints/
    │   ├── best-{epoch}-{val_loss}.ckpt    # Best model checkpoints
    │   └── last.ckpt                       # Latest checkpoint
    ├── final_model.ckpt                    # Final saved model
    └── results/
        └── final_results.json              # Final metrics and hyperparameters
```

**When `save_model_weights: false`:**
```
vae_logs/
└── {experiment_name}/
    └── results/
        └── final_results.json              # Final metrics and hyperparameters
```

## Configuration Options

### Individual Experiments

Add to your YAML config file:
```yaml
# Training parameters
training:
  max_epochs: 100
  patience: 15
  gpus: 1
  precision: 32
  seed: 42
  save_model_weights: true   # true = save weights, false = JSON only
```

### Hyperparameter Sweeps

Control for entire sweep in `scripts/vae_sweep_triton.sh`:
```bash
# Control model weight saving for the entire sweep
SAVE_MODEL_WEIGHTS=false  # Set to true to save model weights, false for JSON-only results
```

## Usage Scenarios

### 1. Hyperparameter Exploration (Recommended: `save_model_weights: false`)
```bash
# Edit scripts/vae_sweep_triton.sh
SAVE_MODEL_WEIGHTS=false

# Run sweep
sbatch scripts/vae_sweep_triton.sh

# Analyze results to find best hyperparameters
python analyze_hp_results.py --log_dir vae_logs
```

**Benefits**: 
- Massive storage savings (can save TBs)
- Faster training (no checkpoint I/O)
- Quick results analysis

### 2. Production Training (Use: `save_model_weights: true`)
```bash
# After identifying best hyperparameters from sweep analysis
# Create/edit config with save_model_weights: true
python train_vae.py --config configs/best_model_config.yaml
```

**Benefits**:
- Full model weights for deployment
- Checkpoint recovery capability
- Complete experimental artifacts

### 3. Hybrid Approach (Recommended Workflow)
```bash
# Stage 1: Fast hyperparameter search (no weights)
SAVE_MODEL_WEIGHTS=false
sbatch scripts/vae_sweep_triton.sh

# Stage 2: Analyze and identify top performers
python analyze_hp_results.py --log_dir vae_logs

# Stage 3: Re-train only best models with weight saving
# Use identified hyperparameters with save_model_weights: true
```

## JSON File Structure

### final_results.json
```json
{
  "hyperparameters": {
    "data": {
      "batch_size": 64,
      "normalize": true,
      "data_path": "..."
    },
    "model": {
      "model_type": "ae",
      "architecture": "vanilla",
      "latent_dim": 128,
      "dropout": 0.0,
      "norm_type": "batchnorm",
      "beta": 1.0
    },
    "optimization": {
      "learning_rate": 0.001,
      "weight_decay": 0.0
    },
    "training": {
      "max_epochs": 100,
      "patience": 15,
      "seed": 42,
      "save_model_weights": false
    }
  },
  "final_metrics": {
    "final_validation": {
      "total_loss": 1.489,
      "recon_loss": 1.489,
      "kl_loss": 0.0,
      "mae": 0.865
    }
  },
  "total_epochs": 45,
  "experiment_name": "jump_ae_vanilla",
  "saved_at": "2024-01-15 14:30:25.123456"
}
```

## Running the Hyperparameter Sweep

### 1. Configure Model Saving
Edit `scripts/vae_sweep_triton.sh`:
```bash
# For exploration (recommended)
SAVE_MODEL_WEIGHTS=false

# For production runs  
SAVE_MODEL_WEIGHTS=true
```

### 2. Submit the Job Array
```bash
sbatch scripts/vae_sweep_triton.sh
```

This runs 216 experiments (180 VAE + 36 AE combinations):
- **Model types**: VAE, AE  
- **Architectures**: vanilla, medium, large
- **Normalization**: batchnorm, layernorm, none
- **Learning rates**: 1e-5, 1e-4, 1e-3, 1e-2
- **Beta values** (VAE only): 0.1, 0.5, 1.0, 5.0, 10.0

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Check specific experiment output
tail -f script_outputs/vae_hp_search_42.out

# Count completed experiments
ls vae_logs/*/results/training_metrics.json | wc -l

# If saving weights, check model files too
ls vae_logs/*/checkpoints/*.ckpt | wc -l
```

### 4. Analyze Results
```bash
# Run comprehensive analysis
python analyze_hp_results.py --log_dir vae_logs --output_dir hp_analysis

# This creates:
# hp_analysis/
# ├── hyperparameter_sweep_results.csv    # Detailed results table
# ├── learning_rate_analysis.png          # Learning rate comparison
# ├── architecture_analysis.png           # Architecture comparison  
# ├── beta_analysis.png                   # Beta analysis (VAE only)
# └── normalization_analysis.png          # Normalization comparison
```

## Benefits of This Flexible Approach

### JSON-Only Mode (`save_model_weights: false`)
1. **Storage Efficient**: No model weights (saves TBs of space)
2. **Faster Training**: No checkpoint I/O overhead
3. **Quick Analysis**: Fast JSON parsing vs loading PyTorch models
4. **Scalable**: Handle hundreds of experiments efficiently

### Full Model Mode (`save_model_weights: true`)
1. **Complete Artifacts**: Full model weights for deployment
2. **Recovery**: Resume training from checkpoints
3. **Best Model Access**: Load and use the best performing model
4. **Production Ready**: Everything needed for model deployment

## Recommended Workflow

```bash
# 1. Fast exploration phase (no weights)
# Edit vae_sweep_triton.sh: SAVE_MODEL_WEIGHTS=false
sbatch scripts/vae_sweep_triton.sh

# 2. Analyze results 
python analyze_hp_results.py --log_dir vae_logs

# 3. Identify top 3-5 configurations from analysis

# 4. Re-train best models with weights
# Create configs with identified hyperparameters + save_model_weights: true
python train_vae.py --config configs/best_vae_config.yaml
python train_vae.py --config configs/best_ae_config.yaml

# 5. Deploy best model
# Use saved checkpoints for final model deployment
```

This two-stage approach maximizes efficiency: fast exploration followed by selective detailed training. 