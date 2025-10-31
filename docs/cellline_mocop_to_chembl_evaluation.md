# CellLine MoCop to Downstream Task Evaluation Pipeline

This document describes how to use a trained `CellLineTripleInputEncoder` molecular encoder for downstream task evaluation, including ChEMBL20 and DILI datasets, following the pattern established for the original MoCop model.

## Overview

The pipeline consists of three main steps:
1. **Extract** the molecular encoder from a trained `CellLineTripleInputEncoder` checkpoint
2. **Convert** it to a `LightningGGNN` format for compatibility with ChEMBL20 evaluation
3. **Evaluate** on downstream tasks (ChEMBL20 or DILI) using linear probing

## Files Created

### Core Scripts
- `configs/chembl20_cellline_mocop_linear.yml` - Configuration for ChEMBL20 evaluation
- `configs/MultiModal_finetuning_DILI.yml` - Configuration for DILI evaluation
- `exp/train_chembl20_cellline_mocop_linear_triton.sh` - SLURM script for ChEMBL20 evaluation
- `exp/MultiModal_finetuning_DILI.sh` - SLURM script for DILI evaluation

### DVC Pipeline Integration
- Added to `dvc.yaml` as `train_chembl20_cellline_mocop_linear` and `train_dili_cellline_mocop_linear` stages
- Can be run with: `dvc repro train_dili_cellline_mocop_linear`

## Quick Start

### Step 1: Prepare Your Checkpoint

Ensure you have a trained `CellLineTripleInputEncoder` checkpoint. This should be a `.ckpt` file from your contrastive training.

```bash
# Example checkpoint path
CELLLINE_CHECKPOINT="/path/to/your/cellline_mocop_checkpoint.ckpt"
```

### Step 2: Run the Evaluation

#### For ChEMBL20 Evaluation:
```bash
sbatch exp/train_chembl20_cellline_mocop_linear_triton.sh \
    /scratch/work/masooda1/trained_model_pred/cellline_mocop \
    "$CELLLINE_CHECKPOINT" \
    chembl20_cellline_mocop_linear.yml \
    mocop
```

#### For DILI Evaluation:
```bash
sbatch scripts/MultiModal_finetuning_DILI.sh \
    /scratch/work/masooda1/trained_model_pred/cellline_mocop \
    "$CELLLINE_CHECKPOINT" \
    MultiModal_finetuning_DILI.yml \
    mocop
```

#### Using DVC Pipeline:
```bash
# For DILI evaluation
dvc repro train_dili_cellline_mocop_linear

# For ChEMBL20 evaluation  
dvc repro train_chembl20_cellline_mocop_linear
```

**Note**: You'll need to update the checkpoint path in the DVC commands before running.

### Step 3: Monitor and Collect Results

```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f /scratch/work/masooda1/trained_model_pred/cellline_mocop/chembl_20_pretrained_cellline_mocop_linear_prob/script_output/*.out  # ChEMBL20
tail -f /scratch/work/masooda1/trained_model_pred/cellline_mocop/dili_pretrained_cellline_mocop_linear_prob/script_output/*.out        # DILI

# Results will be in:
ls /scratch/work/masooda1/trained_model_pred/cellline_mocop/test_results/chembl20_cellline_mocop_linear/  # ChEMBL20
ls /scratch/work/masooda1/trained_model_pred/cellline_mocop/test_results/dili_cellline_mocop_linear/        # DILI
```

## Detailed Usage

### Manual Molecular Encoder Extraction

If you want to extract the molecular encoder separately using the existing `remap_state_dict.py` utility:

```bash
# First remapping: encoder_a -> model
python bin/remap_state_dict.py \
    -i "/path/to/cellline_mocop_checkpoint.ckpt" \
    -o "/path/to/extracted_molecular_encoder.ckpt" \
    --map_from "encoder_a" \
    --map_to "model"

# Second remapping: model.fc_layers.0 -> model.fc_layers.0.0 (for LightningGGNN compatibility)
python bin/remap_state_dict.py \
    -i "/path/to/extracted_molecular_encoder.ckpt" \
    -o "/path/to/extracted_molecular_encoder.ckpt" \
    --map_from "model.fc_layers.0" \
    --map_to "model.fc_layers.0.0"
```

### DVC Pipeline Usage

The evaluation pipeline is integrated into DVC for reproducible experiments:

```bash
# Run DILI evaluation
dvc repro train_dili_cellline_mocop_linear

# Run ChEMBL20 evaluation
dvc repro train_chembl20_cellline_mocop_linear

# Check pipeline status
dvc status

# View pipeline graph
dvc dag
```

**Important**: Before running DVC commands, update the checkpoint path in `dvc.yaml`:
```yaml
# In dvc.yaml, replace /path/to/cellline_mocop_checkpoint.ckpt with actual path
cmd: >
  sbatch scripts/MultiModal_finetuning_DILI.sh
  /scratch/work/masooda1/trained_model_pred/cellline_mocop
  /actual/path/to/your/cellline_mocop_checkpoint.ckpt
  MultiModal_finetuning_DILI.yml
  mocop
```

### Configuration Customization

The configuration files can be customized:

**ChEMBL20**: `configs/chembl20_cellline_mocop_linear.yml`
**DILI**: `configs/MultiModal_finetuning_DILI.yml`

```yaml
# Key parameters
model:
  freeze: true              # Set to false for fine-tuning instead of linear probing
  
optimizer:
  lr: 5.0e-05              # Learning rate
  
trainer:
  max_epochs: 1000         # Maximum epochs
  
callbacks:
  patience: 20             # Early stopping patience
```

### Understanding the Pipeline

#### 1. Molecular Encoder Extraction

The pipeline uses the existing `remap_state_dict.py` utility:
- Loads the `CellLineTripleInputEncoder` checkpoint
- Remaps the parameter names from `encoder_a.*` to `model.*` for `LightningGGNN` compatibility
- Performs additional remapping for final layer structure compatibility
- Saves a new checkpoint that can be loaded as `LightningGGNN.load_from_checkpoint()`

#### 2. Downstream Task Evaluation

The evaluation follows the same pattern as the original MoCop:

**ChEMBL20 Evaluation:**
- **Linear Probing**: Freeze the molecular encoder weights and train only a new classification head
- **Multiple Settings**: Evaluate across different data fractions (1%, 5%, 10%, 25%, 50%, 100%) and splits
- **Metrics**: Tracks AUPRC (Area Under Precision-Recall Curve) as the main metric
- **Task**: Multi-label classification (1310 targets)

**DILI Evaluation:**
- **Linear Probing**: Freeze the molecular encoder weights and train only a new classification head
- **Multiple Seeds**: Evaluate across different random seeds (0, 1, 2) for robust evaluation
- **Metrics**: Tracks AUPRC (Area Under Precision-Recall Curve) as the main metric
- **Task**: Binary classification (toxic vs non-toxic)
- **Data Split**: 70% train, 10% validation, 20% test using Butina clustering

#### 3. Job Array Structure

**ChEMBL20 SLURM script** runs 18 jobs (array 0-17):
- 6 data fractions × 3 splits = 18 combinations
- Each job handles one fraction-split combination
- Results are aggregated across all runs

**DILI SLURM script** runs 3 jobs (array 0-2):
- 3 different random seeds for robust evaluation
- Each job handles one seed
- Results are aggregated across all seeds

## Output Structure

```
save_dir/
├── extracted_molecular_encoders/
│   └── checkpoint_name_molecular_encoder.ckpt
├── chembl20_cellline_mocop_linear_frac{X}_split{Y}_seed{Z}/     # ChEMBL20 results
│   ├── checkpoints/
│   │   ├── best-epoch=X-val_auprc=Y.ckpt
│   │   └── last.ckpt
│   └── logs/
├── dili_cellline_mocop_linear_seed{X}/                          # DILI results
│   ├── checkpoints/
│   │   ├── best-epoch=X-val_auprc=Y.ckpt
│   │   └── last.ckpt
│   └── logs/
├── test_results/
│   ├── chembl20_cellline_mocop_linear/
│   │   └── cellline_mocop_frac{X}_split{Y}_seed{Z}.csv
│   └── dili_cellline_mocop_linear/
│       └── cellline_mocop_seed{X}.csv
└── script_output/
    ├── chembl20-cellline-mocop-linear-*.out
    └── dili-cellline-mocop-linear-*.out
```

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   ```
   Error: Source CellLineTripleInputEncoder checkpoint not found
   ```
   - Verify the checkpoint path is correct
   - Ensure the file exists and is readable

2. **Extraction Failures**
   ```
   Error: Failed to extract molecular encoder
   ```
   - Check if the checkpoint is from `CellLineTripleInputEncoder` 
   - Verify the checkpoint contains `encoder_a` parameters

3. **Training Failures**
   ```
   Error: Training failed
   ```
   - Check GPU availability
   - Verify data files exist:
     - ChEMBL20: `data/chembl20/chembl20.csv`
     - DILI: `/scratch/work/masooda1/datasets/downstream_datasets/DILI/DILI_Goldstandard_1111.csv`
   - Check conda environment activation

### Debugging

Enable verbose logging by adding to the training command:
```bash
++trainer.fast_dev_run=true  # Quick test run
++trainer.limit_train_batches=10  # Limit batches for debugging
```

## Comparison with Original MoCop

| Aspect | Original MoCop | CellLine MoCop |
|--------|----------------|----------------|
| Input Models | `DualInputEncoder` | `CellLineTripleInputEncoder` |
| Molecular Encoder | `encoder_a` | `encoder_a` (same) |
| Additional Modalities | Morphological only | Morphological + Genomic + Cell line context |
| Extraction | Direct remapping | Same extraction process |
| Evaluation Tasks | ChEMBL20 linear probing | ChEMBL20 + DILI linear probing |
| ChEMBL20 Setup | Multi-label (1310 targets) | Multi-label (1310 targets) |
| DILI Setup | N/A | Binary classification (toxic vs non-toxic) |

The molecular encoder extraction and evaluation process is identical, just with a different source model type and additional downstream tasks.

## Next Steps

After successful evaluation:
1. Compare results with original MoCop baseline
2. Analyze performance across different data fractions (ChEMBL20) or seeds (DILI)
3. Consider fine-tuning (set `freeze: false`) for potentially better performance
4. Explore other downstream tasks beyond ChEMBL20 and DILI
5. Compare DILI performance with other toxicity prediction methods
