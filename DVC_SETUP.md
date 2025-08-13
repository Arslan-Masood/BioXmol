# DVC Pipeline Setup and Documentation

This document provides comprehensive documentation for the DVC (Data Version Control) pipeline used in the Multi-Modal Representation Learning project.

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Installation and Setup](#installation-and-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Data Structure](#data-structure)
- [Configuration Options](#configuration-options)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Troubleshooting](#troubleshooting)

## Overview

The DVC pipeline automates the entire data processing workflow for multi-modal molecular representation learning. It handles data download, preprocessing, aggregation, and prepares datasets for model training.

### Pipeline Benefits

- ✅ **Reproducible**: Exact same results every time
- ✅ **Scalable**: Runs on SLURM cluster for large datasets
- ✅ **Error-safe**: Proper error handling and failure detection
- ✅ **Modular**: Can run individual stages independently
- ✅ **Trackable**: Full dependency and output tracking

## Pipeline Stages

The complete pipeline consists of the following stages:

### 1. 📥 Data Download (`download_jump_data`)

**Purpose**: Downloads and extracts JUMP Cell Painting data and ChEMBL datasets

**Command**: 
```bash
bash data/download_jump_data.sh /scratch/work/masooda1/datasets mocop
```

**What it does**:
- Extracts ChEMBL20 data from `data/chembl20.tar.gz`
- Extracts JUMP splits from `data/jump.tar.gz`
- Clones JUMP metadata repository
- Downloads individual JUMP compound plates using SLURM array jobs
- Normalizes cell painting features

**Dependencies**:
- `data/download_jump_data.sh`
- `data/_jump_download_single_plate.py`

**Outputs**:
- `/scratch/work/masooda1/datasets/chembl20/` (ChEMBL data)
- `/scratch/work/masooda1/datasets/jump/` (JUMP splits)
- `/scratch/work/masooda1/datasets/jump_preprocessed/` (Raw plate data)

### 2. 🔄 Data Processing (`process_jump_data`)

**Purpose**: Aggregates JUMP data, cleans up individual files, and creates train/test splits

**Command**:
```bash
sbatch data/process_data_trirton.sh /scratch/work/masooda1/datasets/jump_preprocessed mocop
```

**What it does**:
- Aggregates original JUMP data across all plates
- Aggregates centered JUMP data
- Cleans up individual plate files to save disk space
- Creates train/validation/test splits
- Generates dummy datasets for testing

**Dependencies**:
- `data/process_data_trirton.sh`
- `data/_jump_aggregate.py`
- `data/cleanup_individual_plates.py`
- `data/jump_data_splits.py`

**Outputs**:
- `centered.filtered.parquet` (Main aggregated dataset)
- `jump_data_splits/` (Train/val/test splits)
- `dummy_data/` (Small test datasets)

### 3. 🧬 LINCS Processing (`process_lincs_data`)

**Purpose**: Processes LINCS L1000 transcriptomic data

**Command**:
```bash
sbatch data/process_LINCS_data.sh /scratch/work/masooda1/datasets/LINCS/ /scratch/work/masooda1/datasets/LINCS_preprocessed/ cmappy_env
```

**What it does**:
- Processes raw LINCS L1000 gene expression data
- Filters by cell lines, compounds, doses, and time points
- Creates visualization plots for data analysis
- Generates both full and test datasets

**Dependencies**:
- `data/process_LINCS_data.sh`
- `data/process_LINCS_data.py`

**Outputs**:
- `landmark_cmp_data_min1000compounds_all_measurements.parquet` (Full dataset)
- Various analysis plots (`.png` files)

### 4. 🤖 Model Training Stages (Available but Commented)

The pipeline includes additional stages for model training that can be uncommented when needed:

- `train_vae`: VAE training on JUMP data
- `train_ae`: Autoencoder training
- `train_jump_mocop`: Multi-modal contrastive learning
- `evaluate_chembl_mocop`: ChEMBL evaluation
- `analyze_hyperparameters`: Results analysis
- `generate_umap_plots`: Visualization generation

## Installation and Setup

### Prerequisites

- Access to SLURM cluster (Triton)
- Conda/Mamba environment manager
- DVC installed

### Environment Setup

```bash
# Load required modules
module load mamba

# Create and activate environment
conda env create --name mocop --file environment.yml
source activate mocop

# Install DVC if not already installed
pip install dvc[all]

# Initialize DVC (first time only)
dvc init
```

## Running the Pipeline

### Complete Pipeline Execution

For long-running processes, use screen sessions:

```bash
# Start a screen session
screen -S dvc_pipeline

# Load modules and activate environment
module load mamba
source activate mocop

# Run the complete pipeline
dvc repro

# Detach from screen (Ctrl+A, then D)
# Reattach later with: screen -r dvc_pipeline
```

### Individual Stage Execution

Run specific stages for testing or debugging:

```bash
# Download data only
dvc repro download_jump_data

# Process JUMP data only
dvc repro process_jump_data

# Process LINCS data only
dvc repro process_lincs_data

# Force re-run a stage (ignore cache)
dvc repro -f process_lincs_data
```

### Pipeline Status

Check pipeline status and dependencies:

```bash
# Check which stages need to be run
dvc status

# Show pipeline DAG
dvc dag

# Show pipeline dependencies
dvc dag --ascii
```

## Data Structure

The pipeline organizes data in the following structure:

```
/scratch/work/masooda1/datasets/
├── jump/                              # Raw JUMP data (extracted from tar)
├── jump_preprocessed/                 # Processed JUMP data
│   ├── centered.filtered.parquet      # Main aggregated dataset
│   ├── original.filtered.parquet      # Non-centered aggregated data
│   ├── jump_data_splits/              # Train/validation/test splits
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── dummy_data/                    # Small datasets for testing
│       ├── train_dummy.parquet
│       └── test_dummy.parquet
├── chembl20/                          # ChEMBL20 bioactivity data
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── LINCS/                             # Raw LINCS L1000 data
│   ├── compoundinfo_beta.txt
│   ├── siginfo_beta.txt
│   ├── GSE92742_Broad_LINCS_gene_info.txt.gz
│   └── level5_beta_trt_cp_n720216x12328.gctx
└── LINCS_preprocessed/                # Processed LINCS data
    ├── landmark_cmp_data_min1000compounds_all_measurements.parquet
    ├── cell_line_compound_distribution.png
    ├── dose_unit_distribution.png
    ├── dose_level_distribution.png
    ├── time_point_distribution.png
    ├── dose_distribution_heatmap.png
    └── time_distribution_heatmap.png
```

### Output Logs

All script outputs are logged to:

```
/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/
├── process_jump_data.out              # JUMP processing logs
├── process_lincs_data.out             # LINCS processing logs
└── slurm-*.out                        # Individual SLURM job logs
```

## Configuration Options

### LINCS Processing Parameters

The LINCS processing stage accepts extensive customization options:

```bash
# Basic usage with defaults
bash data/process_LINCS_data.sh /input/dir /output/dir cmappy_env

# Advanced usage with custom parameters
bash data/process_LINCS_data.sh /input/dir /output/dir cmappy_env \
    --cell_line "U2OS" \
    --min_compounds 500 \
    --test_mode true \
    --filter_dose_time true \
    --dose_min 0.01 \
    --dose_max 50 \
    --time_points 6 12 24 \
    --drop_multiple_measurements true \
    --convert_dose_to_bins true
```

#### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cell_line` | string | None | Specific cell line to filter (e.g., "U2OS") |
| `--min_compounds` | int | 1000 | Minimum compounds per cell line |
| `--test_mode` | true/false | false | Use only 1000 compounds for testing |
| `--filter_dose_time` | true/false | true | Filter by dose and time constraints |
| `--dose_min` | float | 0.001 | Minimum dose level (µM) |
| `--dose_max` | float | 100 | Maximum dose level (µM) |
| `--time_points` | list | [6, 24] | Time points to include (hours) |
| `--drop_multiple_measurements` | true/false | false | Keep only single dose/time per compound |
| `--convert_dose_to_bins` | true/false | true | Convert doses to logarithmic bins |

### Modifying DVC Pipeline

To customize the pipeline, edit `dvc.yaml`:

```yaml
# Example: Add custom parameters to LINCS processing
process_lincs_data:
  cmd: sbatch data/process_LINCS_data.sh 
       /scratch/work/masooda1/datasets/LINCS/ 
       /scratch/work/masooda1/datasets/LINCS_preprocessed/ 
       cmappy_env 
       --min_compounds 500 
       --test_mode true
  deps:
    - data/process_LINCS_data.sh
    - data/process_LINCS_data.py
  desc: "Process LINCS data with custom parameters"
```

## Monitoring and Debugging

### SLURM Job Monitoring

```bash
# Check job queue
squeue -u $USER

# Check specific job details
scontrol show job <job_id>

# Check job output (while running)
tail -f script_outputs/process_jump_data.out
```

### DVC Monitoring

```bash
# Check what stages need to run
dvc status

# Check pipeline execution progress
dvc metrics show

# View pipeline DAG
dvc dag
```

### Log Analysis

Check detailed logs for debugging:

```bash
# View recent JUMP processing logs
tail -50 script_outputs/process_jump_data.out

# View LINCS processing logs
tail -50 script_outputs/process_lincs_data.out

# Check SLURM array job logs
ls script_outputs/slurm-*.out
```

## Troubleshooting

### Common Issues

#### 1. SLURM Job Failures

**Problem**: Job fails with "sbatch: error: invalid partition"
```bash
# Solution: Let SLURM auto-select partition
# Remove or comment out #SBATCH --partition lines in scripts
```

**Problem**: Job runs out of memory or time
```bash
# Solution: Increase resources in script headers
#SBATCH --time=04:00:00  # Increase time
#SBATCH --mem=80G        # Increase memory
```

#### 2. Environment Issues

**Problem**: "conda: command not found"
```bash
# Solution: Load mamba module first
module load mamba
source activate mocop
```

**Problem**: Missing Python packages
```bash
# Solution: Install missing packages
conda activate mocop
pip install missing_package
```

#### 3. Data Access Issues

**Problem**: "Required file not found"
```bash
# Solution: Check data paths and permissions
ls -la /scratch/work/masooda1/datasets/LINCS/
# Ensure all required files are present
```

#### 4. DVC Issues

**Problem**: "dvc.lock is git-ignored"
```bash
# Solution: Update .gitignore
# Remove dvc.lock from .gitignore file
```

**Problem**: External outputs not supported
```bash
# Solution: Remove external paths from 'outs:' sections in dvc.yaml
# Keep only 'deps:' for external data
```

### Debug Mode

Run individual scripts in debug mode:

```bash
# Run Python scripts directly for debugging
python data/process_LINCS_data.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --test_mode true

# Run shell scripts with verbose output
bash -x data/process_LINCS_data.sh /input /output cmappy_env
```

### Performance Optimization

For faster processing:

1. **Use test mode** for development:
   ```bash
   # Process only 1000 compounds
   --test_mode true
   ```

2. **Parallel processing**: SLURM array jobs are already optimized

3. **Resource allocation**: Adjust memory/CPU based on data size

### Getting Help

- **Script documentation**: Check comments in individual script files
- **DVC documentation**: https://dvc.org/doc
- **SLURM documentation**: Check cluster-specific documentation
- **Issues**: Report issues on the project GitHub repository 