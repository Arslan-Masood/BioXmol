# Multi-Modal Representation Learning for Molecules

Muhammad Arslan Masood, Markus Heinonen, Samuel Kaski

[[`ICLR 2025 Workshop LMRL`](https://openreview.net/forum?id=WT7BpLvL6D)]
<p align="center">
  <img src="Triencoder_contrastive.png" width="100%" alt="Multi-Modal Contrastive Learning Architecture"/>
</p>

## Abstract

Molecular representation learning is a fundamental challenge in AI-driven drug discovery, with traditional unimodal approaches relying solely on chemical structures often failing to capture the biological context necessary for accurate toxicity and activity predictions. 

We propose a **multimodal representation learning framework** that integrates molecular data with biological modalities, including:
- **Morphological features** from Cell Painting assays
- **Transcriptomic profiles** from the LINCS L1000 dataset

### Key Innovation
Unlike traditional approaches that require complete triplets (molecule, morphological, genomic), our model only requires **paired data**—(molecule-morphological) and (molecule-genomic)—making it more practical and scalable.

Our approach leverages **contrastive learning** to align molecular representations with biological data, even in the absence of fully paired datasets. We evaluate our framework on the **ChEMBL20 dataset** using linear probing across **1,320 tasks**, demonstrating improvements in predictive performance.

## Features

- ✅ **Multimodal Integration**: Combines chemical structures with biological modalities
- ✅ **Flexible Pairing**: Works with paired data instead of requiring complete triplets
- ✅ **Contrastive Learning**: Aligns molecular and biological representations
- ✅ **Scalable Framework**: Practical for real-world drug discovery applications
- ✅ **Comprehensive Evaluation**: Tested on 1,320 ChEMBL20 tasks

## Installation and Setup

#### Cloning and setting up your environment
```bash
git clone https://github.com/Arslan-Masood/Multi_Modal_Contrastive.git
cd Multi_Modal_Contrastive
conda env create --name multi_modal_contrastive --file environment.yml
conda activate multi_modal_contrastive
```

#### Data Version Control (DVC) Setup
This project uses DVC for managing large datasets and model files. See [DVC_SETUP.md](DVC_SETUP.md) for detailed instructions.

```bash
# Install DVC
pip install dvc[all]

# Initialize DVC (first time only)
dvc init

# Pull data from remote storage
dvc pull
```

## DVC Pipeline

This project includes a comprehensive DVC pipeline for automated data processing and model training. The pipeline handles:

1. **📥 Data Download**: Downloads and extracts JUMP Cell Painting data
2. **🔄 Data Processing**: Aggregates JUMP data, creates splits, and generates dummy datasets  
3. **🧬 LINCS Processing**: Processes LINCS L1000 transcriptomic data
4. **🤖 Model Training**: VAE/AE training, multi-modal contrastive learning, ChEMBL evaluation (stages available but commented)

### Quick Start

```bash
# Start a screen session for long-running processes
screen -S dvc_pipeline

# Load required modules and activate environment
module load mamba
source activate mocop

# Run the complete DVC pipeline
dvc repro
```

For detailed pipeline documentation, configuration options, monitoring instructions, and troubleshooting, see **[DVC_SETUP.md](DVC_SETUP.md)**.

## Methodology

### 1. Data Modalities

Our framework integrates three key data modalities:

- **Chemical Structures**: SMILES representations and molecular graphs
- **Morphological Features**: Cell Painting assay data capturing cellular morphology changes
- **Transcriptomic Profiles**: LINCS L1000 gene expression data

### 2. Contrastive Learning Framework

We employ a contrastive learning approach that:
- Learns shared representations across modalities
- Handles missing modality pairs gracefully
- Maximizes agreement between related molecular and biological data
- Minimizes agreement between unrelated pairs

### 3. Architecture

The model consists of:
- **Molecular Encoder**: Processes chemical structures using graph neural networks
- **Morphological Encoder**: Handles Cell Painting features
- **Transcriptomic Encoder**: Processes gene expression profiles
- **Projection Heads**: Map each modality to a shared representation space

## Quick Start

#### Training the Multi-Modal Model

Prepare your datasets in the required format:

**Molecular Data (CSV format):**
```csv
smiles,compound_id
CCO,compound_1
c1ccccc1,compound_2
...
```

**Morphological Data:**
```csv
compound_id,feature_1,feature_2,...,feature_n
compound_1,0.123,0.456,...,0.789
compound_2,0.234,0.567,...,0.890
...
```

**Transcriptomic Data:**
```csv
compound_id,gene_1,gene_2,...,gene_m
compound_1,1.23,2.34,...,3.45
compound_2,2.34,3.45,...,4.56
...
```

#### Training Command
```bash
python bin/train.py \
    --config configs/multi_modal_config.yml \
    --molecular_data path/to/molecular_data.csv \
    --morphological_data path/to/morphological_data.csv \
    --transcriptomic_data path/to/transcriptomic_data.csv \
    --output_dir results/
```

#### Fine-tuning on Downstream Tasks
```bash
python bin/train.py \
    --config configs/finetune_config.yml \
    --pretrained_model path/to/pretrained_model.ckpt \
    --task_data path/to/chembl20_tasks.csv \
    --output_dir results/finetune/
```

## Reproducing Paper Results

### Environment Setup
Set the necessary environment variables:
```bash
export DATA_DIR=/path/to/processed/data
export SAVE_DIR=/path/to/model/outputs
export CONDA_ENV=multi_modal_contrastive
```

### Data Preparation
```bash
# Download and preprocess ChEMBL20, Cell Painting, and LINCS data
bash data/download_and_preprocess.sh $DATA_DIR $CONDA_ENV
```

### Training
```bash
# Multi-modal contrastive pretraining
bash exp/train_multimodal_contrastive.sh $SAVE_DIR $DATA_DIR $CONDA_ENV

# ChEMBL20 evaluation with linear probing
bash exp/evaluate_chembl20_linear.sh $SAVE_DIR $DATA_DIR $CONDA_ENV
```

## Results

Our multi-modal approach demonstrates:
- **Improved performance** on ChEMBL20 tasks compared to unimodal baselines
- **Better generalization** across diverse molecular property prediction tasks
- **Robust representations** that capture both chemical and biological context
- **Scalable training** without requiring complete multimodal triplets

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{masood2025multimodal,
    title={Multi-Modal Representation learning for molecules},
    author={Muhammad Arslan Masood and Markus Heinonen and Samuel Kaski},
    booktitle={ICLR 2025 Workshop Learning Meaningful Representations of Life (LMRL)},
    year={2025},
    url={https://openreview.net/forum?id=WT7BpLvL6D}
}
```

## License

This project is released under the [GPLv3 license](LICENSE-GPLv3) for code and [CC-BY-NC-ND 4.0 license](LICENSE-CC-BY-NC-ND-4.0) for model weights.

## Contact

For questions and collaborations, please contact:
- Muhammad Arslan Masood (arslan.masood@aalto.fi)
- Issues and discussions: [GitHub Issues](https://github.com/Arslan-Masood/Multi_Modal_Contrastive/issues)

---

**Keywords:** Multi-Modal, Representation learning, drug design, Contrastive learning, Cell Painting, LINCS L1000, ChEMBL20