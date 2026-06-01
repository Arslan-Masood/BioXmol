# BioXMol: Unifying Disjoint Phenotypic Contexts: A Multimodal Soft Contrastive Approach to Identify DILI Activity Cliffs

Muhammad Arslan Masood, Tianyu Cui, Markus Heinonen, Samuel Kaski

[[`ICLR 2025 Workshop LMRL`](https://openreview.net/forum?id=WT7BpLvL6D)] — earlier version of this work.
A full manuscript (DILI activity-cliff evaluation) has been submitted to *Journal of Cheminformatics*.

<p align="center">
  <img src="Triencoder_contrastive_v9.png" width="100%" alt="BioXMol Multimodal Contrastive Architecture"/>
</p>

## Overview

BioXMol learns molecular representations by aligning chemical structure with two biological
modalities — Cell Painting morphology (JUMP-CP) and L1000 transcriptomics (LINCS) — under a
**soft contrastive objective**. Biological context is used only during pretraining; at inference
the model operates on molecular structure alone, so the encoder applies to novel compounds with
no assay data.

The framework is evaluated on drug-induced liver injury (DILI) prediction, with particular
emphasis on **activity cliffs**: structurally similar compounds with divergent toxicity outcomes
that structure-based representations systematically fail to separate.

## Key contributions

- **Partial pairing.** Pretraining requires only molecule–morphology and molecule–transcriptomics
  pairs, not complete triplets. Because JUMP-CP and LINCS profile largely non-overlapping compound
  sets, this allows pretraining over the union of two disjoint phenotypic datasets.

- **Soft contrastive objective.** Standard contrastive learning treats all non-matching pairs as
  equally dissimilar negatives — a biologically implausible assumption that erases graded similarity
  between structurally related compounds. BioXMol replaces hard negatives with soft targets derived
  from EMA-updated teacher networks, weighting negative repulsion by biological similarity in the
  teacher's latent space. This preserves the graded structure needed to resolve activity cliffs.

## Results (DILIRank 2.0)

Standard scaffold-split cross-validation does not distinguish representations: all models, including
ECFP, fall within overlapping confidence intervals. The differences appear only under activity-cliff
evaluation (pairwise ranking accuracy on held-out structural-analog pairs):

| Representation              | Activity-cliff pairwise accuracy |
|-----------------------------|----------------------------------|
| **BioXMol-Soft**            | **80.7%**                        |
| ECFP                        | 60.7%                            |
| BioXMol-Hard (ablation)     | 51.0% (≈ chance)                 |

The hard-contrastive ablation is trained on identical data and architecture, isolating the soft
objective as the source of the gain. The choice of evaluation protocol — not the metric — is what
makes the representation differences detectable.

## Datasets

| Dataset       | Molecules | Role         | Modality                       |
|---------------|-----------|--------------|--------------------------------|
| JUMP-CP       | ~120K     | Pretraining  | Morphological (3,479-dim)      |
| LINCS L1000   | ~28K      | Pretraining  | Transcriptomic (978-dim)       |
| DILIRank 2.0  | 1,336     | Downstream   | Ordinal severity (3 classes)   |

JUMP-CP morphological profiles are sourced from the Cell Painting Gallery
(`cpg0016-jump`, https://registry.opendata.aws/cellpainting-gallery/). LINCS L1000 signatures are
accessed through CLUE (https://clue.io/). DILIRank 2.0 provides three ordered hepatotoxicity labels
(*vNo-*, *vLess-*, *vMost-DILI-concern*); 15 structural-analog pairs with divergent labels are
reserved as a held-out activity-cliff set.

## Architecture

- **Molecular encoder** — graph neural network over molecular graphs.
- **Morphological / transcriptomic encoders** — initialized from independently pretrained
  modality-specific autoencoders (the transcriptomic encoder also takes learned condition
  embeddings for cell line, time point, and dose).
- During contrastive pretraining each biological encoder is instantiated twice: a **student**
  updated by gradient descent and a **teacher** updated by an exponential moving average of the
  student (momentum *m* = 0.999). The teacher supplies the stable soft similarity targets.
- All modalities project into a shared 128-dimensional, L2-normalized embedding space.

The momentum-teacher design follows MoCo and DINO. The forward direction (molecule → biology)
minimizes a soft cross-entropy against teacher self-similarity targets; the backward direction
(biology → molecule) uses a hard cross-entropy.

## Installation

```bash
git clone https://github.com/Arslan-Masood/BioXmol.git
cd BioXmol
conda env create --name bioxmol --file environment.yml
conda activate bioxmol
```

### Data Version Control (DVC)

Large datasets and model artifacts are managed with DVC rather than committed to git.
See [DVC_SETUP.md](DVC_SETUP.md) for full instructions.

```bash
pip install dvc[all]
dvc init          # first time only
dvc pull          # fetch data/checkpoints from remote storage
```

## DVC Pipeline

The pipeline automates data processing and training:

1. **Data download** — JUMP Cell Painting data download and extraction.
2. **Data processing** — JUMP aggregation, split generation.
3. **LINCS processing** — L1000 transcriptomic preprocessing.
4. **Model training** — autoencoder pretraining, multimodal contrastive pretraining, downstream
   evaluation (training stages available but commented by default).

```bash
screen -S dvc_pipeline      # for long-running jobs
module load mamba
source activate bioxmol
dvc repro
```

## Training

```bash
python bin/train.py \
    --config configs/multi_modal_config.yml \
    --molecular_data    path/to/molecular_data.csv \
    --morphological_data path/to/morphological_data.csv \
    --transcriptomic_data path/to/transcriptomic_data.csv \
    --output_dir results/
```

Expected input formats:

```csv
# molecular_data.csv
smiles,compound_id
CCO,compound_1

# morphological_data.csv
compound_id,feature_1,...,feature_n
compound_1,0.123,...,0.789

# transcriptomic_data.csv
compound_id,gene_1,...,gene_m
compound_1,1.23,...,3.45
```

## Downstream Evaluation (DILI)

Linear probing with an ordinal logistic-regression head (Frank–Hall decomposition over the three
ordered severity classes), evaluated under 5×5 nested cross-validation with Murcko scaffold
splitting and a held-out activity-cliff set:

```bash
python DILI_linear_probing_3_classes_ordinal.py \
    --features_file path/to/features.csv \
    --label_col     vDILI-Concern_standardized \
    --output_dir    results/dili/ \
    --seed 42
```

The primary cliff metric is pairwise ranking accuracy: the fraction of analog pairs for which the
expected severity E[Y] of the more-toxic compound exceeds that of the safer one.

## Citation

```bibtex
@inproceedings{masood2025multimodal,
    title     = {Multi-Modal Representation learning for molecules},
    author    = {Muhammad Arslan Masood and Markus Heinonen and Samuel Kaski},
    booktitle = {ICLR 2025 Workshop Learning Meaningful Representations of Life (LMRL)},
    year      = {2025},
    url        = {https://openreview.net/forum?id=WT7BpLvL6D}
}
```

The full DILI activity-cliff manuscript is in preparation; this section will be updated on
publication.

## License

Code is released under the [GPLv3 license](LICENSE-GPLv3); model weights under
[CC-BY-NC-ND 4.0](LICENSE-CC-BY-NC-ND-4.0).

## Contact

- Muhammad Arslan Masood — arslan.masood@aalto.fi
- Issues and discussions: [GitHub Issues](https://github.com/Arslan-Masood/BioXmol/issues)

---

**Keywords:** multimodal representation learning, soft contrastive learning, drug-induced liver
injury, activity cliffs, Cell Painting, LINCS L1000, DILIRank