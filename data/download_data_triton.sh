#!/bin/bash
OUTPUT_DIR=$1
CONDA_ENV=$2

MAMBA_CMD="/appl/scibuilder-mamba/aalto-rhel9/prod/software/mamba/2024-01/39cf5e1/bin/mamba"
VENV_PATH="/scratch/work/masooda1/.conda_envs/${CONDA_ENV}"
SCRIPT_OUTPUT_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs"

echo "Uncompressing ChEMBL20 data and splits"
tar -xzvf data/chembl20.tar.gz --directory data/

echo "Uncompressing JUMP-CP splits"
tar -xzvf data/jump.tar.gz --directory data/

echo "Cloning JUMP-CP metadata repo"
git clone https://github.com/jump-cellpainting/datasets

METADATA_PATH=datasets
# Ensure the output directory exists
mkdir -p ${SCRIPT_OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

echo "Downloading and normalizing JUMP-CP compound plates"
sbatch  --time=00:60:00 \
        --mem=40G \
        --array=0-1729 \
        --cpus-per-task=1 \
        --wait \
        --output=${SCRIPT_OUTPUT_DIR}/slurm-%A_%a.out \
        --export=ALL,MAMBA_CMD=${MAMBA_CMD},CONDA_ENV=${CONDA_ENV},OUTPUT_DIR=${OUTPUT_DIR},METADATA_PATH=${METADATA_PATH} \
        --wrap "source /appl/scibuilder-mamba/aalto-rhel9/prod/software/mamba/2024-01/39cf5e1/etc/profile.d/conda.sh && \
                conda activate \${CONDA_ENV} && \
                python data/_jump_download_single_plate.py -o \${OUTPUT_DIR} -m \${METADATA_PATH}"

#echo "Aggregating and cleaning JUMP-CP"
#python data/_jump_aggregate.py -d $OUTPUT_DIR -o $OUTPUT_DIR --is_centered
echo "Done!"
