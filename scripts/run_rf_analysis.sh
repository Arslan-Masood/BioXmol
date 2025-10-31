#!/bin/bash -l
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:0
#SBATCH --array=0-2
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/rf_analysis_%A_%a.out

# Arguments
CONFIG_FILE=$1
OUTPUT_DIR=$2
CONDA_ENV=$3

# Define layer options for array
LAYER_OPTIONS=("GNN" "first_fc" "second_fc")
LAYER_NAME=${LAYER_OPTIONS[$SLURM_ARRAY_TASK_ID]}

# Define additional parameters
BATCH_SIZE=32
SMILES_COL="protonated_smiles_r"
LABEL_COL="TOXICITY"
N_JOBS=${SLURM_CPUS_PER_TASK:-40}  # Use SLURM_CPUS_PER_TASK, default to 40

echo "Starting RF Analysis with Embeddings (Array Job)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Conda Environment: ${CONDA_ENV}"
echo "Batch Size: ${BATCH_SIZE}"
echo "SMILES Column: ${SMILES_COL}"
echo "Label Column: ${LABEL_COL}"
echo "Config File: ${CONFIG_FILE}"
echo "Layer Name: ${LAYER_NAME}"
echo "Number of Jobs (CPUs): ${N_JOBS}"

# Activate conda environment
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Check if config file exists (try both .yml and .yaml extensions)
if [ ! -f "/scratch/work/masooda1/Multi_Modal_Contrastive/configs/${CONFIG_FILE}" ] && [ ! -f "/scratch/work/masooda1/Multi_Modal_Contrastive/configs/${CONFIG_FILE%.yml}.yaml" ]; then
    echo "Error: Config file not found at /scratch/work/masooda1/Multi_Modal_Contrastive/configs/${CONFIG_FILE} or /scratch/work/masooda1/Multi_Modal_Contrastive/configs/${CONFIG_FILE%.yml}.yaml"
    exit 1
fi

# Data paths are now defined in the config file

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Running RF analysis script..."

# Run the Python script with Hydra configuration
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/scripts/extract_embeddings_and_train_rf.py \
    -cn ${CONFIG_FILE} \
    layer="${LAYER_NAME}" \
    batch_size=${BATCH_SIZE} \
    data.smiles_col="${SMILES_COL}" \
    data.label_col="${LABEL_COL}" \
    output.output_dir="${OUTPUT_DIR}" \
    random_forest.halving_search.n_jobs=${N_JOBS} \
    random_forest.param_distributions.n_jobs=[${N_JOBS}]

if [ $? -ne 0 ]; then
    echo "Error: RF analysis failed."
    exit 1
fi

echo "RF analysis completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}/layer_${LAYER_NAME}"
echo "Array task ${SLURM_ARRAY_TASK_ID} completed for layer: ${LAYER_NAME}"
