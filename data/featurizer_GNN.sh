#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=featurizer_gnn
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/featurizer_gnn.out

# Configuration
INPUT_FILE=/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/features/DILIrank_2.0_normalized.csv
BASE_DIR=/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILIRank_v2_FineTuning
SMILES_COL=SMILES_Normalized
BATCH_SIZE=32
DEVICE=cuda
CONDA_ENV=mocop
LR=5.0e-05

# Define lists
LAYERS=("GNN" "first_fc" "second_fc")
MODEL_TYPES=(
    "Soft_Clip_with_Frozen_Teacher"
    "Soft_Clip_with_Teacher"
    "Soft_Clip_with_Teacher_with_centering"
    "Vanilla_Clip_with_VAE"
    "Vanilla_Clip_without_VAE"
)
SEEDS=(0 1 2)

echo "🚀 Starting GNN Feature Extraction for All Models"
echo "=========================================="
echo "📁 Input file: $INPUT_FILE"
echo "📂 Base directory: $BASE_DIR"
echo "🔬 Layers to extract: ${LAYERS[@]}"
echo "🏷️  Model types: ${MODEL_TYPES[@]}"
echo "🎲 Seeds: ${SEEDS[@]}"
echo "📊 SMILES column: $SMILES_COL"
echo "📦 Batch size: $BATCH_SIZE"
echo "💻 Device: $DEVICE"
echo "🐍 Conda environment: $CONDA_ENV"
echo "=========================================="

# Validate required files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: Base directory not found: $BASE_DIR"
    exit 1
fi

# Activate conda environment
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment: ${CONDA_ENV}"
    exit 1
fi

# Calculate total combinations
TOTAL_COMBOS=$((${#MODEL_TYPES[@]} * ${#SEEDS[@]} * ${#LAYERS[@]}))
CURRENT=0
FAILED=0

# Loop through model types
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    # Loop through seeds
    for SEED in "${SEEDS[@]}"; do
        # Construct checkpoint path
        CHECKPOINT_PATH="${BASE_DIR}/${MODEL_TYPE}_seed${SEED}_LR_${LR}/extracted_molecular_encoders/molecular_encoder_seed_${SEED}.ckpt"
        
        # Check if checkpoint exists
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "⚠️  Warning: Checkpoint not found, skipping: $CHECKPOINT_PATH"
            continue
        fi
        
        echo ""
        echo "=========================================="
        echo "Processing: $MODEL_TYPE (seed $SEED)"
        echo "=========================================="
        echo "🧠 Checkpoint: $(basename $CHECKPOINT_PATH)"
        
        # Loop through each layer
        for LAYER_NAME in "${LAYERS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "  [$CURRENT/$TOTAL_COMBOS] Processing layer: $LAYER_NAME"
            
            # Generate output file name with model info (model and seed before layer)
            INPUT_DIR=$(dirname "$INPUT_FILE")
            INPUT_BASENAME=$(basename "$INPUT_FILE" .csv)
            # Clean model type name for filename (replace spaces/special chars)
            MODEL_NAME_CLEAN=$(echo "$MODEL_TYPE" | sed 's/[^a-zA-Z0-9_]/_/g')
            OUTPUT_FILE="${INPUT_DIR}/${INPUT_BASENAME}_GNN_${MODEL_NAME_CLEAN}_seed${SEED}_${LAYER_NAME}.csv"
            
            echo "  📤 Output: $(basename $OUTPUT_FILE)"
            
            # Run the Python script for this layer
            srun python /scratch/work/masooda1/Multi_Modal_Contrastive/data/featurizer_GNN.py \
                --input_file "${INPUT_FILE}" \
                --checkpoint_path "${CHECKPOINT_PATH}" \
                --layer_name "${LAYER_NAME}" \
                --smiles_col "${SMILES_COL}" \
                --batch_size ${BATCH_SIZE} \
                --output_file "${OUTPUT_FILE}" \
                --device "${DEVICE}"
            
            # Check if command succeeded
            if [ $? -ne 0 ]; then
                echo "  ❌ Error: Failed for $MODEL_TYPE seed $SEED layer $LAYER_NAME"
                FAILED=$((FAILED + 1))
            else
                echo "  ✅ Completed: $(basename $OUTPUT_FILE)"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "✅ GNN Feature Extraction Summary"
echo "=========================================="
echo "Total extractions: $CURRENT"
if [ $FAILED -gt 0 ]; then
    echo "❌ Failed: $FAILED"
    echo "✅ Successful: $((CURRENT - FAILED))"
else
    echo "✅ All extractions completed successfully!"
fi
echo ""
INPUT_BASENAME=$(basename "$INPUT_FILE" .csv)
echo "Output files saved to: $(dirname "$INPUT_FILE")"
echo "File naming pattern: ${INPUT_BASENAME}_GNN_{model_type}_seed{seed}_{layer}.csv"
