#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=process_dilirank_v2
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/process_dilirank_v2.out

# Check if arguments are provided (3 required, 2 optional)
if [ $# -lt 3 ] || [ $# -gt 5 ]; then
    echo "❌ ERROR: This script requires 3 required arguments and 2 optional arguments"
    echo "Usage: $0 <input_file> <output_dir> <env_step2_3> [checkpoint_path] [gnn_layers]"
    echo "  <input_file>: Pre-processed DILIRank CSV with SMILES and labels"
    echo "Example (without GNN): $0 /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/Master_DILIRank_Final_Cleaned_with_labels.csv /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2 mocop"
    echo "Example (with GNN): $0 /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/pubchem_data/Master_DILIRank_Final_Cleaned_with_labels.csv /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2 mocop /path/to/checkpoint.ckpt \"GNN,first_fc,second_fc\""
    exit 1
fi

# Get required arguments
INPUT_FILE=$1
OUTPUT_DIR=$2
ENV_STEP2_3=$3

# Get optional arguments (for GNN extraction)
CHECKPOINT_PATH=${4:-""}
GNN_LAYERS=${5:-"GNN,first_fc,second_fc"}

echo "🚀 Starting DILIRank v2.0 data processing pipeline..."
echo "📁 Input file: $INPUT_FILE"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🐍 Processing environment: $ENV_STEP2_3"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "🧠 GNN checkpoint: $CHECKPOINT_PATH"
    echo "🧠 GNN layers to extract: $GNN_LAYERS"
else
    echo "ℹ️  GNN extraction: Skipped (no checkpoint provided)"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define intermediate file paths
NORMALIZED_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized.csv"
ECFP_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ECFP_1024_2.csv"
MOLFORMER_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_MolFormer_MoLFormer-XL-both-10pct.csv"
CHEMBERTA_MTR_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ChemBERTa-MTR.csv"
CHEMBERTA_MLM_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ChemBERTa-MLM.csv"
GNN_GNN_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_GNN.csv"
GNN_FIRST_FC_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_first_fc.csv"
GNN_SECOND_FC_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_second_fc.csv"
FDA_PRODUCTS_FILE="$OUTPUT_DIR/products.txt"
DRUGBANK_FILE="/scratch/work/masooda1/datasets/downstream_datasets/drugbank.tsv"

# Define worker counts
# Use SLURM allocated CPUs if available
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-}

# ============================================
# Step 1: Normalize SMILES (uses env_step2_3 environment)
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Normalizing SMILES"
echo "=========================================="

if [ -f "$NORMALIZED_FILE" ]; then
    echo "⏭️  Skipping Step 1: Output file already exists"
    echo "   Found: $NORMALIZED_FILE"
else
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ Error: Input file not found: $INPUT_FILE"
        echo "   Please provide a pre-processed CSV with SMILES and labels."
        exit 1
    fi
    
    echo "🐍 Activating conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    if [ -n "$NUM_WORKERS" ]; then
        python /scratch/work/masooda1/Multi_Modal_Contrastive/data/SMILES_normalization.py \
                --input_file "$INPUT_FILE" \
                --output_file "$NORMALIZED_FILE" \
                --num_workers "$NUM_WORKERS"
    else
        python /scratch/work/masooda1/Multi_Modal_Contrastive/data/SMILES_normalization.py \
                --input_file "$INPUT_FILE" \
                --output_file "$NORMALIZED_FILE"
    fi

    # Check if Step 2 succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: SMILES normalization script failed."
        exit 1
    fi
fi

# ============================================
# Step 2: Compute molecular features
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Computing molecular features"
echo "=========================================="

if [ -f "$ECFP_FEATURES_FILE" ] && [ -f "$MOLFORMER_FEATURES_FILE" ] && \
   [ -f "$CHEMBERTA_MTR_FEATURES_FILE" ] && [ -f "$CHEMBERTA_MLM_FEATURES_FILE" ]; then
    echo "⏭️  Skipping Step 2: All feature files already exist"
    echo "   Found: $(basename $ECFP_FEATURES_FILE)"
    echo "   Found: $(basename $MOLFORMER_FEATURES_FILE)"
    echo "   Found: $(basename $CHEMBERTA_MTR_FEATURES_FILE)"
    echo "   Found: $(basename $CHEMBERTA_MLM_FEATURES_FILE)"
else
    # Check if input file exists
    if [ ! -f "$NORMALIZED_FILE" ]; then
        echo "❌ Error: Normalized file not found: $NORMALIZED_FILE"
        echo "   Please run SMILES normalization first."
        exit 1
    fi
    
    echo "🐍 Ensuring conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    # Compute features for normalized dataset (all types: ECFP, MolFormer, ChemBERTa-MTR, ChemBERTa-MLM)
    echo "Computing features for normalized dataset..."
    echo "  This will generate 4 feature files: ECFP, MolFormer, ChemBERTa-MTR, and ChemBERTa-MLM..."
    python /scratch/work/masooda1/Multi_Modal_Contrastive/data/fearurizer.py \
            --input_file "$NORMALIZED_FILE" \
            --smiles_col "SMILES_Normalized" \
            --size 1024 \
            --radius 2 \
            --device "cuda"

    # Check if features computation succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: Features computation failed."
        exit 1
    fi
fi

echo "✅ DILIRank v2.0 processing pipeline completed successfully!"
echo "📁 Output files in: $OUTPUT_DIR"