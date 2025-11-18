#!/bin/bash
# Script to run MIL training with specified parameters
#
# NOTE: If using mambamil model, ensure mamba-ssm is installed:
#   pip install mamba-ssm
#   or
#   pip install causal-conv1d>=1.2.0
#   (CUDA extensions will be compiled automatically)
#
# Available MIL model types (--model_type):
#   - att_mil        : Attention-based MIL (ABMIL/DAttention)
#   - trans_mil       : Transformer-based MIL (TransMIL)
#   - max_mil         : Max pooling MIL
#   - mean_mil        : Mean pooling MIL
#   - s4model         : S4-based MIL
#   - wikgmil         : WiKG (Graph-based MIL)
#   - diffabmil       : Differentiable Attention MIL
#   - hgachc          : Hierarchical Graph Attention Cross-Head Communication
#   - rrtmil          : RRT (Region-based Transformer MIL)
#   - dsmil           : Dual-stream MIL
#   - mambamil        : Mamba-based MIL (supports SRMamba, Mamba, BiMamba)
#   - moemil          : Mixture of Experts MIL
#
# Model-specific parameters:
#   - mambamil: --mamba_layers (default: 2), --mamba_rate (default: 10), --mamba_type (SRMamba/Mamba/BiMamba)
#   - moemil: --embed_dim (default: 512), --num_experts (default: 4)
#   - trans_mil, att_mil, max_mil, mean_mil, s4model: --activation (relu/gelu, default varies by model)

set -e  # Exit on error

echo "Starting MIL training script..."
echo "Current directory: $(pwd)"

# Initialize conda for bash script
# Try multiple methods to initialize conda
CONDA_BASE=""
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda initialization script"
    echo "Please ensure conda is installed and accessible"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: aegis"
# Initialize conda for this shell
eval "$(conda shell.bash hook)"
conda activate aegis || {
    echo "Error: Failed to activate conda environment 'aegis'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
}
echo "Conda environment activated. Python path: $(which python)"

# Verify Python script exists
if [ ! -f "train_mil_run.py" ]; then
    echo "Error: train_mil_run.py not found in current directory: $(pwd)"
    echo "Please ensure you're running this script from the project root directory"
    exit 1
fi

echo "Launching training..."
python -u train_mil_run.py \
    --data_root_dir /mnt/e/features_uni_v2 \
    --dataset_csv Data/tcga-ot_train.csv \
    --label_col OncoTreeCode \
    --patient_id_col case_id \
    --slide_id_col slide_id \
    --results_dir ./results \
    --task multiclass \
    --task_type classification \
    --exp_code tcga_ot_multiclass_s1 \
    --seed 42 \
    --num_workers 16 \
    --log_data \
    --testing \
    --k 1 \
    --k_start 0 \
    --k_end 1 \
    --model_type att_mil \
    --backbone uni_v2 \
    --in_dim 1536 \
    --max_epochs 200 \
    --lr 1e-4 \
    --reg 1e-5 \
    --opt adam \
    --drop_out 0.25 \
    --early_stopping \
    --preloading no \
    --weighted_sample \
    --batch_size 4 \
    --use_hdf5 \
    --n_subsamples 2048 \
    

# Example: To use a different model, change --model_type:
# --model_type trans_mil --activation gelu
# --model_type att_mil --activation relu
# --model_type moemil --embed_dim 512 --num_experts 4
# --model_type mambamil --mamba_layers 2 --mamba_rate 10 --mamba_type SRMamba
