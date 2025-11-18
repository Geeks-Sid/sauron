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
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
echo "Activating conda environment: aegis"
conda activate aegis
echo "Conda environment activated. Python path: $(which python)"

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
    --weighted_sample \
    --batch_size 16 \
    --use_hdf5 \
    --n_subsamples 2048 \
    --num_workers 16

# Example: To use a different model, change --model_type:
# --model_type trans_mil --activation gelu
# --model_type att_mil --activation relu
# --model_type moemil --embed_dim 512 --num_experts 4
# --model_type mambamil --mamba_layers 2 --mamba_rate 10 --mamba_type SRMamba
