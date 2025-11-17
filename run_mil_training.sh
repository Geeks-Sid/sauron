#!/bin/bash
# Script to run MIL training with specified parameters

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
    --seed 1 \
    --log_data \
    --testing \
    --k 1 \
    --k_start 0 \
    --k_end 1 \
    --model_type mambamil \
    --backbone resnet50 \
    --in_dim 1024 \
    --max_epochs 200 \
    --lr 1e-4 \
    --reg 1e-5 \
    --opt adam \
    --drop_out 0.25 \
    --early_stopping \
    --weighted_sample \
    --batch_size 2 \
    --use_hdf5
