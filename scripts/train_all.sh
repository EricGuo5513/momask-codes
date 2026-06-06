#!/bin/bash
# Train Stage 2 (MaskTransformer) then Stage 3 (ResidualTransformer)
# Usage: bash scripts/train_all.sh
# Logs go to: log/t2m/ and log/res/
set -e

GPU_ID=${1:-0}
DATASET=t2m
VQ_NAME=RVQVAE

echo "=== Stage 2: MaskTransformer ==="
echo "Started: $(date)"

python train_t2m_transformer.py \
    --dataset_name $DATASET \
    --name MaskTransformer \
    --vq_name $VQ_NAME \
    --gpu_id $GPU_ID \
    --batch_size 64 \
    --max_epoch 500 \
    --latent_dim 384 \
    --n_layers 8 \
    --n_heads 6 \
    --ff_size 1024 \
    --dropout 0.2 \
    --cond_drop_prob 0.1 \
    --seed 3407

echo "Stage 2 complete: $(date)"
echo ""
echo "=== Stage 3: ResidualTransformer ==="
echo "Started: $(date)"

python train_res_transformer.py \
    --dataset_name $DATASET \
    --name ResTransformer \
    --vq_name $VQ_NAME \
    --gpu_id $GPU_ID \
    --batch_size 64 \
    --max_epoch 500 \
    --latent_dim 384 \
    --n_layers 8 \
    --n_heads 6 \
    --ff_size 1024 \
    --dropout 0.2 \
    --cond_drop_prob 0.1 \
    --seed 3407

echo "Stage 3 complete: $(date)"
echo ""
echo "=== All training done ==="
echo "Checkpoints:"
echo "  checkpoints/t2m/MaskTransformer/model/"
echo "  checkpoints/t2m/ResTransformer/model/"
