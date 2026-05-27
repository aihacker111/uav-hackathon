#!/usr/bin/env bash
# =============================================================================
# Train DEIMv2-JDE on VisDrone
#
# Phase 1 (warm-up): backbone+encoder FROZEN, only neck+heads trained
# Phase 2 (fine-tune): unfreeze all, lower LR for backbone/encoder
#
# Usage:
#   chmod +x scripts/train_deimv2_visdrone.sh
#   bash scripts/train_deimv2_visdrone.sh
#
# Prerequisites:
#   1. Download DEIMv2-S COCO checkpoint:
#        https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL
#      → save as: ./models/deimv2_dinov3_s_coco.pth
#
#   2. (Optional) Download ViT-Tiny distilled backbone separately:
#        https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs
#      → save as: ../../DEIMv2/ckpts/vitt_distill.pt
#      (The full DEIMv2 checkpoint already embeds these weights,
#       so this is only needed if you omit --deimv2_pretrained)
#
# =============================================================================

cd "$(dirname "$0")/.."   # run from AMOT root

DEIMV2_CKPT="./models/deimv2_dinov3_s_coco.pth"

# ------------------------------------------------------------------
# Common args
# ------------------------------------------------------------------
COMMON_ARGS="
  --task mot
  --dataset jde
  --arch deimv2
  --head_conv 256
  --down_ratio 4
  --data_cfg src/lib/cfg/visdrone.json
  --gpus 0
  --batch_size 8
  --num_epochs 30
  --lr_step 20,25
  --save_all
  --deimv2_pretrained ${DEIMV2_CKPT}
"

# ------------------------------------------------------------------
# Phase 1: Warm-up — only neck + CenterNet heads (20 epochs)
# backbone + encoder are FROZEN inside get_deimv2_jde (freeze_backbone=True)
# ------------------------------------------------------------------
echo "========== Phase 1: Warm-up (neck + heads only) =========="
python src/train.py \
  ${COMMON_ARGS} \
  --exp_id deimv2_visdrone_warmup \
  --lr 5e-4 \
  --num_epochs 20

# ------------------------------------------------------------------
# Phase 2: Fine-tune all layers (10 more epochs, lower LR for backbone)
# To unfreeze backbone, call unfreeze_all_params(model) in train.py
# or simply resume from Phase-1 checkpoint and train all layers.
# ------------------------------------------------------------------
echo "========== Phase 2: Fine-tune all layers =========="
python src/train.py \
  ${COMMON_ARGS} \
  --exp_id deimv2_visdrone_finetune \
  --lr 1e-4 \
  --num_epochs 30 \
  --load_model exp/mot/deimv2_visdrone_warmup/model_last_deimv2.pth \
  --resume
