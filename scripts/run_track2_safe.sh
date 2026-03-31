#!/bin/bash
# Track 2: Token-conditioned training — safe mode (no persistent workers)
# Resumes from existing checkpoints.

set -e
cd /home/ktl/projects/le-wm
export PYTHONPATH=/home/ktl/projects/le-wm

EPOCHS=15
COMMON="data=rtb_occany wandb.enabled=False trainer.devices=1 trainer.max_epochs=$EPOCHS num_workers=2 loader.persistent_workers=False loader.prefetch_factor=2"

echo "=== Track 2: Token-conditioned training (safe, num_workers=0) ==="

echo ""
echo "--- V1: Livlab 3 -> 8014dd (resume from epoch 7) ---"
CUDA_VISIBLE_DEVICES=0 python train_vp_token.py $COMMON \
    +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
    +exclude='[Gian-Pankyo]' \
    subdir=track2_v1 \
    output_model_name=lewm_track2_v1
echo "V1 done"

echo ""
echo "--- V2: da8241 -> 8014dd (resume from epoch 7) ---"
CUDA_VISIBLE_DEVICES=0 python train_vp_token.py $COMMON \
    +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
    +exclude='[Livlab-Rt-C-1,Livlab-Rt-C-3,Livlab-Rt-C-5]' \
    subdir=track2_v2 \
    output_model_name=lewm_track2_v2
echo "V2 done"

echo ""
echo "--- V3: Livlab 4 -> da8241 (resume from epoch 4) ---"
CUDA_VISIBLE_DEVICES=0 python train_vp_token.py $COMMON \
    +holdout=Gian-Pankyo_JT_2025-08-19_04-59-28_2111_da8241 \
    subdir=track2_v3 \
    output_model_name=lewm_track2_v3
echo "V3 done"

echo ""
echo "=== All variants complete ==="
