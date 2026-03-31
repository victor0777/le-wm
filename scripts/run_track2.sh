#!/bin/bash
# Track 2: Token-conditioned training — 3 variants in parallel on GPUs 0,1,2
# Each runs 15 epochs on a single GPU.

set -e
cd /home/ktl/projects/le-wm
export PYTHONPATH=/home/ktl/projects/le-wm

EPOCHS=15
COMMON="data=rtb_occany wandb.enabled=False trainer.devices=1 trainer.max_epochs=$EPOCHS"

echo "=== Track 2: Token-conditioned training ==="
echo "Starting 3 variants in parallel..."

# V1: Livlab 3 train (b5b236, c48e71, 736fcb) -> 8014dd holdout
# Exclude da8241 (Gian-Pankyo) to keep only Livlab routes
CUDA_VISIBLE_DEVICES=0 python train_vp_token.py $COMMON \
    +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
    +exclude='[Gian-Pankyo]' \
    subdir=track2_v1 \
    output_model_name=lewm_track2_v1 \
    > outputs/track2_v1.log 2>&1 &
PID_V1=$!
echo "V1 started (PID=$PID_V1, GPU=0): Livlab 3 -> 8014dd"

# V2: da8241 only train -> 8014dd holdout (cross-domain)
CUDA_VISIBLE_DEVICES=1 python train_vp_token.py $COMMON \
    +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
    +exclude='[Livlab-Rt-C-1,Livlab-Rt-C-3,Livlab-Rt-C-5]' \
    subdir=track2_v2 \
    output_model_name=lewm_track2_v2 \
    > outputs/track2_v2.log 2>&1 &
PID_V2=$!
echo "V2 started (PID=$PID_V2, GPU=1): da8241 -> 8014dd (cross-domain)"

# V3: Livlab 4 train -> da8241 holdout (cross-domain reverse)
CUDA_VISIBLE_DEVICES=2 python train_vp_token.py $COMMON \
    +holdout=Gian-Pankyo_JT_2025-08-19_04-59-28_2111_da8241 \
    subdir=track2_v3 \
    output_model_name=lewm_track2_v3 \
    > outputs/track2_v3.log 2>&1 &
PID_V3=$!
echo "V3 started (PID=$PID_V3, GPU=2): Livlab 4 -> da8241 (cross-domain reverse)"

echo ""
echo "Waiting for all variants to complete..."
echo "Logs: outputs/track2_v{1,2,3}.log"

# Wait and report
wait $PID_V1 && echo "V1 completed successfully" || echo "V1 FAILED (exit=$?)"
wait $PID_V2 && echo "V2 completed successfully" || echo "V2 FAILED (exit=$?)"
wait $PID_V3 && echo "V3 completed successfully" || echo "V3 FAILED (exit=$?)"

echo ""
echo "=== All variants done. Run eval_track2.sh for motion ablation. ==="
