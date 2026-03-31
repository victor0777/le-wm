#!/bin/bash
# ADR-002 Experiment Matrix
# A1: already done (lewm_rtb_multi)
# A3: 3D action, num_preds=4 (running now)
# B1: 4D action, num_preds=1
# B3: 4D action, num_preds=4

set -e
export PYTHONPATH=/home/ktl/projects/le-wm
export CUDA_VISIBLE_DEVICES=0

HOLDOUT="Namyang-Gian_JT_2025-09-15_07-06-35_2111_6c319d"
COMMON="wandb.enabled=False trainer.max_epochs=10 trainer.devices=1 +holdout=$HOLDOUT"

echo "=== Experiment B1: 4D action, num_preds=1 ==="
python3 -u train_multi.py data=rtb4d $COMMON wm.num_preds=1 output_model_name=lewm_B1_4d_pred1 2>&1 | tee /tmp/lewm_B1.log

echo "=== Experiment B3: 4D action, num_preds=4 ==="
python3 -u train_multi.py data=rtb4d $COMMON wm.num_preds=4 output_model_name=lewm_B3_4d_pred4 2>&1 | tee /tmp/lewm_B3.log

echo "=== All experiments done ==="
