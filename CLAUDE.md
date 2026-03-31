# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeWorldModel (LeWM) — a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) for learning world models from raw pixels. Uses only 2 loss terms (MSE prediction + SIGReg regularization) vs 6+ in prior methods. ~15M parameters, trainable on a single GPU.

## Git Repository
- **Origin**: https://github.com/victor0777/le-wm.git
- **Upstream**: https://github.com/lucas-maes/le-wm

## Setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Datasets: download from [HuggingFace](https://huggingface.co/collections/quentinll/lewm), decompress with `tar --zstd -xvf archive.tar.zst`, place `.h5` files in `$STABLEWM_HOME` (default: `~/.stable-wm/`).

## Commands

### Training
```bash
python train.py data=pusht                    # PushT dataset
python train.py data=tworoom                  # Two-room navigation
python train.py data=dmc                      # DeepMind Control Suite
python train.py data=ogb                      # OpenGov Bench
python train.py data=pusht seed=42 wm.history_size=4  # Override config
```

### Evaluation
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
python eval.py --config-name=cube.yaml policy=cube/lewm
python eval.py --config-name=reacher.yaml policy=reacher/lewm
python eval.py --config-name=pusht.yaml policy=random   # baseline
```

## Architecture

The codebase has 4 source files + Hydra configs:

- **`jepa.py`** — Core JEPA model. `encode()` maps pixels→embeddings via ViT, `predict()` forecasts next embeddings given current embedding + action, `rollout()` does autoregressive multi-step prediction for planning, `get_cost()` scores action candidates for MPC.

- **`module.py`** — Neural components. `SIGReg` (sketch isotropic Gaussian regularizer preventing collapse), `ARPredictor` (autoregressive transformer predictor with AdaLN-zero action conditioning), `Embedder` (action encoder via 1D conv + MLP), standard transformer blocks.

- **`train.py`** — Lightning training pipeline. Loads HDF5 datasets, builds ViT encoder + projector + predictor + action encoder, trains with AdamW + cosine annealing. Loss = MSE + λ·SIGReg (λ=0.09).

- **`eval.py`** — MPC-based evaluation. Creates environments via `stable_worldmodel`, plans with CEM or gradient-based solver over learned world model, runs 50 evaluation episodes.

- **`utils.py`** — ImageNet preprocessing, per-column normalization (StandardScaler), Lightning checkpoint callback.

### Config Structure
- `config/train/lewm.yaml` — master training config (optimizer, model dims, loss weights, WandB)
- `config/train/data/*.yaml` — per-dataset configs (dataset name, keys, frameskip)
- `config/eval/*.yaml` — per-environment eval configs (planning horizon, solver, budget)
- `config/eval/solver/` — CEM vs Adam solver configs

### Key Design Decisions
- Actions are accumulated over frameskip=5 steps into chunks before encoding
- Predictor uses AdaLN-zero modulation (action conditions transformer via scale/shift, not concatenation)
- SIGReg uses random projections + empirical characteristic function to enforce Gaussian embeddings
- Training uses bf16 precision with gradient clipping=1.0
- WandB entity/project must be configured in `lewm.yaml` before training
- 실제 캐시 디렉토리는 `~/.stable_worldmodel/` (README의 `~/.stable-wm/`과 다름)
- DDP multi-GPU 학습 시 HDF5 동시 읽기 데드락 → 단일 GPU 사용

## RTB 자율주행 데이터 학습

### 데이터 경로 (par02)
| 데이터 | 경로 | 설명 |
|--------|------|------|
| rosbag 원본 | `/mnt/phoenix-aap/ingest-output/` | 73개 recording (카메라/LiDAR/IMU/GNSS/velocity) |
| RTB HDF5 (3D action) | `~/.stable_worldmodel/rtb/` | 5개 recording, action=[vx,vy,yaw_rate] |
| RTB HDF5 (4D action) | `~/.stable_worldmodel/rtb4d/` | 5개 recording, action=[vx,vy,yaw_rate,Δψ] |
| 사고 영상 원본 | `/data2/accident_data/` | 30,073개 mp4 (1920x1080, ~20s, 60fps) |
| 사고 VP inference | `/data2/accident_vp_inference/` | 602개 (sample_frames + inference_results + features.json) |

### RTB 학습 명령
```bash
# 단일 recording
python train.py data=rtb data.dataset.name=rtb/RECORDING_ID wandb.enabled=False trainer.devices=1

# Multi-recording with holdout
python train_multi.py data=rtb +holdout=RECORDING_ID wandb.enabled=False trainer.devices=1

# 4D action (with Δψ)
python train_multi.py data=rtb4d +holdout=RECORDING_ID wandb.enabled=False trainer.devices=1
```

### HDF5 변환
```bash
# 3D action
python scripts/convert_rtb_to_hdf5.py --recordings RECORDING_ID

# 4D action (with Δψ from IMU quaternion)
python scripts/convert_rtb_to_hdf5.py --recordings RECORDING_ID --action-4d --output-dir ~/.stable_worldmodel/rtb4d
```

### 평가 스크립트
- `scripts/eval_e0_motion_ablation.py` — motion conditioning 검증 (correct/shuffled/zeroed)
- `scripts/visualize_embeddings.py` — t-SNE, cosine similarity, NN retrieval

### 주요 실험 결과 (ADR-001~006)
- Visual representation은 cross-route 일반화 성공 (speed/scene clustering)
- **Depth supervision이 motion conditioning의 핵심**: OccAny 3D depth로 shuffled gap +20% 달성
- Lane supervision은 motion에 기여하지 않음 (정적 구조)
- ADR-002 실험 (feature/horizon 변경) 모두 실패 → depth가 원인
- Primary application: embedding 기반 anomaly detection + auto-labeling

### Calibration
- 모든 recording의 metadata.json에 동일한 calibration 포함
- `systems[0].components[N].parameters.calibration.extrinsic`: 4x4 transform
- `systems[0].components[N].parameters.calibration.intrinsic`: fx, fy, cx, cy, distortion
- Component IDs: Camera_Front=100, Camera_Left=101, Camera_Rear=102, Camera_Right=103, LiDAR_Front=200

### Cross-Project 데이터
- **OccAny depth**: `/data2/occany-inference/{short_id}/batch_*/pts3d_local.npy` (dense 3D depth + confidence)
- **VP inference**: `/data2/vp-inference/{recording_id}/batch_*/inference_results/` (lane_masks, depth_maps)
- **DVIS panoptic**: perception exp12 (temporal consistent, 1.31% pixel change)

### ADR 문서 (docs/adr/)
- ADR-001: RTB world model 전략 (Codex debate)
- ADR-002: Ego pose + horizon 실험 (실패 → depth가 원인)
- ADR-003: Auto-labeling 서비스 컨셉
- ADR-004: Multi-sensor architecture vision (L1-L6)
- ADR-005: L2 validation & ablation (depth > lane)
- ADR-006: OccAny + LiDAR depth supervision 전략
