# ADR-006: Depth Supervision 전략 — OccAny + LiDAR
**Date**: 2026-03-30
**Status**: Accepted
**Participants**: Claude + User
**Confidence**: 8/10

## Context

ADR-005 ablation 결과:
- **Depth가 motion conditioning의 핵심 원인** (lane-only: +4.8%, depth-only: +9.7%)
- VP monocular depth만으로도 zeroed gap 반전 달성 (+2.4%)
- Depth supervision을 강화하면 motion conditioning이 더 개선될 것으로 기대

사용 가능한 depth source:
- VP monocular depth: 빠르지만 절대 스케일 부정확
- OccAny 3D reconstruction: 정확하고 confidence 맵 포함, semantic도 제공
- LiDAR: 가장 정확한 ground truth, 하지만 sparse

## Decision

### VP depth를 OccAny + LiDAR로 교체

```
기존 (L2):  VP monocular depth → auxiliary loss
변경:       OccAny dense depth + LiDAR sparse GT → auxiliary loss
```

**VP depth는 불필요** — OccAny가 더 정확한 dense depth를 제공하고, LiDAR가 GT로 보정.

### Depth Supervision 구조

```
OccAny pts3d_local → dense depth (288x512, with confidence)
    ↓ resize to (64, 128)
    ↓ confidence-weighted L1 loss

LiDAR point cloud → camera projection → sparse depth (valid pixels only)
    ↓ sparse L1 loss (valid pixels에서만)

Combined loss = λ_occ · L1(pred, occany_depth, weight=conf)
              + λ_lid · L1(pred, lidar_depth, mask=valid)
```

**OccAny**: dense하지만 reconstruction error 있음 → confidence로 가중
**LiDAR**: sparse하지만 정확 → valid pixel에서 강한 signal

### 데이터 준비

| 소스 | 현재 상태 | 필요 작업 |
|------|----------|----------|
| OccAny depth | accident 데이터에서만 실행됨 | **RTB recording에 OccAny inference 실행 필요** |
| OccAny semantic | 위와 동일 | semantic_2ds도 함께 활용 가능 |
| LiDAR | rosbag-ingest에 이미 추출됨 (lidar_front, 45K pts) | camera projection 파이프라인 구현 필요 |
| VP depth | 20 recording에 있음 | **더 이상 사용하지 않음** |

### OccAny RTB Inference

```
RTB camera_front 프레임 (5-frame windows)
    → OccAny inference (occany_plus model, Depth-Anything-3)
    → pts3d_local (5, 288, 512, 3) → depth map
    → semantic_2ds (5, 288, 512) → semantic supervision (bonus)
    → conf (5, 288, 512) → confidence weight
```

### LiDAR → Camera Projection

```
lidar_front/*.npy (N, 4) XYZI
    → camera extrinsic/intrinsic 적용
    → 이미지 평면에 투영
    → sparse depth map (valid pixels만)
    → calibration은 rosbag-ingest metadata.json에 있음
```

### 예상 Architecture

```
Camera → ViT encoder → embedding
                          ├── predict next embedding (기존 MSE + SIGReg)
                          ├── decoder → depth prediction
                          │               ↕ OccAny dense depth (confidence weighted)
                          │               ↕ LiDAR sparse depth (valid pixels)
                          ├── decoder → semantic prediction (bonus)
                          │               ↕ OccAny semantic_2ds
                          └── decoder → lane prediction (선택)
                                          ↕ VP lane_masks (있으면)
```

## Cross-Project Integration

### OccAny 프로젝트에 요청 필요

```
요청: RTB recording에 OccAny inference 실행
입력: /mnt/phoenix-aap/ingest-output/{recording}/batch_*/camera_front/*.jpg
출력: pts3d_local, conf, semantic_2ds per 5-frame window
대상: VP가 있는 20개 recording 중 학습용 5개 우선
```

### le-wm에서 구현 필요

1. OccAny output → HDF5 변환 (convert 스크립트 확장)
2. LiDAR → camera projection 파이프라인
3. Confidence-weighted depth loss 구현
4. Sparse LiDAR loss 구현

## Branching (다음 단계)

| OccAny+LiDAR 결과 | 해석 | 다음 |
|-------------------|------|------|
| E0 gap > 20% | Depth supervision이 motion 핵심 lever 확정 | Application 우선 (auto-label, anomaly) |
| E0 gap 10-20% | 개선은 있으나 한계 | Cross-attention 아키텍처 변경 검토 |
| E0 gap < 10% | Depth 정확도는 bottleneck 아님 | 다른 방향 필요 (pixel masking 등) |

## Explicitly NOT Doing

- VP depth 계속 사용 (OccAny가 대체)
- OccAny 코드 수정 (inference만 실행 요청)
- Multi-camera 입력 (single front camera 유지)
- OccAny+le-wm feature fusion (ADR-004 L5, 아직 아님)

## Action Items

1. [ ] OccAny 프로젝트에 RTB inference 요청 (collaboration request)
2. [ ] LiDAR → camera projection 파이프라인 구현
3. [ ] convert_rtb_to_hdf5.py에 OccAny depth + LiDAR depth 포함
4. [ ] Confidence-weighted + sparse depth loss 구현 (train_vp.py 확장)
5. [ ] 학습 + E0 평가
