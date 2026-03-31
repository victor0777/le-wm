# ADR-004: Multi-Sensor Architecture Vision + Cross-Project Integration
**Date**: 2026-03-30
**Status**: Proposed
**Participants**: Claude + User
**Confidence**: 6/10 (vision document, 구현 전 설계)

## Context

le-wm Phase 1-4 결과:
- Visual representation은 우수 (cross-route 일반화, auto-labeling 가능)
- Motion conditioning은 약함 (AdaLN-zero가 action 무시)
- 사고 surprise 1.35x 신호 있으나 pre-crash pattern 약함

사용 가능한 센서: Camera(4대), LiDAR(3대), IMU, GNSS, Velocity
기존 inference 결과: VP(lane+depth+detection), OccAny(3D voxel occupancy)
관련 프로젝트: OccAny, vehicle-tracker, perception, map, accident_analysis

## Decision: 단계적 Multi-Sensor + Cross-Project Integration

### Level 1: 현재 (완료)
```
Camera → ViT → 192-dim embedding → next embedding prediction
```
- 장면 분류, anomaly detection, auto-labeling
- Motion conditioning 약함

### Level 2: VP Auxiliary Supervision (다음 단계, 1-2주)
```
Camera → ViT → embedding → decoder → predict lane_mask + depth_map
                                         ↕ auxiliary loss
                                    VP inference 결과 (GT)
```

**사용 데이터**:
- `lane_masks.npy`: (N, 3, 80, 160) — ego-left, ego-right, 3rd lane
- `depth_maps.npy`: (N, 320, 640) — monocular depth (meter)

**구현**:
- 기존 ViT encoder에 lightweight decoder head 추가 (2-layer CNN)
- Loss = MSE_pred + SIGReg + λ_lane·BCE(lane) + λ_depth·L1(depth)
- Inference 시 VP 불필요 (encoder만 사용)

**기대 효과**:
- Encoder가 차선 구조 + 깊이를 내재적으로 이해
- Auto-labeling 정확도 향상 (road_type 구분에 차선 정보 반영)
- Anomaly detection 개선 (차선 이탈 감지 가능)

### Level 3: LiDAR Cross-Modal Supervision (2-4주)
```
Camera → ViT → embedding → decoder → predict depth + occupancy
                                        ↕ supervision
                               LiDAR projected depth (real 3D GT)
                               VP depth (monocular, 보조)
```

**사용 데이터**:
- `lidar_front/`: (N, 45K, 4) — XYZI point cloud
- rosbag-ingest에서 이미 추출됨, frame_index로 카메라와 동기화

**구현**:
- LiDAR point cloud → 카메라 projection → sparse depth GT
- VP depth는 dense하지만 부정확, LiDAR는 sparse하지만 정확 → 상호 보완
- Loss: L1(depth, lidar_projected) on valid pixels

**기대 효과**:
- Encoder가 **진짜 3D 거리**를 학습 (VP의 pseudo-depth 보정)
- 앞 차량까지의 거리 인식 → following distance 관련 anomaly 감지

### Level 4: IMU/GNSS Conditioning + Action Architecture 변경 (1-2개월)
```
Camera + (LiDAR supervision) → encoder → embedding
                                              ↓
IMU/GNSS → trajectory encoder ──→ cross-attention → conditioned embedding
                                              ↓
                                     predict future embedding
```

**핵심 변경**: AdaLN-zero 대신 **cross-attention**으로 ego motion 주입
- AdaLN-zero는 zero initialization으로 action 무시 학습 → 근본 원인
- Cross-attention은 query-key 매칭으로 관련 motion 정보를 능동적으로 참조
- 또는 **pixel masking**: 일부 프레임을 가리고 motion으로 복원 강제

**사용 데이터**:
- IMU quaternion → ego pose trajectory (Δx, Δy, heading)
- GNSS → absolute position → location-aware prediction
- Velocity → instantaneous ego motion

### Level 5: OccAny Integration — Spatial-Temporal Fusion (연구 단계)
```
                    ┌── le-wm: temporal prediction (T→T+1,T+2...)
                    │   → "미래에 무슨 일이 일어날까"
                    │
Camera sequence ────┤
                    │
                    └── OccAny: 3D spatial snapshot (at T)
                        → "지금 3D 공간이 어떻게 생겼는가"
                              ↓
                     Feature Fusion
                              ↓
                    Temporal-Spatial Feature Vector
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        Anomaly Score   Scenario Class   Accident Risk
```

**OccAny와의 관계** (collaboration insight 기반):

| | OccAny | le-wm | 결합 |
|--|--------|-------|------|
| **차원** | 3D spatial (voxel) | temporal (embedding sequence) | 4D (space + time) |
| **강점** | 기하학적 정확성, 충돌 영역 | 시간적 예측, 경량 | 공간+시간 통합 이해 |
| **약점** | 시간축 없음 (F1≈0.65 ceiling) | 3D 구조 약함 | — |
| **데이터** | Multi-view RGB (5 frames) | Single-view sequence | 상호보완 |

**OccAny insight에서 확인된 fusion 전략**:

```
Priority 1 (즉시): OccAny + VP → lane-aware occupancy (F1 0.65→0.75+ 기대)
Priority 2 (중기): OccAny + DriveStudio → novel view augmentation
Priority 3 (장기): OccAny + le-wm → temporal-spatial feature fusion
```

le-wm이 Level 2-3을 완료하면 **P3 실행 조건**이 충족됨:
- le-wm temporal anomaly score → OccAny spatial features와 결합
- 사고 분석: "공간적으로 위험한 상황" + "시간적으로 예상 밖인 상황" = 강력한 risk signal

### Level 6: 4D Occupancy World Model (Ultimate Vision)
```
Camera (4대) → multi-view encoder ─┐
LiDAR (3대) → voxel encoder ──────┤
IMU/GNSS ──→ ego pose encoder ────┤── BEV Fusion → 4D occupancy prediction
VP results → auxiliary supervision ┤
OccAny ────→ 3D spatial prior ────┘
                     ↓
           Future 3D Occupancy Grid (T+1, T+2, ...)
                     ↓
           ┌─────────┼─────────────┐
           ↓         ↓             ↓
     ISO 34502    Accident      Planning
     Scenario     Prediction    (if simulator)
```

이 단계에서는:
- BEV alignment 문제가 자연스럽게 해결됨 (모든 센서가 같은 3D 공간으로 투영)
- cut-in, lane change 등 행동 시나리오를 점유 변화로 감지
- OccAny의 현재 결과를 spatial prior로 활용

## Implementation Priority

| 단계 | 소요 | 필요 데이터 | 의존성 | ROI |
|------|------|-----------|--------|-----|
| **L2: VP aux loss** | 1-2주 | VP lane+depth (있음) | 없음 | **★★★★★** |
| **L3: LiDAR supervision** | 2-4주 | LiDAR (있음) | L2 | ★★★★ |
| **L4: Cross-attention + IMU** | 1-2월 | IMU/GNSS (있음) | L2 | ★★★ |
| **L5: OccAny fusion** | 2-3월 | OccAny output | L2+OccAny P1 | ★★★ |
| **L6: 4D occupancy** | 6월+ | 전체 | L3+L4+L5 | ★★★★★ |

**L2가 압도적으로 ROI가 높음**: 데이터 이미 있고, 구현 간단하고, 효과 큼.

## Cross-Project Integration Map

```
rosbag-ingest ─── Camera/LiDAR/IMU/GNSS ───→ le-wm (학습 데이터)
                                                 │
autoware_vision_pilot ─── VP results ──────→ le-wm L2 (aux supervision)
                                                 │
OccAny ─── 3D occupancy ──────────────────→ le-wm L5 (spatial fusion)
                                                 │
                                                 ↓
                                          le-wm outputs:
                                           ├── scene embedding → rtb-vlm (auto-label)
                                           ├── surprise score → accident_analysis
                                           ├── temporal feature → OccAny P3 fusion
                                           └── lane/depth aware → traffic_rule_checker

vehicle-tracker ─── BEV tracks ──→ map (BEV alignment) ──→ le-wm L6
perception ─── Mask2Former ──→ SAM2 대안 (temporal consistent)
map ─── lane graph ──→ le-wm L6 (BEV ground truth)
```

## Risks

1. **L2 실패 시**: VP depth/lane이 부정확하여 encoder에 noise 주입 → ablation 필수
2. **L5 interface**: le-wm latent ↔ OccAny explicit 변환이 기술적 난제
3. **L6 compute**: 4D occupancy prediction은 현재 단일 A100으로 부족할 수 있음
4. **Mask2Former temporal 불안정**: perception의 현재 이슈, SAM2 video mode가 대안

## Explicitly NOT Doing (이번 라운드)

- Level 4-6 구현 (L2-3 결과 후 결정)
- Multi-camera 입력 (single front camera로 L2-3 진행)
- BEV alignment 직접 해결 (map 프로젝트 역할)
- OccAny 코드 수정 (OccAny는 별도 프로젝트에서 진행)

## Action Items

1. [ ] **L2 구현**: VP lane/depth auxiliary loss 추가
   - decoder head 구현
   - VP 데이터를 HDF5에 포함하는 변환 스크립트 수정
   - 학습 + E0 재평가
2. [ ] **OccAny P3 interface 설계**: le-wm temporal feature 추출 API
3. [ ] **L3 데이터 준비**: LiDAR → camera projection 파이프라인
