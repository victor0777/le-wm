# ADR-001: RTB 자율주행 데이터 기반 World Model 전략
**Date**: 2026-03-28
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4)
**Debate Style**: constructive
**Confidence**: 8/10

## Context

LeWM (JEPA world model, 15M params)을 RTB 자율주행 데이터에 적용.
CAN 데이터 없이 velocity/IMU 센서로 action space를 구성하여 1개 recording (69K frames)에서 5 epoch 학습 완료.
Embedding 분석 결과 속도/시간/회전 구분 및 장면 검색 능력 확인.
향후 방향 결정 필요.

## Decision

### 1. 시스템 명칭
- "ego-motion-conditioned latent predictor" (action-conditioned world model이 아님)
- 이유: [vx, vy, angular_yaw]는 ego-motion proxy이지 counterfactual intervention이 아님

### 2. Action Space
- **기본**: [vx, vy, angular_yaw] raw twist block (10Hz, frameskip=5로 0.5s window)
- Embedder의 Conv1d가 frameskip window 내 temporal pattern을 학습 → 적분된 Δx/Δy/Δψ보다 우월
- **Ablation**: vy 제거 테스트 (도로 주행에서 lateral velocity는 low-SNR일 수 있음)

### 3. Architecture
- **현재 유지**: ViT-tiny, history_size=3, num_preds=1
- 데이터/평가가 더 큰 bottleneck. 모델 스케일링은 scaling curve 이후 결정
- num_preds 증가는 training에서 skip prediction이므로 안전 (autoregressive rollout은 inference 시만)

### 4. Scaling 전략
- **점진적**: 1 → 4 → 16 → 32 → 73 recordings
- **다양성 기반 샘플링**: route/speed/curvature/time-of-day 기준
- 장거리 고속도로 cruise가 과대 대표되지 않도록 주의
- Whole-recording holdout으로 train/test 분리 (현재 random frame split은 temporal leakage)

### 5. Multi-camera
- **보류**: front camera 벤치마크 확립 후 진행
- 추후 shared encoder + camera token + late fusion 방식

### 6. Primary Application
- **이상 탐지 (anomaly/surprise detection)**: |predicted_emb - actual_emb|² as surprise score
- traffic_rule_checker, accident_analysis 프로젝트에 직접 연결 가능
- Place recognition은 평가 도구로 활용 (primary application 아님)

## Evaluation Plan

### Primary Validity (필수)

| ID | 평가 | 설명 | 우선순위 |
|----|------|------|----------|
| E0 | Motion-conditioning ablation | 동일 visual history에서 correct/shuffled/zeroed motion block 비교. Action proxy 유효성 검증 | **최우선** |
| E1 | Whole-recording holdout prediction | Latent MSE + cosine similarity + retrieval rank. Speed/yaw-rate bin별 층화 분석 | 높음 |
| E5 | Cross-recording GNSS place recognition | 다른 recording의 동일 위치가 embedding 공간에서 가까운지 (GNSS ground truth) | 높음 |
| E3 | Surprise score correlation | Prediction error와 실제 이벤트 상관관계. Speed/turn/lighting nuisance 보정 필수 | 높음 |

### Secondary Diagnostics

| ID | 평가 | 설명 |
|----|------|------|
| E2 | Speed-decorrelated retrieval | Same-speed-bin 제외한 nearest neighbor 검색 |
| E4 | Linear scene probe | Frozen embedding 위에 highway/urban/intersection 분류기 |

### Critical Fix (E1 전 필수)
- **Normalizer data leak 수정**: 현재 `train.py:54-63`에서 전체 데이터셋으로 normalizer fit → train recording만으로 fit하도록 수정

## Debate Summary

### Round 1 (Codex 분석)
- Speed clustering이 dominant axis일 수 있으며, kinematics만 학습한 것인지 검증 필요
- [vx, vy, yaw_rate]는 action이 아니라 ego-motion이므로 counterfactual planning 주장 불가
- Scaling curve → 평가 체계 → 모델 확장 순서 제안

### Round 2 (Claude 반론 → 합의)
- Raw twist block이 Conv1d Embedder와 더 적합 (Codex 동의)
- Anomaly detection이 place recognition보다 actionable (Codex 동의)
- E0 (motion ablation)이 전체 전제를 검증하는 최우선 실험 (합의)
- num_preds 증가는 training에서 skip prediction이므로 예상보다 안전 (Codex 교정)

### Round 3 (합의)
- 모든 결정 확정, confidence 8/10
- 주요 리스크: E0에서 motion 사용이 약하면 reframe 필요, route-level holdout 필요, surprise score의 nuisance sensitivity

## Risks

1. **E0 실패 시**: action proxy가 무의미하면 "temporal prediction" 모델로 reframe
2. **Route 중복**: train/test가 같은 route를 공유하면 holdout이 너무 쉬움 → route-level split 필요
3. **Nuisance surprise**: lighting/exposure/sync drift가 anomaly score를 오염 → calibration 필수
4. **센서 정렬**: recording 증가 시 camera-velocity timestamp 오차가 conditional prediction 품질 저하 유발 가능

## Action Items

1. [ ] E0: motion-conditioning ablation 구현 및 실행
2. [ ] Normalizer data leak 수정 (train-only fit)
3. [ ] Whole-recording train/test split 구현
4. [ ] 4개 recording 추가 변환 (diversity sampling)
5. [ ] E1 + E5 평가 스크립트 구현
6. [ ] Scaling curve (1/4/16 recordings) 실행

## Explicitly NOT Doing

- ViT-small/base로 모델 확장 (scaling curve 전)
- Multi-camera 통합 (front camera 벤치마크 전)
- Δx/Δy/Δψ 적분 action (ablation으로만)
- Counterfactual planning claim
- 전체 73 recording 일괄 변환 (scaling curve 전)

## Consequences

- 평가 체계 우선 → 데이터 확장 → 모델 확장 순서가 확정됨
- anomaly detection이 primary downstream으로 확정, traffic_rule_checker/accident_analysis와 integration path 설정
- 1-2주 내 E0 결과에 따라 전략 방향이 결정됨 (action-conditioned vs temporal-only)
