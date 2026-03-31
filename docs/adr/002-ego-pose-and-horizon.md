# ADR-002: Ego Pose 추가 및 Prediction Horizon 확장
**Date**: 2026-03-29
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4)
**Debate Style**: constructive
**Confidence**: 7.5/10

## Context

E0 holdout 결과에서 motion conditioning이 약함 (correct vs shuffled +8%, zeroed가 correct보다 나음).
Visual temporal continuity가 dominant signal. GNSS/IMU에서 ego pose를 추출하여 motion 신호를 강화하고,
prediction horizon 증가로 motion 의존도를 높이는 실험 설계 필요.

## Decision

### 1. Action Space 확장: Δψ 추가
- action = [vx, vy, yaw_rate, **Δψ**] (3D → 4D)
- Δψ: 카메라 프레임 간 IMU quaternion에서 추출한 상대 yaw 변화량
- absolute heading (sin/cos)은 action에 넣지 않음 (rollout interface 오염 방지)

### 2. Prediction Horizon 확장
- num_preds = 1 → 4 테스트
- 더 먼 미래를 예측하면 visual continuity만으로 불충분 → motion 신호 의존도 증가

### 3. 4-Experiment Matrix

| Exp | Action | num_preds | 목적 |
|-----|--------|-----------|------|
| A1 | [vx,vy,yaw_rate] (3D) | 1 | 기존 baseline |
| A3 | [vx,vy,yaw_rate] (3D) | 4 | horizon 효과만 |
| B1 | [vx,vy,yaw_rate,Δψ] (4D) | 1 | ego pose 효과만 |
| B3 | [vx,vy,yaw_rate,Δψ] (4D) | 4 | combined |

### 4. Δψ 계산 방법
- IMU quaternion → yaw 추출 (per camera frame)
- Episode 내 unwrap
- Δψ_t = ψ_t - ψ_{t-1}
- 기존 frameskip loader가 5-step turn profile을 자동으로 묶어줌
- 검증: angular_yaw와 방향/부호 일치 확인 필수

### 5. Evaluation Metrics
- Holdout correct MSE (낮을수록 좋음)
- Holdout shuffled-correct MSE gap (클수록 motion 사용)
- Holdout zeroed-correct MSE gap (클수록 motion 사용)
- **Gap만 보지 말고 correct MSE도 안정적이어야 함**

### 6. State Encoder: 보류
- Δx, Δy, sin(ψ), cos(ψ)를 proprio에 추가하는 것은 현재 모델에서 predictor에 영향 없음
- 별도 state encoder 구현은 이번 라운드 결과 후 결정

## Debate Summary

### Round 1
- Claude: Option B (action에 ego pose 추가)로 predictor 강제 conditioning
- Codex: Modified C 제안 — action은 control-like 유지, Δψ만 추가. 절대 heading/Δx,Δy는 state에
- 합의: action에는 Δψ만, state encoder는 보류

### Round 2
- Claude: horizon vs feature — 병렬 테스트 제안 (2×2 factorial)
- Codex: 6→4 실험으로 trim, num_preds=2 skip. Δψ는 quaternion yaw difference가 최적
- 합의: 4-experiment matrix 확정

### Round 3
- 전체 합의, confidence 7.5/10
- 리스크: Δψ 부호/프레임 오류, horizon이 진짜 원인이 아닐 수 있음, rollout interface 호환성

## Action Items
1. [ ] convert_rtb_to_hdf5.py 수정: Δψ를 4번째 action channel로 추가
2. [ ] 기존 3D action HDF5와 별도로 4D action HDF5 생성 (또는 Δψ를 별도 key로)
3. [ ] train_multi.py에서 num_preds override 지원 확인
4. [ ] 4개 실험 실행 + E0 holdout 평가
5. [ ] 결과에 따라 state encoder 도입 여부 결정

## Explicitly NOT Doing
- Absolute heading을 action에 추가
- Δx, Δy를 action에 추가 (state/proprio에만)
- Curvature 추가 (yaw_rate + speed와 중복)
- State encoder 구현 (이번 라운드 결과 후)
- num_preds=2 실험 (ambiguous 시에만 추가)

## Consequences
- HDF5 변환 스크립트 수정 필요 (4D action 옵션)
- config에 action_dim override 필요
- 4개 실험 각 ~2.5시간 (single A100) → 총 ~10시간
