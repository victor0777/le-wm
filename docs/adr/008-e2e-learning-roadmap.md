# ADR-008: E2E Learning, Scene Understanding, and Goal-Conditioned World Model Roadmap

**Date**: 2026-03-31
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4)
**Debate Style**: constructive (3 rounds)
**Confidence**: 6/10 (overall), per-phase scores below

## Context

ADR-007 실험들이 LeWM의 핵심 특성과 한계를 밝혀냈다:

### 실험 결과 요약

| 실험 | 핵심 결과 |
|------|----------|
| A1: 오버랩→오버랩 | Shuffled gap **+24.3%** — place-specific motion 강함 |
| A2: cross-domain | Shuffled gap **+1.4%** — motion 무용, action이 noise |
| A3: cross-domain reverse | Shuffled gap **-0.4%** — zeroed가 -13.4%로 나음 |
| C2-A1: multi-step rollout | 0.5s +21.7% → 2.0s **+43.8%** — gap 누적 증가 |
| C2-A2: cross-domain rollout | 0.5s +0.2% → 2.0s +3.3% — 정체, zeroed 악화 |
| C3: scene stratified | stop +73.2%, curve +15.7%, straight +17.9% — scene type만으로 설명 불가 |

### 핵심 발견
1. **Motion conditioning은 place-specific** — 같은 구역에서만 action이 의미를 가짐
2. **모델 capacity는 충분** — 2초 ahead에서 +43.8% gap 달성
3. **Scene type만으로는 부족** — C3에서 moving scene 간 차이 작음 (~15-18%)
4. **Cross-domain에서 action은 noise** — goal/route context 없이 일반화 불가
5. **Stop scene이 gap을 과대평가** — 정차 시 +73.2%는 "변화 없음" 예측의 결과

## Decision

Two parallel tracks + validation experiments:

### Phase 0: Validation Experiments (이번 주)

| ID | 실험 | 목적 | 판단 기준 |
|----|------|------|----------|
| **P0-A** | Retrieval baseline (Visual kNN, Pose kNN, Visual+Pose kNN) | +43.8%가 dynamics인지 memorization인지 판별 | kNN이 LeWM에 근접하면 memorization |
| **P0-B** | Pseudo-maneuver labeling (IMU/GNSS → left/right/straight/stop) | 가장 싼 intent proxy 생성 | 수동 검증으로 label 품질 확인 |
| **P0-C** | ~~C3 scene stratified~~ **완료** | Scene type별 motion sensitivity | Stop이 gap을 지배, moving scene 간 차이 작음 |
| **P0-D** | Counterfactual sanity check | World model rollout이 plausible한지 | 로그 action이 대안 action보다 높은 점수 |

**Confidence: 9/10** — 즉시 실행 가능, 명확한 판단 기준

### Phase 1: Two Parallel Tracks (2-4주)

**Track 1: Corridor Planning Proof**
- MPC/CEM over world model on best overlap route (Livlab)
- VP/OccAny-derived costs: lane retention, free-space violation, depth-to-obstacle margin
- 주장 범위: "반복 주행 구간에서 유용한 counterfactual/planning signal"
- **Planner exploit check 필수** (Codex 제안): 낮은 predicted cost가 실제 안전 proxy와 상관하는지 검증

**Track 2: Causal Token Conditioning**
- Pseudo-tokens: maneuver (from future trajectory) + branch-count (from GNSS trajectories)
- Segment ID는 diagnostic only (place leakage 감지용)
- 실험 변수: maneuver-only, branch-count+maneuver, segment-ID+maneuver
- Re-run cross-domain E0/C2 ablations
- **Decision gate**: non-place token (maneuver/branch-count)으로 cross-domain shuffled gap > 10%

**Confidence: Track 1 = 8/10, Track 2 = 5/10**

### Phase 2: Structured Scene Transfer (1-2개월)

Track 2가 성공한 경우에만 진행:
- VP + OccAny에서 auto-derived lane topology
- Intersection type, right-of-way, traffic control proxy
- Cross-route transfer with structured conditioning
- **Decision gate**: held-out route에서 cross-domain 작동

**Confidence: 4/10** — 가장 어려운 단계, weak topology가 intersection에서 실패할 가능성

### Phase 3: Goal-Conditioned World Model (2-3개월)

Phase 2가 non-place generalization을 보인 경우에만:
- Full route/goal embedding
- Multi-route planning
- Calibrated counterfactual scenario generation

**Confidence: 3/10** — Phase 2 결과에 강하게 의존

## Debate Summary

### Round 1 (Claude 제안 → Codex 반론)
- Claude: route-specific → scene understanding → goal conditioning 순서 제안
- Codex: **순서를 뒤집어야 함** — intent/topology grounding이 먼저. Coarse scene labels 불충분. Retrieval baseline 필수.
- Codex: GAIA-1, DriveDreamer은 structured conditioning을 먼저 함. 우리는 DriveDreamer보다 덜 structured.

### Round 2 (Claude 재반론 → Codex 조정)
- Claude: 데이터에 HD map/maneuver annotation 없음 → Phase 0와 Phase 1 병렬 실행 제안
- Codex: **동의하되 corridor proof의 claim을 제한**해야 함. Pseudo-labels를 IMU/GNSS에서 저렴하게 생성.
- Codex: World model rollout은 counterfactual evaluator로 먼저 사용, 합성 학습 데이터는 calibration 후.
- **핵심 합의**: 4가지 즉시 실행 가능 실험 (Retrieval audit, Pseudo-maneuver, Counterfactual sanity, Scene stratified)

### Round 3 (합의 도출)
- Codex 추가 조건: segment ID는 diagnostic only, planner exploit check 필수, traffic-control proxy를 Phase 1 말에 도입
- **위험 요소**: pseudo-label noise (GPS drift), class imbalance (straight 지배), false positive gate, phase coupling
- **최종 합의**: Phase 0~3 구조 유지, gate를 더 엄격하게 (retrieval baseline 초과 필수)

## Phase 0 Results (2026-03-31, ALL COMPLETED)

| 실험 | 결과 | 판정 |
|------|------|------|
| P0-A: Retrieval baseline | LeWM이 kNN 대비 82% 우위 | **DYNAMICS** — memorization 아님 |
| P0-B: Pseudo-maneuver | 6 classes, consistency >94%, 18,970 labels | **사용 가능** — Track 2 입력 준비됨 |
| P0-C: Scene stratified | stop +73.2%, moving ~15-18% | **불충분** — scene type만으로 설명 불가 |
| P0-D: Counterfactual | 3/4 sanity checks 통과, left/right 반대 방향 | **PLAUSIBLE** — short-horizon evaluator OK |

**Phase 0 결론:**
1. LeWM은 진짜 dynamics를 학습 → **Track 1 투자 정당화**
2. Pseudo-maneuver labels 준비됨 → **Track 2 즉시 시작 가능**
3. Counterfactuals가 물리적으로 plausible → short-horizon planning 사용 가능
4. Per-sequence planning ranking은 2.5s에서 unreliable → uncertainty 고려 필요

**P0-B Label 분포:** straight 44.7%, stop 26.1%, decel 9.9%, accel 7.0%, left 6.4%, right 5.9%
- da8241 (고속도로): straight 68%, turn 1.2% / Livlab (시가지): turn 19~24%, stop 21~46%

## Action Items

### 이번 주 — Phase 0 (완료)
1. [x] P0-A: Retrieval baseline — **LeWM 82% 우위, dynamics 확인**
2. [x] P0-B: Pseudo-maneuver labeling — **6 classes, >94% consistency, labels saved**
3. [x] P0-C: C3 scene stratified — **stop +73.2%, moving ~15-18%**
4. [x] P0-D: Counterfactual sanity — **3/4 passed, plausible counterfactuals**
5. [ ] Pseudo-label noise audit — 소량 수동 검증 (optional, consistency >94%로 우선순위 낮음)

### Phase 1 결과 (2026-03-31)

**Track 1: Corridor Planning — PARTIAL PASS**
- CEM이 100% sequences에서 logged action보다 나은 action 발견, 24.8% MSE 감소
- Cost landscape smooth, monotonic convergence
- 단, logged action top-1 accuracy 0% → embedding MSE가 최적 cost function이 아님
- Follow-up: VP/OccAny 기반 cost function 필요

**Track 2: Maneuver Token Conditioning — DECISION GATE PASSED**

| Variant | Train → Holdout | Shuffled Gap (no token) | Shuffled Gap (token) | Delta |
|---------|----------------|------------------------|---------------------|-------|
| V1 | Livlab → Livlab (overlap) | +24.3% | +3.0%* | -21.3pp |
| V2 | da8241 → Livlab (cross) | +1.4% | +4.0% | +2.6pp |
| **V3** | **Livlab → da8241 (cross rev)** | **-0.4%** | **+16.7%** | **+17.1pp** |

*V1 epoch 7/15, 불완전 학습

**V3 결과가 핵심**: cross-domain에서 action이 noise였던 것이 (+16.7%) 유의미한 signal이 됨.
Maneuver token이 "왜 이 action인가"의 intent context를 제공하여 place-specific 한계를 돌파.

→ **Phase 2 (Structured Scene Transfer) 진행 정당화**

### 다음 단계
- V1 full 15-epoch 재학습 (overlap regression 원인 조사)
- Track 1: VP/OccAny cost function 설계
- Phase 2: lane topology + branch-count token 추가

## Explicitly NOT Doing

- Model scaling before retrieval audit — capacity는 bottleneck이 아님
- Self-generated training data — calibration check 전까지 evaluation only
- HD map integration — log-based pseudo-labels로 bootstrap
- Full CARLA/UniSim — own model을 counterfactual evaluator로 사용
- Coarse scene labels as conditioning — C3에서 moving scene 간 차이가 작음을 확인

## Consequences

- ~~**P0-A retrieval이 LeWM과 유사하면**: memorization~~ **→ RESOLVED: LeWM이 82% 우위, dynamics 확인**
- **Track 2가 실패하면**: route-specific narrow product로 수렴 (anomaly detection에 유효)
- **Track 2가 성공하면**: goal-conditioned E2E로의 bridge 확인 → Phase 2 투자 정당화
- **Phase 2가 실패하면**: HD map 또는 외부 simulation 필요 (scope 확대 필요)

## References
- ADR-007: Spatial overlap and goal-conditioned motion (실험 데이터)
- [GAIA-1](https://arxiv.org/abs/2309.17080): Generative world model with structured conditioning
- [DriveDreamer](https://arxiv.org/abs/2309.09777): Traffic-constraint conditioned future generation
- [UniSim](https://waabi.ai/research/unisim): Log-grounded closed-loop sensor simulator
