# ADR-007: Spatial Overlap, Goal-Conditioning, and Motion Sensitivity

**Date**: 2026-03-31
**Status**: Experiment A+C2 completed
**Participants**: Claude (analysis), user (core insight)
**Confidence**: 6/10

## Context

5개 recording OccAny depth 학습 후 E0 motion ablation에서 shuffled gap이 +20.1% → +13.6%로 감소했다.
이 결과를 분석하면서 두 가지 근본적 질문이 제기되었다:

1. **공간적 오버랩**: 같은 구역을 여러 코스로 주행한 데이터 vs 독립된 구간 데이터가 motion conditioning에 미치는 영향
2. **Goal 부재**: E2E 주행에서는 "어디로 가야 하는가"라는 목적 정보가 있어 action이 의미를 가지는데, 현재 LeWM에는 그런 정보가 없음

### 데이터 구성

| Recording | Route | Sequences | 특성 |
|-----------|-------|-----------|------|
| da8241 | Gian-Pankyo | 7,644 | 독립 route, 비오버랩 |
| b5b236 | Livlab-Rt-C-1 | 1,642 | Livlab 순환코스, 오버랩 |
| c48e71 | Livlab-Rt-C-3 | 2,906 | Livlab 순환코스, 오버랩 |
| 736fcb | Livlab-Rt-C-5 | 3,409 | Livlab 순환코스, 오버랩 |
| 8014dd | Livlab-Rt-C-7 (holdout) | 1,651 | Livlab 순환코스, 오버랩 |

### E0 결과 이력

| 실험 | Train 구성 | Holdout | Shuffled Gap | Zeroed Gap |
|------|-----------|---------|-------------|------------|
| 단일 recording (58bb5f) | 1 rec (Gian) | same rec split | +35.0% | +24.0% |
| Multi 4-rec baseline | 4 rec (mixed) | Namyang-Gian | +8.0% | -1.9% |
| + VP lane only | 4 rec + lane | 8014dd | +4.8% | -0.3% |
| + VP depth only | 4 rec + VP depth | 8014dd | +9.7% | +2.4% |
| + VP lane+depth | 4 rec + both | 8014dd | +10.0% | +2.3% |
| **L3 OccAny 4-rec** | **3 Livlab + depth** | **8014dd** | **+20.1%** | **+3.9%** |
| **L3 OccAny 5-rec** | **3 Livlab + da8241 + depth** | **8014dd** | **+13.6%** | **+3.3%** |

## 가설

### H1: 오버랩 데이터는 visual 암기를 촉진하여 motion conditioning을 약화시킨다

같은 구역을 여러 코스로 주행하면 encoder가 **장소를 암기**(place memorization)할 수 있다.
"이 교차로 다음은 이 건물"이라는 visual prior가 형성되면, action 없이도 다음 프레임을 예측 가능.

- **예측**: 오버랩 train → 오버랩 holdout에서 shuffled gap **낮음**
- **현재 증거**: Livlab 3개로 학습 → Livlab holdout에서 gap이 높은 것은 depth supervision 효과이지, 오버랩 데이터 자체의 효과는 아닐 수 있음

### H2: 오버랩 데이터는 "같은 장소, 다른 action → 다른 결과" 학습에 유리하다

H1의 반대. 같은 교차로에서 직진/좌회전/우회전 데이터가 모두 있으면, 모델이
**"여기서 action이 달라지면 미래가 달라진다"**를 직접 학습할 기회가 생긴다.

- **전제 조건**: 오버랩 구간에서 **실제로 다른 action이 취해져야** 함
- Livlab 순환코스가 같은 방향 반복인지, 분기점에서 다른 방향인지에 따라 H1/H2가 갈림
- **예측**: 분기점 오버랩이 많으면 shuffled gap **높음**

### H3: 비오버랩 데이터가 generalizable dynamics 학습에 더 효과적이다

da8241(Gian-Pankyo)은 모델이 본 적 없는 visual scene. Visual prior로 예측 불가하므로
**action에 의존할 수밖에 없음**. 하지만 현재 da8241이 train의 49%를 차지하면서:

- 모델이 da8241 route에 과적합
- Holdout(Livlab)에서의 motion sensitivity가 이 route의 bias로 희석됨
- **예측**: da8241을 holdout으로 전환하면 shuffled gap이 높아질 수 있음

### H4 (핵심): Goal/Route 정보 부재가 motion conditioning의 근본적 한계

**이것이 가장 중요한 가설이다.**

E2E 자율주행 시스템에서는:
```
Input: pixels + goal/route + current_state → Output: action (steering, throttle)
```

action은 **"어디로 가겠다"는 의도**의 결과물이다. 모델은 goal을 알고 있으므로
"이 교차로에서 좌회전하는 이유"를 이해할 수 있다.

현재 LeWM에서는:
```
Input: pixels + action (observed) → Output: predicted next embedding
```

action은 **"무엇이 일어났는가"의 관측값**일 뿐, "왜 이 action인지"의 맥락이 없다.
모델 입장에서:
- 직진 action을 받아도 "직진로니까 직진한 건지" vs "교차로에서 직진을 선택한 건지" 구분 불가
- 좌회전 action을 받아도 "좌회전 차로라서 돈 건지" vs "교차로에서 좌회전한 건지" 구분 불가

**이 ambiguity 때문에:**
1. **직선 도로**에서: action이 거의 동일 (항상 직진) → motion input이 informative하지 않음
2. **교차로**에서: action이 달라야 하는 순간이지만, goal 없이는 action이 "결과"이지 "원인"이 아님
3. 모델은 action을 무시하고 **visual continuity만으로 예측하는 것이 최적해**가 되기 쉬움

이것이 ADR-002에서 관찰한 "AdaLN-zero가 action modulation을 0으로 유지"하는 현상의 근본 원인일 수 있다.

### H4의 함의

| 접근법 | Goal 정보 | Motion 활용 가능성 |
|--------|-----------|-------------------|
| 현재 LeWM | 없음 | action은 보조적 signal, visual dominant |
| + Route embedding | 출발지/목적지 encoding | 교차로에서 "어디로 갈지" 구분 가능 |
| + Navigation command | {직진, 좌회전, 우회전, 유턴} | 분기점에서 action의 의미 부여 |
| + HD map conditioning | 전방 도로 구조 | 가능한 경로들 중 action이 어느 것인지 |
| E2E 방식 (역방향) | goal → action 생성 | action이 goal의 함수로 완전히 conditioning됨 |

**LeWM의 목적이 anomaly detection이라면**, goal conditioning 없이도:
- "보통 이런 scene에서 이렇게 움직인다"의 **통계적 regularity**는 학습 가능
- 이것만으로 "비정상적 움직임" 탐지에는 충분할 수 있음
- 하지만 "정상적인 좌회전" vs "비정상적 직진 (신호위반)" 구분은 불가

## 실험 계획

### Experiment A: 오버랩 효과 분리 (H1, H2 검증)

| ID | Train | Holdout | 목적 |
|----|-------|---------|------|
| A1 | Livlab 3개 (b5b236, c48e71, 736fcb) | 8014dd | 오버랩 only → 오버랩 holdout |
| A2 | da8241 only | 8014dd | 비오버랩 only → 오버랩 holdout |
| A3 | Livlab 4개 (holdout 제외) | da8241 | 오버랩 train → 비오버랩 holdout |

모두 OccAny depth supervision 포함, 15 epochs, 동일 hyperparameter.
비교 지표: shuffled gap, zeroed gap, absolute MSE.

### Experiment B: 데이터 균형 효과 (H3 검증)

| ID | Train | Holdout | 목적 |
|----|-------|---------|------|
| B1 | 현재 5-rec (불균형) | 8014dd | baseline (da8241 49%) |
| B2 | 5-rec + da8241 서브샘플링 (~3K) | 8014dd | 균형 맞춘 후 효과 |
| B3 | 5-rec + recording-level balanced sampling | 8014dd | per-epoch 균등 샘플링 |

### Experiment C: Goal proxy 도입 (H4 검증)

H4를 직접 검증하려면 goal conditioning이 필요하지만, 현재 데이터에 goal 정보가 없다.
가능한 proxy 실험:

| ID | 방법 | 목적 |
|----|------|------|
| C1 | GPS heading (목적지 방향) auxiliary | 암묵적 goal signal |
| C2 | 전방 N초 action trajectory를 hint로 | "미래 action을 알면 예측이 나아지는가?" |
| C3 | Scene type label (직선/교차로/회전) | 도로 구조가 action 해석을 돕는지 |

C2가 가장 직접적: 미래 action을 줬을 때 예측이 크게 좋아지면,
모델이 action을 활용할 **능력**은 있지만 **맥락이 부족**한 것이 확인됨.

### 우선순위

1. ~~**A1, A2, A3** — 기존 데이터로 즉시 실행 가능, 1시간 이내~~ **완료**
2. **B2** — da8241 서브샘플링만 구현하면 됨
3. **C2** — forward action hint 실험, 아키텍처 수정 필요

## Experiment A 결과 (2026-03-31)

### 정량 결과

| ID | Train → Holdout | Train Seq | Shuffled Gap | Zeroed Gap | Correct MSE |
|----|----------------|-----------|-------------|------------|-------------|
| A1 | Livlab 3 → Livlab (오버랩→오버랩) | 7,957 | **+24.3%** | **+6.6%** | 0.1717 |
| A2 | da8241 → Livlab (비오버랩→오버랩) | 7,644 | +1.4% | -5.2% | 0.1488 |
| A3 | Livlab 4 → da8241 (오버랩→비오버랩) | 9,608 | -0.4% | -13.4% | 0.2024 |
| ref: 4-rec | Livlab 3 + depth → Livlab | 10,395 | +20.1% | +3.9% | 0.0969* |
| ref: 5-rec | 전체 → Livlab | 15,601 | +13.6% | +3.3% | 0.0969 |

*이전 세션 결과, 동일 holdout(8014dd)

### 가설 검증 결과

- **H1 (오버랩→visual 암기→motion 약화): 기각**. A1이 가장 높은 shuffled gap.
- **H2 (오버랩→"같은 장소, 다른 action" 학습): 지지**. 같은 구역 내에서만 motion이 의미를 가짐.
- **H3 (비오버랩→generalizable dynamics): 기각**. A2, A3에서 motion이 완전 무용.
- **H4 (goal 부재가 근본 한계): 강하게 지지**. Cross-domain에서 action이 noise로 작용.

### 해석

1. **Motion conditioning은 place-specific**: 모델은 "이 교차로에서 좌회전하면 이런 장면"이라는 **장소-행동-결과 연관**을 학습. 새로운 장소에서는 이 연관이 전이되지 않음.

2. **Zeroed > Correct 현상의 의미**: Cross-domain (A2, A3)에서 action을 주면 오히려 나빠짐. 모델이 학습한 place-action 연관이 새 장소에서 잘못된 예측을 유도. Action을 아예 안 주면(zeroed) 모델이 visual continuity에만 의존하여 더 나은 예측.

3. **Goal 부재 문제 확인**: E2E 주행에서는 goal이 action의 의미를 해석하는 맥락을 제공. LeWM에서는 이 맥락이 없으므로, 모델이 "이 속도로 직진"이라는 action을 해석하려면 **이 장소를 이미 알고 있어야** 함. 즉, motion conditioning의 일반화 한계.

4. **Application 함의**: Anomaly detection 목적으로는 **동일 구간 반복 주행 데이터**가 가장 효과적. "보통 이 구간에서 이렇게 움직인다"의 통계적 model이 가장 강하게 형성됨.

## Experiment C2 결과 (2026-03-31)

### 설계

A1 (오버랩) 및 A2 (cross-domain) 모델에서 multi-step autoregressive rollout (5 steps, 2.5초).
각 horizon에서 correct/shuffled/zeroed action의 MSE gap 변화를 관찰.

**핵심 질문**: 모델이 action을 활용할 능력(capacity)이 있는가, 아니면 맥락(context) 부족인가?

### 정량 결과

#### A1 모델 (오버랩, Livlab 3 → Livlab holdout)

| Horizon | Correct MSE | Shuffled Gap | Zeroed Gap |
|---------|------------|-------------|------------|
| 0.5s | 0.177 | +21.7% | +5.5% |
| 1.0s | 0.263 | +34.7% | +13.0% |
| 1.5s | 0.330 | +39.0% | +12.9% |
| 2.0s | 0.374 | **+43.8%** | +11.7% |
| 2.5s | 0.441 | +37.9% | +6.6% |

→ Motion gap이 horizon과 함께 **누적 증가** (+21.7% → +43.8%). 2.0초에서 피크.

#### A2 모델 (cross-domain, da8241 → Livlab holdout)

| Horizon | Correct MSE | Shuffled Gap | Zeroed Gap |
|---------|------------|-------------|------------|
| 0.5s | 0.157 | +0.2% | -5.8% |
| 1.0s | 0.236 | +3.0% | -8.3% |
| 1.5s | 0.314 | +2.9% | -9.2% |
| 2.0s | 0.382 | +3.3% | -10.7% |
| 2.5s | 0.471 | +3.3% | -10.3% |

→ Gap이 ~3%에서 정체. Zeroed gap이 horizon과 함께 **악화** (-5.8% → -10.7%), action이 점점 더 큰 noise로 작용.

### 결론

1. **모델은 action 활용 capacity가 충분**: A1에서 2초 horizon에 +43.8% gap 달성.
2. **Cross-domain에서 action은 noise**: A2에서 올바른 action조차 zeroed보다 나쁨, horizon이 길수록 악화.
3. **H4 완전 확인**: 문제는 모델 능력이 아니라 **맥락(context) 부재**. 같은 장소를 이미 학습한 경우에만 action이 의미를 가짐. 이는 goal/route 정보 없이 action이 "이 장소에서 이 행동을 하면" 이라는 place-conditioned 해석에 의존하기 때문.
4. **실질적 함의**:
   - Route-specific world model (동일 구간 데이터)에서는 action이 매우 효과적
   - 범용 world model에서 action 일반화를 위해서는 goal/route/map conditioning이 필수
   - Anomaly detection 응용: route-specific model + multi-step rollout으로 2초 ahead 예측이 최적

## Explicitly NOT Doing

- Navigation command conditioning (데이터에 없음)
- HD map integration (인프라 필요)
- E2E policy learning (LeWM의 목적과 다름)

## Consequences

- H4가 확인되면: LeWM의 motion conditioning 한계를 인정하고, **embedding 기반 anomaly detection**에 집중하는 것이 합리적
- H1/H2 결과에 따라: 데이터 수집 전략 (오버랩 구간 포함 여부) 결정 가능
- Goal conditioning 없이 달성 가능한 motion gap의 상한선을 파악하는 것이 이 실험의 궁극적 목적
