# Findings

## 2026-03-28: RTB 데이터의 velocity 토픽으로 CAN 데이터 대체 가능
**맥락**: LeWM 학습용 action space 구성 시 CAN 데이터(조향각, 스로틀, 브레이크)가 없음
**발견**: rosbag-ingest가 추출한 velocity 토픽 (vx, vy, angular_yaw)이 CAN 대체로 사용 가능. 100Hz로 충분한 시간 해상도, frame_index.npy로 카메라 프레임에 정확히 정렬됨
**영향**: CAN 데이터 없이도 자율주행 world model 학습이 가능. action=[vx, vy, angular_yaw] (3D), proprio=[speed, heading_rate, accel_xyz, gyro_xyz] (8D)

## 2026-03-28: RTB recording 데이터 규모
**맥락**: rosbag-ingest 출력 73개 recording을 LeWM HDF5로 변환
**발견**: 1 recording (Gian-Pankyo, 116 batches) = 69,368 frames, HDF5 7.82GB (224x224 gzip). vx 범위 0~19.4 m/s (정지~시속 70km), 약 115분 주행 데이터
**영향**: 전체 73 recordings 변환 시 ~500만 프레임, ~500GB 예상. 단일 recording으로도 69K 학습 시퀀스 확보

## 2026-03-28: swm.data.HDF5Dataset의 action 처리 방식
**맥락**: HDF5에 action을 pre-aggregated로 저장할지 raw로 저장할지 결정
**발견**: HDF5Dataset은 action을 카메라 프레임 rate로 저장하고, __getitem__에서 frameskip만큼 연속 action을 묶어 reshape. pixels/proprio는 frameskip 간격으로 subsampling
**영향**: 변환 스크립트에서 frameskip 처리 불필요, 10Hz raw action만 저장하면 됨

## 2026-03-28: E0 Motion-Conditioning Ablation 결과
**맥락**: 학습된 모델이 action (ego-motion) input을 실제로 사용하는지 검증
**발견**:
- correct motion MSE=0.056, shuffled MSE=0.076 (+35%), zeroed MSE=0.070 (+24%)
- Cosine similarity 차이는 작음 (0.948 vs 0.944) → visual continuity가 dominant signal
- 고속 구간에서 motion 기여도가 가장 큼, sharp turn에서 전체적으로 예측 어려움
**영향**: Action proxy가 유효하지만 아직 보조적 역할. 더 긴 horizon (num_preds 증가)이나 더 많은 데이터로 motion 의존도 강화 필요

## 2026-03-29: Cross-route holdout에서 motion conditioning 효과 약화
**맥락**: 4개 route로 학습, 1개 holdout (Namyang-Gian)에서 E0 ablation 재실행
**발견**:
- 단일 recording에서 correct vs shuffled MSE 차이 +35% → holdout에서 +8%로 축소
- zeroed motion이 correct보다 오히려 MSE가 낮음 (-1.9%)
- 그러나 visual representation 일반화는 우수: speed clustering 유지, NN 검색 sim=0.97+
- 전체 MSE가 0.056 → 0.028로 낮아짐 (다양한 데이터 학습 효과)
**영향**: 현재 1-step prediction (0.5s ahead)에서는 visual continuity가 dominant. motion 의존도를 높이려면 num_preds 증가 (더 먼 미래 예측) 필수. Codex 경고가 현실화됨

## 2026-03-29: Feature/Horizon 변경으로 motion conditioning 강화 불가
**맥락**: ADR-002 실험 — 4-experiment matrix (3D/4D action × num_preds 1/4)
**발견**:
- Horizon 증가 (num_preds=4): correct MSE 상승하지만 shuffled gap은 +8%→+6.5%→+3.9%로 오히려 악화
- Δψ 추가: shuffled gap 변화 없음 (+8.3% 동일)
- 모든 조건에서 zeroed motion이 correct보다 MSE 낮음 (action이 noise 역할)
- AdaLN-zero 초기화로 인해 predictor가 action modulation을 0으로 유지하는 것이 최적해
**영향**: motion-conditioned prediction을 원한다면 아키텍처 수준 변경 필요 (state encoder, pixel masking, action-bottleneck 등). 현재 모델은 "ego-motion-independent visual predictor"로 reframe하고, anomaly detection은 embedding 기반으로 진행 가능

## 2026-03-30: VP Auxiliary Supervision이 motion conditioning을 간접적으로 개선
**맥락**: L2 구현 — VP lane mask + depth map을 auxiliary loss로 추가
**발견**:
- Lane loss 15.8→7.9 (50% 감소): encoder가 차선 구조를 학습
- E0 holdout에서 zeroed gap이 -1.9% → +2.3%로 반전 (correct가 zeroed보다 나아짐)
- Shuffled gap +8.3% → +10.0%로 소폭 개선
- 가설: VP supervision이 encoder에 구조적 이해를 강제 → motion input이 "어떤 구조가 어떻게 변하는가"의 맥락에서 유용해짐
**영향**: VP auxiliary supervision은 motion conditioning 개선에 효과적. L3 (LiDAR), L4 (cross-attention) 조합으로 추가 개선 기대

## 2026-03-30: OccAny 3D depth supervision으로 motion gap 20% 달성
**맥락**: ADR-006 — VP monocular depth를 OccAny 3D reconstruction depth로 교체
**발견**:
- Depth loss: 1.14 → 0.32 (72% 감소, VP 대비 더 빠르게 수렴)
- E0 shuffled gap: +9.7% (VP) → **+20.1%** (OccAny) — 2배 증가
- E0 zeroed gap: +2.4% → +3.9% — correct가 zeroed보다 확실히 나음
- OccAny confidence-weighted loss가 noisy depth를 효과적으로 처리
**영향**: 3D reconstruction 기반 depth가 monocular depth보다 확실히 우월. LiDAR sparse GT를 추가하면 더 개선 가능. 모델이 "물체 거리 + ego motion → 장면 변화" 관계를 학습함

## 2026-03-31: 5-rec OccAny depth 학습에서 motion gap 감소
**맥락**: da8241 (Gian-Pankyo) 추가하여 5개 recording으로 OccAny depth 재학습
**발견**:
- Shuffled gap: +20.1% (4-rec) → +13.6% (5-rec) — 6.5pp 감소
- Zeroed gap: +3.9% → +3.3% — 소폭 감소
- da8241이 train의 49% (7,644/15,601 seq)를 차지하는 데이터 불균형
- Depth loss 수렴은 정상 (1.12 → 0.21), sigreg_loss 증가 (54.5 → 65.4)
**영향**: 단순히 데이터를 늘리는 것이 motion conditioning을 개선하지 않음. 데이터 균형, 공간적 오버랩 특성, goal conditioning 부재가 복합적으로 작용. ADR-007에서 가설 정리

## 2026-03-31: Goal/Route 정보 부재가 motion conditioning의 근본적 한계일 수 있음
**맥락**: E0 결과 분석 중 E2E 주행과의 차이점 고찰
**발견**:
- E2E 주행: goal → action (action은 의도의 결과), LeWM: action은 관측값 (의도 없음)
- 교차로에서 "왜 좌회전했는지" 모르면, action이 다음 scene 예측에 informative하지 않음
- ADR-002의 "AdaLN-zero가 action modulation을 0으로 유지" 현상의 근본 원인일 수 있음
- 이는 JEPA 아키텍처 한계가 아니라 **task formulation의 한계**
**영향**: motion conditioning 강화 방향이 "더 나은 depth"가 아니라 "goal/route proxy 도입"일 수 있음. 단, anomaly detection 목적이면 통계적 regularity만으로 충분할 가능성

## 2026-03-31: Motion conditioning은 place-specific이며 cross-domain 전이 불가
**맥락**: ADR-007 Experiment A — 오버랩/비오버랩 데이터 분리 실험
**발견**:
- 오버랩 train → 오버랩 holdout (A1): shuffled gap **+24.3%** (최고)
- 비오버랩 train → 오버랩 holdout (A2): shuffled gap **+1.4%** (무용)
- 오버랩 train → 비오버랩 holdout (A3): shuffled gap **-0.4%** (무용, zeroed가 -13.4%로 오히려 나음)
- Cross-domain에서 action이 noise로 작용: 학습된 place-action 연관이 새 장소에서 잘못된 예측 유도
**영향**: motion conditioning의 일반화에는 goal/route context가 필수. 현재 LeWM으로는 **동일 구간 반복 주행 데이터**에서만 motion이 유효. Anomaly detection은 route-specific model로 접근해야 함

## 2026-03-31: Multi-step rollout에서 motion gap이 horizon과 누적 증가 (최대 +43.8%)
**맥락**: C2 실험 — A1 (오버랩) 모델로 5-step autoregressive rollout
**발견**:
- 오버랩 모델: 1-step +21.7% → 4-step +43.8% (2초 ahead)로 gap 누적
- Cross-domain 모델: 1-step +0.2% → 4-step +3.3%로 정체, zeroed가 -10.7%로 점점 악화
- 모델은 action 활용 capacity가 충분하나, 같은 구역에서만 작동
- Action이 cross-domain에서 horizon이 길수록 더 큰 noise가 됨
**영향**: route-specific model + 2초 horizon rollout이 anomaly detection 최적 구성. 범용 모델에서 action 일반화는 goal conditioning 없이 불가

## 2026-03-31: Scene type별 motion sensitivity — stop이 gap을 지배
**맥락**: C3 실험 — A1 모델을 scene type (straight/curve/accel-decel/stop)으로 계층화
**발견**:
- stop: shuffled gap **+73.2%** (전체 36%) — 정차 시 "변화 없음" 예측이 올바르므로 shuffled가 크게 틀림
- curve: +15.7%, straight: +17.9%, accel/decel: +13.8% — moving scene 간 차이 작음
- Multi-step rollout에서는 모든 scene type이 +35~43%로 수렴
- Scene type은 place-specific behavior를 설명하지 못함 — visual place identity가 핵심
**영향**: coarse scene labels는 motion conditioning의 조건변수로 불충분. Lane topology + maneuver + traffic state 수준이 필요 (ADR-008)

## 2026-03-31: LeWM은 memorization이 아닌 dynamics를 학습 (P0-A 확인)
**맥락**: P0-A retrieval baseline — Visual/Action/Proprio kNN vs LeWM predictor
**발견**:
- LeWM (MSE 0.175) vs Visual kNN (0.319): **LeWM이 82% 우위** → memorization이 아님
- No-change baseline (0.157)이 kNN보다 나음 → embedding 공간에서 인접 프레임 변화가 작음
- No-change가 LeWM보다 10% 나은 것은 단일프레임 반복 history 설정의 한계 (보수적 조건)
- Action sensitivity: correct vs shuffled +23.3% 유지
- kNN에 action을 추가하면 오히려 악화 (+125%) → retrieval에서 action은 noise
**영향**: LeWM의 +43.8% motion gap은 진짜 dynamics 학습의 결과. Corridor planning (Track 1)에 투자 정당화됨. 모델이 place-action 연관을 넘어 시각적 dynamics를 이해하고 있음

## 2026-03-31: World model counterfactual이 물리적으로 plausible (P0-D)
**맥락**: P0-D counterfactual sanity check — logged/brake/steer_left/steer_right/faster 비교
**발견**:
- 3/4 sanity check 통과: logged가 GT에 가장 가까움, divergence가 horizon과 증가, left/right가 반대 방향
- Steer left/right cosine similarity = -0.17 (모든 horizon step에서 음수) → 반대 방향 확인
- Brake가 가장 큰 divergence (0.41), faster가 가장 작음 (0.056) → 물리적으로 합리적
- Planning ranking은 실패 (logged 13.5% rank-1 vs 20% random) — 2.5s autoregressive error 누적 때문
- 하지만 logged의 평균 rank (2.48)은 random (3.0)보다 나음
**영향**: world model이 action 변경에 물리적으로 일관된 반응을 보임. Short-horizon (1-2s) counterfactual evaluator로 사용 가능. Long-horizon planning에는 uncertainty-aware 접근 필요

## 2026-03-31: Pseudo-maneuver labeling이 저렴하고 일관성 높음 (P0-B)
**맥락**: P0-B — IMU/velocity 데이터에서 미래 2초 궤적 기반 maneuver label 도출
**발견**:
- 전체 18,970 프레임 분포: straight 44.7%, stop 26.1%, decel 9.9%, accel 7.0%, left 6.4%, right 5.9%
- Temporal consistency 94.7~98.0% — 인접 프레임이 같은 label (평균 run length 17~39 프레임)
- da8241 (고속도로): straight 68%, left/right 합계 1.2% — 회전 거의 없음
- Livlab routes: left/right 합계 19~24%, stop 21~46% — 시가지 주행 특성 명확
- Codex 경고대로 straight이 지배적 → 평가 시 반드시 stratify 필요
**영향**: pseudo-maneuver labels가 Track 2 token conditioning의 입력으로 사용 가능. 품질 충분 (consistency >94%), 비용 제로 (기존 데이터만 사용). 다만 left/right가 전체의 12%에 불과하여 class imbalance 주의
