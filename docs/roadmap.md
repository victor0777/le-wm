# Roadmap

## Phase 1: 초기 검증 (완료)
- [x] RTB → HDF5 변환 파이프라인 구축 (`scripts/convert_rtb_to_hdf5.py`)
- [x] Velocity/IMU 기반 action space 설계 [vx, vy, angular_yaw]
- [x] 단일 recording (69K frames) 변환 및 학습 (5 epochs, loss 5.0→0.3)
- [x] Embedding 시각화 분석 (t-SNE, cosine similarity, NN retrieval)
- [x] Codex debate → ADR-001 작성

## Phase 2: 평가 체계 구축 ← **current**
- [x] E0: motion-conditioning ablation (correct/shuffled/zeroed) — motion +35% MSE improvement ← **done**
- [x] Normalizer data leak 수정 (train-only fit) — train.py, utils.py 수정 완료
- [x] Whole-recording train/test split 구현 (`train_multi.py`) — 4 train / 1 holdout
- [x] E1: held-out recording prediction — holdout loss 10.1→8.1, cross-route 일반화 확인. E0 holdout: motion +8% (약함)
- [ ] E4: linear scene probe (highway/urban/intersection)

## Phase 3: 데이터 확장 + Scaling Curve
- [x] Diversity-based recording selection (4개 추가: Chunhju-Gian, Gian-Songdo, Pankyo-Gian, Namyang-Gian)
- [ ] Scaling curve: 1 → 4 → 16 recordings
- [ ] E5: cross-recording GNSS place recognition
- [ ] vy ablation 실험

## Phase 4: Anomaly Detection Pipeline
- [x] E3: surprise score 구현 + 사고 영상 적용 — 사고/비사고 1.35x 비율, pre-crash signal 약함
- [ ] Domain gap 축소 (사고 영상 정상 구간으로 fine-tuning)
- [ ] Surprise 변화율 기반 사고 징후 탐지
- [ ] ROC curve / threshold 최적화
- [ ] traffic_rule_checker / accident_analysis 연동

## Phase 5: L2 Validation & Scaling (ADR-005)
- [x] Lane-only ablation → shuffled +4.8%, zeroed -0.3% (motion 기여 없음)
- [x] Depth-only ablation → shuffled +9.7%, zeroed +2.4% (**depth가 핵심**)
- [ ] Depth-only를 16 VP recordings로 scaling
- [ ] Non-VP holdout zero-shot 전이 테스트
- [ ] Auto-label benchmark (L1 vs L2)
- [ ] Accident surprise 재분석 (L2 model)

## Phase 5.5: OccAny Depth Supervision (ADR-006)
- [x] OccAny 프로젝트에 RTB inference 요청 → 4개 완료 (da8241 진행중)
- [x] OccAny depth HDF5 변환 (4 recordings, 10K frames)
- [x] Confidence-weighted depth loss 구현
- [x] 학습 + E0 평가 → **shuffled gap +20.1% 달성!** (VP +9.7% 대비 2배)
- [ ] da8241 변환 + 5개 recording 전체로 OccAny depth 재학습 ← **in progress**
- [ ] LiDAR → camera projection (calibration 확보 완료, 구현 필요) ← **next**
- [ ] Auto-labeling benchmark (L1 vs L3 encoder 비교) ← **next**

## Phase 6: Auto-Labeling 서비스 (ADR-003)
- [ ] Embedding 추출 CLI (`scripts/extract_embeddings.py`)
- [ ] rtb-vlm segments.json → anchor embedding 구축
- [ ] NN auto-labeling + temporal smoothing 구현
- [ ] 출력 포맷 rtb-vlm segments.json 호환

## Phase 7: E2E Learning Roadmap (ADR-008) ← **current**

### Phase 7.0: Validation Experiments (이번 주)
- [x] C3 scene stratified analysis — stop +73.2%, moving scene 간 차이 작음
- [x] Codex debate → ADR-008 작성
- [x] P0-A: Retrieval baseline — **LeWM 82% 우위, dynamics 확인** ← **done**
- [x] P0-B: Pseudo-maneuver labeling — 6 classes, consistency >94%, labels saved ← **done**
- [x] P0-D: Counterfactual sanity — 3/4 checks passed, plausible counterfactuals ← **done**

### Phase 7.1: Two Parallel Tracks (2-4주, P0 결과에 따라)
- [x] Track 1: Corridor planning — CEM works (24.8% improvement), partial pass ← **done**
- [x] Track 2: Maneuver token conditioning — **V3 cross-domain +16.7%, gate passed!** ← **done**
- [x] Track 2 decision gate: cross-domain shuffled gap > 10% — **V3 +16.7% > 10% PASSED**
- [x] Track 2 follow-up: V1 full 15-epoch — **+25.8%, regression 해결, A1 초과** ← **done**
- [x] Track 1 follow-up: VP/OccAny-based cost function for planning — composite cost rank 22.9 vs v1 MSE 29.9, top-10 29% ← **done**

### Phase 7.2: Structured Scene Transfer (1-2개월, Track 2 성공 시) ← **current**
- [x] Topology token labeling (curvature x speed_zone x dynamics = 60 classes) ← **done**
- [x] Dual-token (maneuver+topology) training V3-topo — **shuffled gap +2.5% (regression!)** ← **done**
- [ ] Auto-derived lane topology from VP + OccAny
- [ ] Cross-route transfer with structured conditioning

### Phase 7.3: Goal-Conditioned World Model (2-3개월, Phase 7.2 성공 시)
- [ ] Route/goal embedding
- [ ] Calibrated counterfactual scenario generation

## Phase 8: 확장
- [ ] L3 LiDAR supervision
- [ ] Cross-attention (scaling 후에도 motion 약할 경우)
- [ ] ISO 34502 시나리오 분류 (vehicle-tracker + perception 결합)
