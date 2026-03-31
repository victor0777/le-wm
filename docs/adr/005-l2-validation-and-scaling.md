# ADR-005: L2 Validation, Ablation, and Scaling Plan
**Date**: 2026-03-30
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.4)
**Debate Style**: constructive
**Confidence**: 8/10

## Context

L2 (VP auxiliary supervision)에서 질적 전환 발생:
- Lane loss 50% 감소 — encoder가 차선 구조 학습
- E0 zeroed gap이 -1.9% → +2.3%로 반전 — motion이 더 이상 noise가 아님
- Shuffled gap +8.3% → +10.0%로 소폭 개선

이 결과가 진짜인지, 무엇이 원인인지, 스케일링 시 유지되는지 검증 필요.

## Decision

### 실험 계획 (총 ~6시간)

| # | 실험 | 목적 | 소요 |
|---|------|------|------|
| 1 | Lane-only aux loss | 차선 구조가 핵심인지 분리 검증 | 1.5h |
| 2 | Depth-only aux loss | 깊이 이해가 핵심인지 분리 검증 | 1.5h |
| 3 | Winner를 16 VP recordings로 확대 | 구조적 supervision의 스케일링 효과 | 2h |
| 4 | Scaled 모델로 VP holdout + non-VP holdout 평가 | 도메인 내 + 전이 테스트 | 30min |
| 5 | Auto-label benchmark (L1 vs L2 vs L2-scaled) | Encoder 품질 + 실용적 산출물 | 30min |
| 6 | 사고 영상 재분석 (best model) | Anomaly 신호 개선 확인 | 10min |

### 실험 설계 원칙

- **1 seed로 ablation** (빠른 스크리닝). Winner만 2nd seed로 확인
- **Train on VP recordings only** (구조적 supervision 효과 격리)
- **Zero-shot eval on non-VP recordings** (전이 테스트, fine-tuning 없음)
- **Auto-label: anchor/eval recording 분리** (leakage 방지)

### Branching Logic (결과에 따른 분기)

| 결과 | 해석 | 다음 단계 |
|------|------|----------|
| lane > depth | 차선 구조가 핵심 lever | LiDAR (L3) 우선순위 낮춤 |
| depth > lane | 깊이 이해가 핵심 | L3 (LiDAR real depth) 즉시 진행 |
| lane+depth >> either alone | Multi-task stacking이 핵심 | 더 많은 supervision 추가 (L3 + segmentation) |
| Scaled VP → non-VP 전이 성공 | **진짜 breakthrough** | Application 우선 (auto-label, anomaly) |
| Scaled VP → non-VP 전이 실패 | VP supervision이 domain-specific | Cross-attention 아키텍처 변경 검토 |

### Metrics (3개 동시 추적)

| Metric | 측정 대상 | 용도 |
|--------|----------|------|
| E0 motion ablation (shuffled/zeroed gap) | Motion conditioning 품질 | World model 평가 |
| Auto-label accuracy (road_type NN) | Encoder semantic 품질 | 실용적 산출물 |
| Accident surprise ratio | Anomaly detection 성능 | Downstream application |

## Debate Summary

### Round 1 (Codex)
- L2 breakthrough이지만 "what caused it" 불명확 → ablation 필수
- Cross-attention, OccAny, LiDAR 모두 시기상조 → L2 먼저 확정
- Auto-labeling은 side track으로 병행

### Round 2 (Claude pushback → 합의)
- 2-seed 대신 1-seed ablation → 합의 (seed variance보다 data diversity가 리스크)
- Scaling: train within VP, eval on both VP + non-VP → 합의
- Auto-label benchmark의 anchor/eval 분리 → 합의

### Round 3 (합의)
- 전체 합의, confidence 8/10
- 리스크: metric mismatch, holdout bias, auto-label leakage, VP subset bias

## Risks

1. E0 "winner"가 auto-label이나 anomaly에선 worst일 수 있음 → 3개 metric 동시 추적
2. Non-VP holdout 1개로는 전이 결론 불충분 → 가능하면 2개 이상
3. VP 20개 recording이 전체 73개와 분포 다를 수 있음
4. Auto-label anchor/eval이 같은 recording이면 과대평가

## Action Items

1. [ ] `train_vp.py` 수정: lane-only, depth-only 모드 지원 (loss weight 0으로)
2. [ ] Exp 1-2: lane-only / depth-only 학습 + E0 평가
3. [ ] Winner 확정 후 16 VP recording으로 scaling
4. [ ] Non-VP holdout에서 zero-shot E0 평가
5. [ ] Auto-label benchmark 구현 (rtb-vlm segments.json 기반)
6. [ ] Accident surprise 재분석

## Explicitly NOT Doing

- Cross-attention 아키텍처 변경 (ablation 결과 후)
- L3 LiDAR supervision (depth ablation 결과 후)
- OccAny fusion (L2 스케일링 후)
- 2-seed validation (winner 확정 전)
- Non-VP fine-tuning (zero-shot 전이 먼저)
