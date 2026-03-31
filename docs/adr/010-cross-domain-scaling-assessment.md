# ADR-010: Cross-Domain Scaling Assessment — Data Requirements and Strategic Direction

**Date**: 2026-03-31
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.3-codex)
**Debate Style**: constructive (2 rounds)
**Confidence**: 8/10 (assessment), 3/10 (cross-domain success with current data)

## Context

After extensive experimentation (ADR-007, 008, 009), the question became: is cross-domain motion conditioning achievable with our data, and if not, what should we do instead?

## Key Data

### Current Data (5 recordings, 2 areas)
- Overlap: +24.3% shuffled gap, +43.8% at 2s rollout
- Cross-domain: ~0%, action is noise
- Maneuver token: +16.7% peak, declines to +6.5% (overfitting)
- LeWM beats kNN by 82% — real dynamics, not memorization

### Available Data (73 recordings)
| Area Cluster | Recordings | Notes |
|-------------|-----------|-------|
| Pankyo (시내) | ~16 | 판교 내부 순환 |
| Gian↔Pankyo (왕복) | ~31 | 같은 도로 양방향 — 실질 1개 area |
| Livlab (순환코스) | ~20 | Rt-C-1~8, 같은 구역 |
| Others (Namyang, Chunhju, Songdo) | 4 | 유일한 장거리, 각 1개 |

**실질 독립 지역: 4-5개** (Codex 권장 최소 6개, 권장 8-10개)

### Industry Comparison
| System | Data Scale | Structured Priors |
|--------|-----------|-------------------|
| GAIA-1 | 4,700h, ~420M images | Multi-billion params |
| DriveDreamer | ~1M frames (nuScenes) | HD map + 3D boxes |
| UniSim | Massive log data | Log-grounded simulation |
| **LeWM (ours)** | **~60K frames, 5 recordings** | **OccAny depth only** |

## Decision

### Primary Direction: Route-Specific Applications
현재 검증된 능력을 활용하여 실용적 application 구축:

1. **Route anomaly detection** (최우선) — surprise score on known corridors
2. **Auto-labeling service** — embedding similarity for scene classification
3. **Counterfactual analysis** — what-if scenarios on logged drives

### Secondary Direction: 73-Recording Go/No-Go (조건부)
새로운 지역 데이터가 추가되어 독립 지역 6개 이상이 확보되면:
- VP depth 전체 + OccAny 15-30% 샘플링
- Area-holdout 평가
- Gate: unseen area shuffled gap >10%, 2s rollout >15%
- 예상 소요: 5-11일

### Cross-Domain은 장기 목표로 전환
현재 4-5개 지역으로는 불충분. 필요 조건:
- 최소 6개, 권장 8-10개 독립 지역
- 0.5-1M+ diverse frames
- Structured conditioning (maneuver token 이상)

## Debate Summary

### Round 1 (Codex)
- 5 recordings으로 cross-domain 불가능 — 명확
- 73 recordings으로 연구 수준 가능하나 robust deployment은 아님
- 0.5-1M frames: 첫 signal, 2-5M: short-horizon 유용
- 지역 다양성 > 프레임 수
- Route-specific application이 최우선

### Round 2 (Claude → data analysis)
- 73개 recording의 실질 독립 지역이 4-5개에 불과
- Codex 권장 최소(6개)에 미달
- Cross-domain go/no-go는 새 지역 데이터 확보 후로 연기

## Consequences
- Route-specific anomaly detection 즉시 구축 → 가장 높은 impact/effort
- Cross-domain은 데이터 수집 전략과 연계 (새 지역 recording 확보)
- 현재 architecture (20M JEPA + maneuver token)는 route-specific에서 충분히 강력
