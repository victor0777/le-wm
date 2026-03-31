# ADR-003: LeWM Embedding 기반 Auto-Labeling 서비스
**Date**: 2026-03-29
**Status**: Proposed
**Participants**: Claude + User
**Confidence**: 7/10

## Context

LeWM으로 RTB 주행 데이터 238K 프레임을 학습한 결과, visual representation이 cross-route에서도
일반화되는 것을 확인함. 한편 rtb-vlm에서 Qwen-VL 기반 scene classification (road_type, weather,
lighting, traffic_density, events)이 이미 수행되어 있음.

VLM 추론은 정확하지만 비용이 높음 (50 프레임 → 수 분). le-wm embedding은 69K 프레임을 ~30초에
처리 가능. 이 속도 차이를 활용하여 **소수 VLM 라벨 → 대량 auto-labeling** 파이프라인을 구축할 수 있음.

## Decision

### le-wm을 "주행 장면 Embedding + Auto-Labeling 서비스"로 위치시킨다

#### 역할 정의

```
le-wm의 역할:
  ✅ 장면 수준 auto-labeling (road_type, weather, lighting, traffic_density)
  ✅ 시간 구간 분할 (segment boundary detection via embedding 변화)
  ✅ 이상 탐지 (surprise score → 평소와 다른 구간)
  ✅ 장면 검색 (유사 장면 retrieval)

  ❌ 객체 수준 시나리오 (cut-in, lane change, overtaking) — 단독 불가
  ❌ ISO 34502 행동 시나리오 분류 — 다른 프로젝트와 결합 필요
```

#### Auto-Labeling 파이프라인

```
Phase 1: Anchor 구축 (1회)
  rtb-vlm segments.json (VLM 라벨, 소수 프레임)
    → 해당 프레임의 le-wm embedding 추출
    → 라벨별 anchor embedding 저장 (centroid + 분산)

Phase 2: Auto-Label (새 recording마다)
  새 recording → le-wm embedding 추출 (전체 프레임)
    → 각 프레임의 embedding과 anchor embedding간 cosine similarity
    → 가장 가까운 anchor의 라벨 할당
    → 연속 프레임 smoothing → temporal segment 생성

Phase 3: 검수 (선택)
  confidence가 낮은 프레임만 VLM 재검증 또는 사람 검수
```

#### 라벨링 가능 범위

| 카테고리 | 라벨 | 방식 | 정확도 기대 |
|---------|------|------|-----------|
| **road_type** | parking, urban, tunnel, highway, intersection, rural | Embedding NN | 높음 (시각적으로 뚜렷) |
| **weather** | clear, cloudy, rain, fog | Embedding NN | 중간 (미묘한 차이) |
| **lighting** | day, night, dawn/dusk, tunnel_dark | Embedding NN | 높음 |
| **traffic_density** | low, medium, high | Embedding NN | 중간 |
| **segment boundary** | 장면 전환 시점 | Embedding 변화율 | 높음 |
| **anomaly** | 평소와 다른 구간 | Surprise score | 중간 (calibration 필요) |

#### ISO 34502 시나리오는 다른 프로젝트와 결합

```
ISO 34502 시나리오 분류에 필요한 것:

vehicle-tracker → "누가 어디서 움직였는가" (BEV track, 상대 위치)
perception     → "차선 위치, 객체 위치" (3D detection, lane marking)
le-wm          → "이 상황이 평소와 다른가" (surprise) + "어떤 환경인가" (scene context)
rtb-vlm        → "무슨 일이 일어나고 있는가" (자연어 서술)

결합 예시:
  cut-in 감지 = vehicle-tracker(타차 lateral 이동) + perception(차선 침범) + le-wm(surprise 상승)
  lane change = vehicle-tracker(ego lateral 이동) + le-wm(scene=highway) + perception(차선 변경)
```

## Architecture

### 컴포넌트

```
┌─────────────────────────────────────────────────┐
│                le-wm Auto-Labeling              │
│                                                 │
│  ┌───────────┐    ┌──────────────┐             │
│  │ ViT-tiny  │───▶│  Embedding   │             │
│  │ Encoder   │    │  (192-dim)   │             │
│  └───────────┘    └──────┬───────┘             │
│                          │                      │
│         ┌────────────────┼────────────────┐     │
│         ▼                ▼                ▼     │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  NN Label   │ │  Segment    │ │ Surprise  │ │
│  │  Assignment │ │  Boundary   │ │  Score    │ │
│  └─────────────┘ └─────────────┘ └───────────┘ │
│         │                │                │     │
│         ▼                ▼                ▼     │
│  ┌──────────────────────────────────────────┐   │
│  │        Output: segments.json             │   │
│  │  - road_type, weather, lighting per seg  │   │
│  │  - anomaly flags                         │   │
│  │  - confidence scores                     │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         │                              ▲
         ▼                              │
  ┌──────────────┐              ┌───────────────┐
  │  Downstream  │              │   rtb-vlm     │
  │  Projects    │              │   Anchors     │
  │  - rtb-viewer│              │  (VLM labels) │
  │  - traffic_  │              └───────────────┘
  │    rule_chk  │
  │  - accident_ │
  │    analysis  │
  └──────────────┘
```

### 데이터 흐름 (Cross-Project)

```
rosbag-ingest ──────── 카메라 프레임 ──────────▶ le-wm (embedding 추출)
                                                    │
rtb-vlm ────── VLM 라벨 (소수 anchor) ─────────▶ le-wm (anchor 등록)
                                                    │
                                                    ▼
                                            auto-labeled segments
                                                    │
                    ┌───────────────────────────────┼──────────────────┐
                    ▼                               ▼                  ▼
              rtb-viewer                   traffic_rule_checker    accident_analysis
              (시각화)                      (scene context)        (anomaly detection)
                                                    │
                                                    ▼
                                    vehicle-tracker + perception 결합
                                    → ISO 34502 시나리오 분류
```

### 출력 포맷 (rtb-vlm segments.json 호환)

```json
{
  "video_id": "RECORDING_ID",
  "source": "le-wm-auto-label",
  "model": "lewm_rtb_multi_epoch_9",
  "anchor_source": "rtb-vlm/exp10_segments",
  "field_segments": {
    "road_type": [
      {
        "field": "road_type",
        "start_s": 0.0,
        "end_s": 62.0,
        "label": "urban",
        "confidence": 0.94,
        "method": "embedding_nn"
      }
    ],
    "lighting": [...],
    "traffic_density": [...]
  },
  "anomalies": [
    {
      "start_s": 45.2,
      "end_s": 47.8,
      "surprise_mean": 12.5,
      "surprise_zscore": 3.2
    }
  ]
}
```

## Cost-Benefit 분석

| 방식 | 1 recording (69K frames) | 정확도 | 비용 |
|------|-------------------------|--------|------|
| **VLM 전수 분류** (rtb-vlm) | ~수 시간 | 높음 | GPU 시간 + API 비용 |
| **VLM 샘플링** (현재 rtb-vlm) | ~수 분 (50 frames) | 중간 | 낮음 |
| **le-wm auto-label** | **~30초** | 중간-높음 | **거의 무료** |
| **사람 annotation** | ~수 일 | 최고 | 인건비 |

le-wm은 VLM 대비 **100-1000x 빠르고**, 정확도는 anchor 품질에 의존함.

## Action Items

1. [ ] Embedding 추출 CLI 구현 (`scripts/extract_embeddings.py`)
2. [ ] rtb-vlm segments.json → anchor embedding 구축 스크립트
3. [ ] NN auto-labeling + temporal smoothing 구현
4. [ ] 출력 포맷을 rtb-vlm segments.json과 호환
5. [ ] 정확도 평가: VLM 라벨 vs auto-label 일치율

## Explicitly NOT Doing

- ISO 34502 행동 시나리오 (cut-in, lane change) 단독 분류
- 객체 수준 detection/tracking (vehicle-tracker, perception의 역할)
- VLM 대체 (VLM은 anchor 생성용으로 계속 사용)
- 학습 데이터 추가 (현재 embedding 품질로 충분)

## Consequences

- le-wm이 "embedding 서비스"로 다른 프로젝트에 가치 제공
- rtb-vlm의 VLM 비용을 대폭 절감 (전수 → 샘플링 → le-wm 확장)
- 새 recording 추가 시 auto-labeling까지 수 분 내 완료
- ISO 34502 시나리오 분류는 vehicle-tracker + perception과의 결합 작업 별도 필요
