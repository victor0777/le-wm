# Lessons Learned

## 2026-03-30: Action-derived tokens are redundant with action embedding
**문제**: Maneuver token (+6.3%) 위에 topology token (curvature/speed/dynamics)을 additive로 추가하면 motion conditioning이 개선될 것으로 기대
**원인**: Topology token의 speed_zone과 dynamics_state는 action (vx, vy, yaw_rate)과 proprio (speed)에서 직접 파생된 정보. 모델에 이미 있는 정보를 discrete token으로 다시 주입하면:
  1. Redundancy: 같은 정보의 두 경로가 생겨 gradient가 분산됨
  2. Information leakage: token이 action의 shortcut이 되어 predictor가 action embedding 대신 token에 의존
  3. Optimization 어려움: 3개 embedding (action + maneuver + topology)의 합이 predictor에 유용한 representation을 형성하기 어려움
**해결**: Token은 action에 없는 정보를 제공해야 함. 유효한 후보:
  - Visual topology (VP lane mask에서 추출한 차선 수, 교차로 타입) -- action과 독립
  - Goal/route token -- 의도를 나타내는 정보
  - Curvature-only token (road shape, action과 약하게 상관)
**교훈**: Conditioning token은 action에 orthogonal한 정보를 제공할 때만 유효. Action-derived feature는 token 대신 action encoder 자체를 개선하는 것이 나음.

## 2026-03-28: DDP multi-GPU training deadlocks with HDF5 dataset
**문제**: 8-GPU 및 5-GPU DDP 학습 시 NCCL 초기화 후 학습이 시작되지 않음 (GPU utilization 100%이나 VRAM ~640MB로 모델 로딩 전 상태)
**원인**: `swm.data.HDF5Dataset`이 각 DDP 프로세스에서 동일 HDF5 파일을 동시에 열고, `num_workers=6`으로 DataLoader가 추가 워커를 생성하여 파일 lock 경합 발생 추정
**해결**: `trainer.devices=1`로 단일 GPU 사용. 14GB VRAM으로 정상 학습 확인
**교훈**: HDF5는 multi-reader에 안전하지 않음. DDP 사용 시 recording을 sharded 파일로 분할하거나, 메모리에 미리 로드 후 학습 필요

## 2026-03-28: stable_worldmodel 캐시 디렉토리가 README와 다름
**문제**: README에는 `$STABLEWM_HOME` (default: `~/.stable-wm/`)로 명시되어 있으나, 실제 `swm.data.utils.get_cache_dir()`은 `~/.stable_worldmodel/` 반환
**원인**: 패키지 내부 기본값과 문서 불일치
**해결**: 심볼릭 링크로 연결 (`ln -sf ~/.stable-wm/rtb ~/.stable_worldmodel/rtb`)
**교훈**: 패키지 설치 후 `swm.data.utils.get_cache_dir()` 확인 필수

## 2026-03-30: Depth supervision이 JEPA world model의 motion conditioning을 결정한다
**문제**: LeWM (JEPA)이 action input을 무시 (AdaLN-zero가 0으로 유지). Feature 추가(Δψ), horizon 확장(num_preds=4) 모두 실패
**원인**: 2D 이미지 embedding만으로는 ego motion → 장면 변화 관계를 학습할 수 없음. 깊이(3D)를 이해해야 "내가 앞으로 가면 앞 물체가 커진다"를 예측 가능
**해결**: Depth prediction을 auxiliary loss로 추가. VP monocular depth (+9.7%) → OccAny 3D reconstruction depth (+20.1%)
**교훈**:
- JEPA 류 world model에서 motion conditioning이 약하면 **feature/horizon이 아닌 3D 이해가 bottleneck**
- Depth supervision 품질이 중요: monocular < 3D reconstruction < LiDAR GT
- Cross-project 데이터(OccAny)를 auxiliary supervision으로 활용하는 패턴이 매우 효과적
