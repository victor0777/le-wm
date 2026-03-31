# Backlog

## bug
- [x] ~~Normalizer data leak~~ — 수정 완료 (2026-03-28). train.py에서 split 후 train indices로만 fit

## improvement
- [ ] DDP multi-GPU 학습 지원: HDF5 동시 읽기 데드락 해결 (발견일: 2026-03-28)
  - 현재 단일 GPU만 동작. sharded HDF5 또는 메모리 pre-load 필요
- [ ] convert_rtb_to_hdf5.py: 다양성 기반 recording selection 기능 추가 (발견일: 2026-03-28)
  - route/speed/curvature/time-of-day 기준 coverage sampling

## tech-debt
- [ ] `$STABLEWM_HOME` 경로 불일치: README는 `~/.stable-wm/`, 실제는 `~/.stable_worldmodel/` (발견일: 2026-03-28)
  - 현재 symlink으로 우회 중

## idea
- [ ] Surprise score 기반 위험 구간 시각화를 rtb-viewer에 overlay (발견일: 2026-03-28)
- [ ] Cross-recording place recognition으로 map 프로젝트의 loop closure 지원 (발견일: 2026-03-28)
