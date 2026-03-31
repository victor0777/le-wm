# ADR-009: Token Regularization + Visual Topology for Cross-Domain Motion

**Date**: 2026-03-31
**Status**: Accepted
**Participants**: Claude vs Codex (gpt-5.3-codex)
**Debate Style**: constructive (3 rounds)
**Confidence**: 0.78 (at least one positive variant)

## Context

Maneuver token conditioning achieves +16.7% cross-domain shuffled gap at epoch 4-7 but declines to +6.5% at epoch 15 (token overfitting/shortcut). Action-derived topology tokens (speed × curvature × dynamics) failed entirely (+2.5%, worse than no-token +11.9%) due to redundancy with action input.

## Decision

Two parallel experiment tracks:

### Experiment R: Token Regularization

| ID | Method | Confidence |
|----|--------|-----------|
| R0 | Maneuver-only baseline (same setup, control) | — |
| R1 | Clip-level maneuver masking p=0.3 | 0.45 |
| R2 | Clip-level maneuver masking p=0.5 | 0.22 |
| R3 | Freeze maneuver embedding at epoch 7 | 0.33 |
| R4 | Concat+proj fusion (hidden=192) instead of additive | 0.60 |

- All on V3 setup (Livlab 4 → da8241 holdout), 15 epochs
- Eval at ep 5/10/15
- Diagnostic at epoch 7: correct+masked_token, shuffled+correct_token
- **Success**: shuffled gap >12% at ep15 AND zeroed gap does not improve (anti-shortcut guard)

### Experiment V: Visual Topology

| ID | Token | Source | Confidence |
|----|-------|--------|-----------|
| V0 | Maneuver-only baseline on rtb_vp | control | — |
| V1 | Binary branching (third_lane + width ratio) | VP lane mask | 0.50 |
| V2 | 3-class visual curvature (lane centerline quadratic) | VP lane mask | 0.34 |

- Train on rtb_vp data directly (avoid OccAny frame mapping)
- V3 setup with rtb_vp recordings
- **Success**: shuffled gap > maneuver-only baseline at ep15

## Debate Summary

### Round 1 (Codex)
- Prioritize regularization first (cheap diagnostic), visual topology as main research direction
- Clip-level token masking > weight decay
- Binary branching token from VP as minimum viable visual topology
- Separate fusion (concat+proj) before cross-attention
- Data diversity wall, not capacity wall

### Round 2 (Claude pushback → Codex adjustment)
- Lane-mask curvature IS orthogonal (visual, not yaw-derived) — agreed to include
- Cross-attention overkill for small vocab — deferred
- VP data available for all recordings, but frame mapping needed
- Concat+proj hidden=192, clip-level masking confirmed

### Round 3 (Codex final)
- R0 control run required for fair comparison
- Anti-shortcut guard: zeroed gap must not improve with shuffled gap
- Matched baseline on rtb_vp before comparing visual tokens
- R4 (concat+proj) highest confidence at 0.60

## Explicitly NOT Doing
- Cross-attention (overkill for tiny vocab)
- Per-frame token masking (leaks through unmasked timesteps)
- Action-derived topology (proven redundant)
- Model scaling (data diversity is the bottleneck)

## Consequences
- If R4 succeeds: architectural fix validated, adopt concat+proj as default
- If V1 succeeds: visual branching is the orthogonal signal we need
- If all fail: data diversity wall confirmed, need more diverse recordings
