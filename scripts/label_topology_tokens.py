#!/usr/bin/env python3
"""
Phase 2: Topology token labeling from ego-motion signals.

Derives multi-dimensional driving context tokens from HDF5 recordings:
  a. road_curvature: from rolling 1s yaw_rate history
     - straight (|mean_yaw| < 0.02), gentle_curve (< 0.08), sharp_curve (>= 0.08)
  b. speed_zone: from current speed
     - stopped (< 1 m/s), slow (< 5), medium (< 12), fast (>= 12)
  c. dynamics_state: from speed + acceleration pattern
     - cruising, accelerating, braking, stopped, starting
  d. combined_token: product encoding -> 3 * 4 * 5 = 60 classes

Saves .npz files to ~/.stable_worldmodel/rtb_occany_topology/
"""

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/.stable_worldmodel/rtb_occany")
TOPOLOGY_DIR = os.path.expanduser("~/.stable_worldmodel/rtb_occany_topology")
OUTPUT_DIR = "outputs/topology_tokens"

# Rolling window for curvature: 10 frames = 1 second at 10 Hz
CURVATURE_WINDOW = 10

# Curvature thresholds (on mean |yaw_rate| over window)
CURV_STRAIGHT = 0.02
CURV_GENTLE = 0.08

# Speed zone thresholds (m/s)
SPEED_STOPPED = 1.0
SPEED_SLOW = 5.0
SPEED_MEDIUM = 12.0

# Dynamics: acceleration smoothing window
DYNAMICS_WINDOW = 5  # 0.5s
ACCEL_THRESH = 0.5   # m/s^2
BRAKE_THRESH = -0.5
START_SPEED = 2.0     # must be below this AND accelerating to count as "starting"

# Dimension sizes
N_CURVATURE = 3   # straight=0, gentle_curve=1, sharp_curve=2
N_SPEED = 4       # stopped=0, slow=1, medium=2, fast=3
N_DYNAMICS = 5    # cruising=0, accelerating=1, braking=2, stopped=3, starting=4
N_TOTAL = N_CURVATURE * N_SPEED * N_DYNAMICS  # 60

CURVATURE_NAMES = ["straight", "gentle_curve", "sharp_curve"]
SPEED_NAMES = ["stopped", "slow", "medium", "fast"]
DYNAMICS_NAMES = ["cruising", "accelerating", "braking", "stopped", "starting"]


def get_short_id(filepath: str) -> str:
    basename = os.path.basename(filepath)
    return basename.split("_")[-1].replace(".h5", "")


def compute_curvature(yaw_rate: np.ndarray, ep_len: np.ndarray, ep_offset: np.ndarray,
                      window: int = CURVATURE_WINDOW) -> np.ndarray:
    """Classify road curvature from rolling yaw_rate history."""
    N = len(yaw_rate)
    curvature = np.zeros(N, dtype=np.int64)

    for ep_idx in range(len(ep_len)):
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        end = start + length

        for i in range(start, end):
            win_start = max(i - window + 1, start)
            mean_abs_yaw = np.abs(yaw_rate[win_start:i + 1]).mean()

            if mean_abs_yaw < CURV_STRAIGHT:
                curvature[i] = 0  # straight
            elif mean_abs_yaw < CURV_GENTLE:
                curvature[i] = 1  # gentle_curve
            else:
                curvature[i] = 2  # sharp_curve

    return curvature


def compute_speed_zone(speed: np.ndarray) -> np.ndarray:
    """Classify speed zone."""
    zone = np.zeros(len(speed), dtype=np.int64)
    zone[speed < SPEED_STOPPED] = 0  # stopped
    zone[(speed >= SPEED_STOPPED) & (speed < SPEED_SLOW)] = 1  # slow
    zone[(speed >= SPEED_SLOW) & (speed < SPEED_MEDIUM)] = 2  # medium
    zone[speed >= SPEED_MEDIUM] = 3  # fast
    return zone


def compute_dynamics(speed: np.ndarray, ep_len: np.ndarray, ep_offset: np.ndarray,
                     window: int = DYNAMICS_WINDOW, dt: float = 0.1) -> np.ndarray:
    """Classify dynamics state from speed + acceleration."""
    N = len(speed)
    dynamics = np.zeros(N, dtype=np.int64)

    for ep_idx in range(len(ep_len)):
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        end = start + length

        for i in range(start, end):
            # Compute smoothed acceleration over window
            win_start = max(i - window + 1, start)
            win_speed = speed[win_start:i + 1]

            if len(win_speed) >= 2:
                accel = (win_speed[-1] - win_speed[0]) / (len(win_speed) * dt)
            else:
                accel = 0.0

            cur_speed = speed[i]

            # Priority-based classification
            if cur_speed < SPEED_STOPPED and accel <= ACCEL_THRESH:
                dynamics[i] = 3  # stopped
            elif cur_speed < START_SPEED and accel > ACCEL_THRESH:
                dynamics[i] = 4  # starting (low speed + accelerating)
            elif accel > ACCEL_THRESH:
                dynamics[i] = 1  # accelerating
            elif accel < BRAKE_THRESH:
                dynamics[i] = 2  # braking
            else:
                dynamics[i] = 0  # cruising

    return dynamics


def combine_tokens(curvature: np.ndarray, speed_zone: np.ndarray,
                   dynamics: np.ndarray) -> np.ndarray:
    """Product encoding: token_id = curvature * N_SPEED * N_DYNAMICS + speed * N_DYNAMICS + dynamics."""
    return curvature * N_SPEED * N_DYNAMICS + speed_zone * N_DYNAMICS + dynamics


def label_recording(filepath: str) -> dict:
    """Compute topology tokens for a single recording."""
    hf = h5py.File(filepath, "r")
    action = hf["action"][:]      # (N, 3)
    proprio = hf["proprio"][:]    # (N, 8)
    ep_len = hf["ep_len"][:]
    ep_offset = hf["ep_offset"][:]
    hf.close()

    short_id = get_short_id(filepath)
    speed = proprio[:, 0]
    yaw_rate = action[:, 2]

    curvature = compute_curvature(yaw_rate, ep_len, ep_offset)
    speed_zone = compute_speed_zone(speed)
    dynamics = compute_dynamics(speed, ep_len, ep_offset)
    combined = combine_tokens(curvature, speed_zone, dynamics)

    return {
        "short_id": short_id,
        "curvature": curvature,
        "speed_zone": speed_zone,
        "dynamics": dynamics,
        "combined": combined,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "ep_len": ep_len,
        "ep_offset": ep_offset,
    }


def compute_stats(result: dict) -> dict:
    N = len(result["combined"])
    stats = {
        "total_frames": N,
        "curvature": {},
        "speed_zone": {},
        "dynamics": {},
        "combined_unique": int(len(np.unique(result["combined"]))),
        "combined_top10": [],
    }

    for i, name in enumerate(CURVATURE_NAMES):
        count = int(np.sum(result["curvature"] == i))
        stats["curvature"][name] = {"count": count, "pct": round(100 * count / N, 2)}

    for i, name in enumerate(SPEED_NAMES):
        count = int(np.sum(result["speed_zone"] == i))
        stats["speed_zone"][name] = {"count": count, "pct": round(100 * count / N, 2)}

    for i, name in enumerate(DYNAMICS_NAMES):
        count = int(np.sum(result["dynamics"] == i))
        stats["dynamics"][name] = {"count": count, "pct": round(100 * count / N, 2)}

    # Top-10 combined tokens
    unique, counts = np.unique(result["combined"], return_counts=True)
    order = np.argsort(-counts)[:10]
    for idx in order:
        token_id = int(unique[idx])
        c = token_id // (N_SPEED * N_DYNAMICS)
        s = (token_id % (N_SPEED * N_DYNAMICS)) // N_DYNAMICS
        d = token_id % N_DYNAMICS
        stats["combined_top10"].append({
            "token_id": token_id,
            "curvature": CURVATURE_NAMES[c],
            "speed_zone": SPEED_NAMES[s],
            "dynamics": DYNAMICS_NAMES[d],
            "count": int(counts[idx]),
            "pct": round(100 * counts[idx] / N, 2),
        })

    return stats


def plot_distributions(all_stats: dict, output_dir: str):
    """Plot token distributions across recordings."""
    n_rec = len(all_stats)
    fig, axes = plt.subplots(3, n_rec + 1, figsize=(5 * (n_rec + 1), 12))

    dim_names = ["curvature", "speed_zone", "dynamics"]
    dim_labels = [CURVATURE_NAMES, SPEED_NAMES, DYNAMICS_NAMES]

    for row, (dim, labels) in enumerate(zip(dim_names, dim_labels)):
        # Per-recording
        for col, (short_id, stats) in enumerate(all_stats.items()):
            ax = axes[row, col]
            counts = [stats[dim].get(l, {}).get("count", 0) for l in labels]
            total = stats["total_frames"]
            bars = ax.bar(range(len(labels)), counts, color=plt.cm.Set2(np.arange(len(labels))))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{short_id} - {dim}", fontsize=9)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{100*c/total:.0f}%", ha="center", va="bottom", fontsize=7)

        # Aggregate
        ax = axes[row, -1]
        agg = {}
        total = 0
        for stats in all_stats.values():
            total += stats["total_frames"]
            for l in labels:
                agg[l] = agg.get(l, 0) + stats[dim].get(l, {}).get("count", 0)
        counts = [agg.get(l, 0) for l in labels]
        bars = ax.bar(range(len(labels)), counts, color=plt.cm.Set2(np.arange(len(labels))))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"AGGREGATE - {dim}", fontsize=9)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{100*c/total:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Topology Token Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "topology_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/topology_distribution.png")


def plot_combined_heatmap(all_stats: dict, output_dir: str):
    """Heatmap of combined token usage."""
    # Aggregate combined tokens
    token_counts = np.zeros(N_TOTAL, dtype=np.int64)
    for stats in all_stats.values():
        for entry in stats["combined_top10"]:
            token_counts[entry["token_id"]] += entry["count"]

    # Reshape: curvature x (speed * dynamics)
    mat = token_counts.reshape(N_CURVATURE, N_SPEED * N_DYNAMICS)

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(N_CURVATURE))
    ax.set_yticklabels(CURVATURE_NAMES)

    xlabels = [f"{s[:3]}-{d[:3]}" for s in SPEED_NAMES for d in DYNAMICS_NAMES]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_title("Combined Token Usage (Curvature x Speed-Dynamics)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/combined_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Topology token labeling")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--topology-dir", default=TOPOLOGY_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.topology_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"Found {len(h5_files)} HDF5 files in {args.data_dir}")
    print(f"Curvature thresholds: straight < {CURV_STRAIGHT}, gentle < {CURV_GENTLE}, sharp >= {CURV_GENTLE}")
    print(f"Speed zones: stopped < {SPEED_STOPPED}, slow < {SPEED_SLOW}, medium < {SPEED_MEDIUM}, fast >= {SPEED_MEDIUM}")
    print(f"Total token classes: {N_TOTAL}")
    print()

    all_stats = {}
    all_results = {}

    for filepath in h5_files:
        short_id = get_short_id(filepath)
        print(f"Processing {short_id}...")

        result = label_recording(filepath)
        all_results[short_id] = result

        stats = compute_stats(result)
        all_stats[short_id] = stats

        print(f"  Total frames: {stats['total_frames']}")
        curv_str = ", ".join(f"{n}={stats['curvature'][n]['pct']:.1f}%" for n in CURVATURE_NAMES)
        speed_str = ", ".join(f"{n}={stats['speed_zone'][n]['pct']:.1f}%" for n in SPEED_NAMES)
        dyn_str = ", ".join(f"{n}={stats['dynamics'][n]['pct']:.1f}%" for n in DYNAMICS_NAMES)
        print(f"  Curvature: {curv_str}")
        print(f"  Speed:     {speed_str}")
        print(f"  Dynamics:  {dyn_str}")
        print(f"  Unique combined tokens: {stats['combined_unique']} / {N_TOTAL}")
        print(f"  Top-3 combined:")
        for entry in stats["combined_top10"][:3]:
            print(f"    [{entry['token_id']:2d}] {entry['curvature']}/{entry['speed_zone']}/{entry['dynamics']}: {entry['pct']:.1f}%")

        # Save
        save_path = os.path.join(args.topology_dir, f"{short_id}_topology.npz")
        np.savez(
            save_path,
            curvature=result["curvature"],
            speed_zone=result["speed_zone"],
            dynamics=result["dynamics"],
            combined=result["combined"],
            n_curvature=N_CURVATURE,
            n_speed=N_SPEED,
            n_dynamics=N_DYNAMICS,
            n_total=N_TOTAL,
            curvature_names=json.dumps(CURVATURE_NAMES),
            speed_names=json.dumps(SPEED_NAMES),
            dynamics_names=json.dumps(DYNAMICS_NAMES),
        )
        print(f"  Saved: {save_path}")
        print()

    # Aggregate
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    total = sum(s["total_frames"] for s in all_stats.values())
    all_combined = np.concatenate([r["combined"] for r in all_results.values()])
    unique_all = len(np.unique(all_combined))
    print(f"Total frames: {total}")
    print(f"Unique combined tokens used: {unique_all} / {N_TOTAL}")

    # Cross-recording token overlap
    per_rec_tokens = {sid: set(np.unique(r["combined"]).tolist()) for sid, r in all_results.items()}
    shared = set.intersection(*per_rec_tokens.values()) if per_rec_tokens else set()
    print(f"Tokens shared across ALL recordings: {len(shared)}")

    # Plots
    print("\nGenerating plots...")
    plot_distributions(all_stats, args.output_dir)
    plot_combined_heatmap(all_stats, args.output_dir)

    # Save report
    report = {
        "config": {
            "curvature_window": CURVATURE_WINDOW,
            "curv_straight": CURV_STRAIGHT,
            "curv_gentle": CURV_GENTLE,
            "speed_stopped": SPEED_STOPPED,
            "speed_slow": SPEED_SLOW,
            "speed_medium": SPEED_MEDIUM,
            "dynamics_window": DYNAMICS_WINDOW,
            "accel_thresh": ACCEL_THRESH,
            "brake_thresh": BRAKE_THRESH,
            "n_total_classes": N_TOTAL,
        },
        "per_recording": all_stats,
        "aggregate": {
            "total_frames": total,
            "unique_tokens": unique_all,
            "shared_tokens": len(shared),
            "shared_token_ids": sorted(list(shared)),
        },
    }
    report_path = os.path.join(args.output_dir, "topology_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
