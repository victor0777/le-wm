#!/usr/bin/env python3
"""
P0-B: Pseudo-maneuver labeling from IMU/GNSS trajectory data.

Derives maneuver labels from future trajectory in HDF5 recordings:
  - left / right: cumulative yaw change over future window
  - straight: low yaw change + moving
  - stop: low speed
  - accel / decel: speed change over window

Saves:
  - Label arrays to ~/.stable_worldmodel/rtb_occany_labels/
  - Statistics + visualizations to outputs/p0b_pseudo_maneuvers/
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
LABEL_DIR = os.path.expanduser("~/.stable_worldmodel/rtb_occany_labels")
OUTPUT_DIR = "outputs/p0b_pseudo_maneuvers"

# Window = 20 frames = 2 seconds at 10 Hz
WINDOW = 20

# Thresholds
YAW_THRESHOLD = 0.15        # rad (~8.6 deg) cumulative yaw for turn
SPEED_STOP = 1.0            # m/s — below this = stop
SPEED_MOVING = 2.0           # m/s — above this for "straight"
SPEED_CHANGE_THRESH = 2.0    # m/s change over window for accel/decel
STOP_MAJORITY = 0.6          # fraction of window that must be < SPEED_STOP

# Label encoding
LABEL_MAP = {
    "left": 0,
    "right": 1,
    "straight": 2,
    "stop": 3,
    "accel": 4,
    "decel": 5,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
LABEL_COLORS = {
    "left": "#e74c3c",
    "right": "#3498db",
    "straight": "#2ecc71",
    "stop": "#95a5a6",
    "accel": "#f39c12",
    "decel": "#9b59b6",
}


def get_short_id(filepath: str) -> str:
    """Extract short recording ID (last 6 chars before .h5)."""
    basename = os.path.basename(filepath)
    return basename.split("_")[-1].replace(".h5", "")


def label_recording(filepath: str, window: int = WINDOW, yaw_threshold: float = YAW_THRESHOLD) -> dict:
    """
    Label each frame with a pseudo-maneuver based on future trajectory.

    Priority (highest to lowest):
      1. stop — if majority of future window has speed < SPEED_STOP
      2. left/right — if |cumulative yaw| > YAW_THRESHOLD
      3. accel/decel — if speed change magnitude > SPEED_CHANGE_THRESH
      4. straight — if speed > SPEED_MOVING
      5. straight (fallback) — everything else
    """
    hf = h5py.File(filepath, "r")
    action = hf["action"][:]      # (N, 3) — [vx, vy, yaw_rate]
    proprio = hf["proprio"][:]    # (N, 8) — [speed, heading_rate, ...]
    ep_len = hf["ep_len"][:]
    ep_offset = hf["ep_offset"][:]
    hf.close()

    N = len(action)
    short_id = get_short_id(filepath)

    speed = proprio[:, 0]             # m/s
    yaw_rate = action[:, 2]           # rad/s (or proprio[:,1])
    dt = 0.1                          # 10 Hz

    labels = np.full(N, LABEL_MAP["straight"], dtype=np.int64)
    cum_yaw = np.full(N, 0.0, dtype=np.float32)
    speed_change = np.full(N, 0.0, dtype=np.float32)

    # Process per episode to avoid cross-episode windows
    for ep_idx in range(len(ep_len)):
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        end = start + length

        for i in range(start, end):
            # Future window end (clamp to episode boundary)
            win_end = min(i + window, end)
            actual_win = win_end - i

            if actual_win < 5:
                # Too few future frames — mark straight
                labels[i] = LABEL_MAP["straight"]
                continue

            # Cumulative yaw change over future window
            future_yaw = yaw_rate[i:win_end]
            c_yaw = np.sum(future_yaw) * dt
            cum_yaw[i] = c_yaw

            # Speed change
            future_speed = speed[i:win_end]
            s_change = future_speed[-1] - future_speed[0]
            speed_change[i] = s_change

            # Stop fraction
            stop_frac = np.mean(future_speed < SPEED_STOP)

            # ── Priority-based labeling ──
            # 1. Stop
            if stop_frac >= STOP_MAJORITY:
                labels[i] = LABEL_MAP["stop"]
            # 2. Turn
            elif c_yaw < -yaw_threshold:
                labels[i] = LABEL_MAP["left"]
            elif c_yaw > yaw_threshold:
                labels[i] = LABEL_MAP["right"]
            # 3. Accel/Decel
            elif s_change > SPEED_CHANGE_THRESH:
                labels[i] = LABEL_MAP["accel"]
            elif s_change < -SPEED_CHANGE_THRESH:
                labels[i] = LABEL_MAP["decel"]
            # 4. Straight (moving)
            elif speed[i] > SPEED_MOVING:
                labels[i] = LABEL_MAP["straight"]
            # 5. Fallback straight
            else:
                labels[i] = LABEL_MAP["straight"]

    return {
        "short_id": short_id,
        "labels": labels,
        "cum_yaw": cum_yaw,
        "speed_change": speed_change,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "action": action,
        "proprio": proprio,
        "ep_len": ep_len,
        "ep_offset": ep_offset,
    }


def compute_stats(result: dict) -> dict:
    """Compute label distribution statistics."""
    labels = result["labels"]
    N = len(labels)
    stats = {"total_frames": N, "distribution": {}}
    for name, idx in LABEL_MAP.items():
        count = int(np.sum(labels == idx))
        stats["distribution"][name] = {
            "count": count,
            "pct": round(100 * count / N, 2),
        }
    return stats


def consistency_analysis(result: dict) -> dict:
    """
    Analyze label consistency: how often does the label change between
    adjacent frames within the same episode?
    """
    labels = result["labels"]
    ep_len = result["ep_len"]
    ep_offset = result["ep_offset"]

    total_transitions = 0
    same_transitions = 0
    diff_transitions = 0

    for ep_idx in range(len(ep_len)):
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        end = start + length

        ep_labels = labels[start:end]
        changes = np.diff(ep_labels) != 0
        total_transitions += len(changes)
        diff_transitions += int(np.sum(changes))
        same_transitions += int(np.sum(~changes))

    # Run-length stats
    all_runs = []
    for ep_idx in range(len(ep_len)):
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        end = start + length
        ep_labels = labels[start:end]
        if len(ep_labels) == 0:
            continue
        run_len = 1
        for j in range(1, len(ep_labels)):
            if ep_labels[j] == ep_labels[j - 1]:
                run_len += 1
            else:
                all_runs.append(run_len)
                run_len = 1
        all_runs.append(run_len)

    all_runs = np.array(all_runs)
    return {
        "total_transitions": total_transitions,
        "same_label": same_transitions,
        "diff_label": diff_transitions,
        "consistency_pct": round(100 * same_transitions / max(total_transitions, 1), 2),
        "mean_run_length": round(float(all_runs.mean()), 2) if len(all_runs) > 0 else 0,
        "median_run_length": round(float(np.median(all_runs)), 2) if len(all_runs) > 0 else 0,
        "max_run_length": int(all_runs.max()) if len(all_runs) > 0 else 0,
        "run_length_std": round(float(all_runs.std()), 2) if len(all_runs) > 0 else 0,
    }


def plot_distribution(all_stats: dict, output_dir: str):
    """Plot label distribution across all recordings."""
    fig, axes = plt.subplots(1, len(all_stats) + 1, figsize=(5 * (len(all_stats) + 1), 5))

    # Per-recording distributions
    for idx, (short_id, stats) in enumerate(all_stats.items()):
        ax = axes[idx]
        names = list(stats["distribution"].keys())
        counts = [stats["distribution"][n]["count"] for n in names]
        colors = [LABEL_COLORS[n] for n in names]
        bars = ax.bar(names, counts, color=colors)
        ax.set_title(f"{short_id}\n(N={stats['total_frames']})")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        for bar, count in zip(bars, counts):
            pct = 100 * count / stats["total_frames"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + stats["total_frames"] * 0.01,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Aggregate
    ax = axes[-1]
    agg = {}
    total = 0
    for stats in all_stats.values():
        total += stats["total_frames"]
        for name, d in stats["distribution"].items():
            agg[name] = agg.get(name, 0) + d["count"]
    names = list(LABEL_MAP.keys())
    counts = [agg.get(n, 0) for n in names]
    colors = [LABEL_COLORS[n] for n in names]
    bars = ax.bar(names, counts, color=colors)
    ax.set_title(f"AGGREGATE\n(N={total})")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Pseudo-Maneuver Label Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/label_distribution.png")


def plot_examples(result: dict, output_dir: str, n_examples: int = 5):
    """
    For each label class, show n_examples with action/proprio time series.
    Each example shows a window of frames around the labeled frame.
    """
    labels = result["labels"]
    speed = result["speed"]
    yaw_rate = result["yaw_rate"]
    cum_yaw = result["cum_yaw"]
    speed_change = result["speed_change"]
    ep_len = result["ep_len"]
    ep_offset = result["ep_offset"]
    short_id = result["short_id"]

    context = 10  # frames before
    forward = WINDOW  # frames after

    fig, axes = plt.subplots(
        len(LABEL_MAP), n_examples, figsize=(4 * n_examples, 3 * len(LABEL_MAP))
    )

    for label_name, label_idx in LABEL_MAP.items():
        candidates = np.where(labels == label_idx)[0]
        if len(candidates) == 0:
            for j in range(n_examples):
                axes[label_idx, j].text(0.5, 0.5, "No examples", ha="center", va="center")
                axes[label_idx, j].set_title(f"{label_name} #{j+1}")
            continue

        # Pick spread-out examples
        step = max(len(candidates) // n_examples, 1)
        selected = candidates[::step][:n_examples]

        for j, frame_idx in enumerate(selected):
            ax = axes[label_idx, j]

            # Find which episode this frame belongs to
            ep_idx = None
            for e in range(len(ep_len)):
                s = int(ep_offset[e])
                if s <= frame_idx < s + int(ep_len[e]):
                    ep_idx = e
                    break

            if ep_idx is None:
                continue

            ep_start = int(ep_offset[ep_idx])
            ep_end = ep_start + int(ep_len[ep_idx])

            # Slice: context before, forward after
            view_start = max(frame_idx - context, ep_start)
            view_end = min(frame_idx + forward, ep_end)
            t = np.arange(view_start, view_end) - frame_idx  # relative time (frames)

            ax2 = ax.twinx()
            ax.plot(t, speed[view_start:view_end], "b-", linewidth=1.5, label="speed")
            ax2.plot(t, yaw_rate[view_start:view_end], "r-", linewidth=1.0, alpha=0.7, label="yaw_rate")
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.axvspan(0, min(forward, view_end - frame_idx), alpha=0.1, color=LABEL_COLORS[label_name])

            if j == 0:
                ax.set_ylabel(f"{label_name}\nspeed (m/s)", color="b", fontsize=9)
            ax2.set_ylabel("yaw_rate", color="r", fontsize=8)
            ax.set_title(
                f"f={frame_idx} cy={cum_yaw[frame_idx]:.2f} ds={speed_change[frame_idx]:.1f}",
                fontsize=8,
            )
            ax.set_xlabel("frames from label", fontsize=8)
            ax.tick_params(labelsize=7)
            ax2.tick_params(labelsize=7)

    fig.suptitle(f"Example Sequences per Label — {short_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"examples_{short_id}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/examples_{short_id}.png")


def plot_consistency(all_consistency: dict, output_dir: str):
    """Plot consistency metrics across recordings."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ids = list(all_consistency.keys())
    cons_pcts = [all_consistency[i]["consistency_pct"] for i in ids]
    mean_runs = [all_consistency[i]["mean_run_length"] for i in ids]
    median_runs = [all_consistency[i]["median_run_length"] for i in ids]

    ax = axes[0]
    bars = ax.bar(ids, cons_pcts, color="#3498db")
    ax.set_title("Label Consistency (% adjacent frames same label)")
    ax.set_ylabel("Consistency %")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, cons_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[1]
    x = np.arange(len(ids))
    w = 0.35
    ax.bar(x - w / 2, mean_runs, w, label="Mean run", color="#2ecc71")
    ax.bar(x + w / 2, median_runs, w, label="Median run", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_title("Run Length Statistics (frames)")
    ax.set_ylabel("Run length (frames)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "consistency_analysis.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/consistency_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="P0-B: Pseudo-maneuver labeling")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--yaw-threshold", type=float, default=YAW_THRESHOLD)
    args = parser.parse_args()

    # Use local variables instead of mutating globals
    window = args.window
    yaw_threshold = args.yaw_threshold

    os.makedirs(args.label_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"Found {len(h5_files)} HDF5 files in {args.data_dir}")
    print(f"Window: {window} frames ({window * 0.1:.1f}s)")
    print(f"Yaw threshold: {yaw_threshold:.3f} rad ({np.degrees(yaw_threshold):.1f} deg)")
    print()

    all_stats = {}
    all_consistency = {}
    all_results = {}

    for filepath in h5_files:
        short_id = get_short_id(filepath)
        print(f"Processing {short_id} ({os.path.basename(filepath)})...")

        # Label
        result = label_recording(filepath, window=window, yaw_threshold=yaw_threshold)
        all_results[short_id] = result

        # Stats
        stats = compute_stats(result)
        all_stats[short_id] = stats
        print(f"  Total frames: {stats['total_frames']}")
        for name, d in stats["distribution"].items():
            print(f"    {name:>8s}: {d['count']:6d} ({d['pct']:5.1f}%)")

        # Consistency
        cons = consistency_analysis(result)
        all_consistency[short_id] = cons
        print(f"  Consistency: {cons['consistency_pct']:.1f}% (mean run={cons['mean_run_length']:.1f}, median={cons['median_run_length']:.1f}, max={cons['max_run_length']})")

        # Save labels
        label_path = os.path.join(args.label_dir, f"{short_id}_labels.npz")
        np.savez(
            label_path,
            labels=result["labels"],
            cum_yaw=result["cum_yaw"],
            speed_change=result["speed_change"],
            label_map=json.dumps(LABEL_MAP),
            window=window,
            yaw_threshold=yaw_threshold,
        )
        print(f"  Saved labels: {label_path}")

        # Plot examples
        plot_examples(result, args.output_dir, n_examples=5)
        print()

    # Aggregate stats
    print("=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    total = sum(s["total_frames"] for s in all_stats.values())
    agg_dist = {}
    for stats in all_stats.values():
        for name, d in stats["distribution"].items():
            agg_dist[name] = agg_dist.get(name, 0) + d["count"]
    print(f"Total frames: {total}")
    for name in LABEL_MAP.keys():
        count = agg_dist.get(name, 0)
        print(f"  {name:>8s}: {count:6d} ({100*count/total:5.1f}%)")

    print()
    print("CONSISTENCY ANALYSIS")
    print("-" * 40)
    for short_id, cons in all_consistency.items():
        print(f"  {short_id}: {cons['consistency_pct']:.1f}% same-label "
              f"(mean_run={cons['mean_run_length']:.1f}, median={cons['median_run_length']:.1f})")

    # Plots
    print()
    print("Generating plots...")
    plot_distribution(all_stats, args.output_dir)
    plot_consistency(all_consistency, args.output_dir)

    # Save full report
    report = {
        "config": {
            "window": window,
            "yaw_threshold": yaw_threshold,
            "speed_stop": SPEED_STOP,
            "speed_moving": SPEED_MOVING,
            "speed_change_thresh": SPEED_CHANGE_THRESH,
            "stop_majority": STOP_MAJORITY,
        },
        "per_recording": {},
        "aggregate": {
            "total_frames": total,
            "distribution": {
                name: {"count": agg_dist.get(name, 0), "pct": round(100 * agg_dist.get(name, 0) / total, 2)}
                for name in LABEL_MAP.keys()
            },
        },
        "consistency": all_consistency,
    }
    for short_id in all_stats:
        report["per_recording"][short_id] = {
            "stats": all_stats[short_id],
            "consistency": all_consistency[short_id],
        }

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
