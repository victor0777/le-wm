#!/usr/bin/env python3
"""Anomaly detection threshold analysis using ROC curves.

Binary classification: train (736fcb) = "normal", holdout (8014dd) = "anomalous".
This tests whether surprise scores can distinguish driving on a trained route
vs an unseen route, and analyzes pre-event detection capability.

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/ktl/projects/le-wm \
    python scripts/anomaly_threshold_analysis.py
"""

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

STABLEWM_HOME = Path.home() / ".stable_worldmodel"
OUTPUT_DIR = Path("outputs/anomaly_threshold")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_surprise_scores(path: str = "outputs/anomaly_detection/surprise_scores.npz"):
    d = np.load(path)
    holdout_mse = {}
    holdout_cos = {}
    train_mse = {}
    train_cos = {}
    for h in [1, 3, 5]:
        holdout_mse[h] = d[f"holdout_mse_h{h}"]
        holdout_cos[h] = d[f"holdout_cos_h{h}"]
        train_mse[h] = d[f"train_mse_h{h}"]
        train_cos[h] = d[f"train_cos_h{h}"]
    holdout_frames = d["holdout_frames"]
    train_frames = d["train_frames"]
    return holdout_mse, holdout_cos, train_mse, train_cos, holdout_frames, train_frames


def load_labels(rec_id: str):
    path = STABLEWM_HOME / "rtb_occany_labels" / f"{rec_id}_labels.npz"
    d = np.load(path, allow_pickle=True)
    label_map = json.loads(str(d["label_map"]))
    inv_map = {v: k for k, v in label_map.items()}
    return d["labels"], label_map, inv_map


def load_proprio(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        return f["proprio"][:]


# ---------------------------------------------------------------------------
# 1. ROC / AUC analysis (train=normal, holdout=anomalous)
# ---------------------------------------------------------------------------

def roc_analysis(holdout_mse, train_mse):
    """Compute ROC curves treating train as normal (0) and holdout as anomalous (1)."""
    results = {}
    horizons = sorted(holdout_mse.keys())

    for h in horizons:
        scores = np.concatenate([train_mse[h], holdout_mse[h]])
        labels = np.concatenate([
            np.zeros(len(train_mse[h])),
            np.ones(len(holdout_mse[h])),
        ])

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Youden's J statistic: max(TPR - FPR)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        # Metrics at optimal threshold
        preds = (scores >= best_thresh).astype(int)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Precision-Recall curve
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(pr_recall, pr_precision)

        results[h] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(roc_auc),
            "best_threshold": float(best_thresh),
            "best_j": float(j_scores[best_idx]),
            "best_tpr": float(tpr[best_idx]),
            "best_fpr": float(fpr[best_idx]),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
            "pr_auc": float(pr_auc),
        }

    return results


# ---------------------------------------------------------------------------
# 2. Within-holdout analysis: top 5% vs bottom 5%
# ---------------------------------------------------------------------------

def within_holdout_analysis(
    holdout_mse, holdout_frames, holdout_labels, inv_map, proprio, frameskip=5
):
    """Analyze what distinguishes high vs low surprise frames within holdout."""
    results = {}
    mse = holdout_mse[1]  # use horizon 1
    n = len(mse)

    top5_mask = mse >= np.percentile(mse, 95)
    bot5_mask = mse <= np.percentile(mse, 5)

    # Speed from proprio col 0, mapped via frame indices
    speeds = np.full(n, np.nan)
    for i in range(n):
        fi = holdout_frames[i]
        if fi < len(proprio):
            speeds[i] = proprio[fi, 0]

    # Maneuver labels
    maneuvers = np.full(n, -1, dtype=int)
    for i in range(n):
        fi = holdout_frames[i]
        if fi < len(holdout_labels):
            maneuvers[i] = holdout_labels[fi]

    # Speed comparison
    top5_speed = speeds[top5_mask]
    bot5_speed = speeds[bot5_mask]
    top5_speed = top5_speed[~np.isnan(top5_speed)]
    bot5_speed = bot5_speed[~np.isnan(bot5_speed)]

    results["top5_speed_mean"] = float(np.mean(top5_speed)) if len(top5_speed) > 0 else None
    results["top5_speed_std"] = float(np.std(top5_speed)) if len(top5_speed) > 0 else None
    results["bot5_speed_mean"] = float(np.mean(bot5_speed)) if len(bot5_speed) > 0 else None
    results["bot5_speed_std"] = float(np.std(bot5_speed)) if len(bot5_speed) > 0 else None

    # Maneuver distribution
    top5_man = maneuvers[top5_mask]
    bot5_man = maneuvers[bot5_mask]
    top5_man_dist = {}
    bot5_man_dist = {}
    for lbl_val, lbl_name in inv_map.items():
        top5_man_dist[lbl_name] = int((top5_man == lbl_val).sum())
        bot5_man_dist[lbl_name] = int((bot5_man == lbl_val).sum())
    results["top5_maneuver_dist"] = top5_man_dist
    results["bot5_maneuver_dist"] = bot5_man_dist

    # Temporal position (are high-surprise frames clustered at certain segments?)
    top5_time = holdout_frames[top5_mask] / 10.0  # seconds
    bot5_time = holdout_frames[bot5_mask] / 10.0
    results["top5_time_mean"] = float(np.mean(top5_time))
    results["bot5_time_mean"] = float(np.mean(bot5_time))
    results["top5_time_std"] = float(np.std(top5_time))
    results["bot5_time_std"] = float(np.std(bot5_time))

    # Speed-surprise correlation
    valid = ~np.isnan(speeds)
    if valid.sum() > 10:
        corr = np.corrcoef(speeds[valid], mse[valid])[0, 1]
        results["speed_surprise_correlation"] = float(corr)
    else:
        results["speed_surprise_correlation"] = None

    return results, speeds, maneuvers


# ---------------------------------------------------------------------------
# 3. Pre-event detection: warning time analysis
# ---------------------------------------------------------------------------

def pre_event_detection(holdout_mse, holdout_frames, frameskip=5):
    """For top-20 surprise peaks, compute how early surprise starts rising."""
    mse = holdout_mse[1]
    n = len(mse)
    mean_mse = mse.mean()
    std_mse = mse.std()
    threshold = mean_mse + 1.0 * std_mse  # mean + 1 sigma

    # Find top-20 peaks (with minimum separation of 10 frames to avoid duplicates)
    sorted_idx = np.argsort(mse)[::-1]
    peaks = []
    used = set()
    for idx in sorted_idx:
        if len(peaks) >= 20:
            break
        # Check no neighbor within 10 frames already selected
        if any(abs(idx - p) < 10 for p in used):
            continue
        peaks.append(idx)
        used.add(idx)

    peaks = sorted(peaks)

    warning_times = []  # in frames
    warning_times_sec = []  # in seconds
    peak_details = []

    for peak_idx in peaks:
        peak_val = mse[peak_idx]
        # Walk backward to find where surprise first exceeded threshold
        first_above = peak_idx
        for j in range(peak_idx - 1, -1, -1):
            if mse[j] >= threshold:
                first_above = j
            else:
                break

        lead_frames = peak_idx - first_above
        # Convert to time: each frame index is frameskip original frames at ~10Hz
        lead_time_sec = lead_frames * frameskip / 10.0

        warning_times.append(lead_frames)
        warning_times_sec.append(lead_time_sec)

        peak_details.append({
            "peak_idx": int(peak_idx),
            "peak_frame": int(holdout_frames[peak_idx]),
            "peak_mse": float(peak_val),
            "first_above_idx": int(first_above),
            "lead_frames": int(lead_frames),
            "lead_time_sec": float(lead_time_sec),
        })

    return {
        "threshold": float(threshold),
        "mean_mse": float(mean_mse),
        "std_mse": float(std_mse),
        "warning_times_frames": warning_times,
        "warning_times_sec": warning_times_sec,
        "mean_warning_sec": float(np.mean(warning_times_sec)) if warning_times_sec else 0,
        "median_warning_sec": float(np.median(warning_times_sec)) if warning_times_sec else 0,
        "peaks": peak_details,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_roc_curves(roc_results, output_dir):
    """Plot ROC curves for all horizons."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {1: "#e74c3c", 3: "#f39c12", 5: "#8e44ad"}

    for i, h in enumerate(sorted(roc_results.keys())):
        ax = axes[i]
        r = roc_results[h]
        ax.plot(r["fpr"], r["tpr"], color=colors[h], linewidth=2,
                label=f"AUC = {r['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

        # Mark optimal point
        ax.scatter(r["best_fpr"], r["best_tpr"], color=colors[h],
                   s=100, zorder=5, edgecolors="black", linewidth=1.5)
        ax.annotate(
            f"J={r['best_j']:.3f}\nThresh={r['best_threshold']:.5f}\n"
            f"P={r['precision']:.3f}, R={r['recall']:.3f}\nF1={r['f1']:.3f}",
            xy=(r["best_fpr"], r["best_tpr"]),
            xytext=(r["best_fpr"] + 0.15, r["best_tpr"] - 0.15),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Horizon {h} (h={h})")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "ROC Curves: Train (normal) vs Holdout (anomalous)\n"
        "Binary classification via MSE surprise score",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'roc_curves.png'}")
    plt.close(fig)


def plot_precision_recall(roc_results, output_dir):
    """Plot Precision-Recall curves for all horizons."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {1: "#e74c3c", 3: "#f39c12", 5: "#8e44ad"}

    for i, h in enumerate(sorted(roc_results.keys())):
        ax = axes[i]
        r = roc_results[h]
        ax.plot(r["pr_recall"], r["pr_precision"], color=colors[h], linewidth=2,
                label=f"PR-AUC = {r['pr_auc']:.3f}")

        # Baseline: fraction of positives
        baseline = 350.0 / (350.0 + 730.0)  # holdout / total
        ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                    label=f"Baseline = {baseline:.3f}")

        # Mark operating point at optimal ROC threshold
        ax.scatter(r["recall"], r["precision"], color=colors[h],
                   s=100, zorder=5, edgecolors="black", linewidth=1.5)
        ax.annotate(
            f"P={r['precision']:.3f}\nR={r['recall']:.3f}\nF1={r['f1']:.3f}",
            xy=(r["recall"], r["precision"]),
            xytext=(r["recall"] - 0.2, r["precision"] - 0.1),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Horizon {h}")
        ax.legend(loc="upper right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Precision-Recall Curves: Train (normal) vs Holdout (anomalous)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'precision_recall.png'}")
    plt.close(fig)


def plot_warning_time(pre_event_results, output_dir):
    """Plot histogram of warning times before surprise peaks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wt_sec = pre_event_results["warning_times_sec"]
    wt_frames = pre_event_results["warning_times_frames"]

    # Histogram of warning times in seconds
    ax = axes[0]
    if len(wt_sec) > 0:
        bins = np.arange(0, max(wt_sec) + 1, 0.5)
        if len(bins) < 3:
            bins = 10
        ax.hist(wt_sec, bins=bins, color="#3498db", edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(wt_sec), color="red", linestyle="--",
                   label=f"Mean = {np.mean(wt_sec):.1f}s")
        ax.axvline(np.median(wt_sec), color="orange", linestyle="--",
                   label=f"Median = {np.median(wt_sec):.1f}s")
    ax.set_xlabel("Warning Time (seconds)")
    ax.set_ylabel("Count (top-20 peaks)")
    ax.set_title("Pre-event Detection Lead Time")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bar chart per peak
    ax = axes[1]
    peaks = pre_event_results["peaks"]
    peak_labels = [f"#{i+1}\nf={p['peak_frame']}" for i, p in enumerate(peaks)]
    colors = ["#e74c3c" if p["lead_time_sec"] == 0 else "#2ecc71" for p in peaks]
    ax.barh(range(len(peaks)), [p["lead_time_sec"] for p in peaks],
            color=colors, edgecolor="black", alpha=0.7)
    ax.set_yticks(range(len(peaks)))
    ax.set_yticklabels(peak_labels, fontsize=7)
    ax.set_xlabel("Lead Time (seconds)")
    ax.set_title("Warning Time per Peak (green=early, red=none)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        f"Pre-event Detection: mean+1sigma threshold = {pre_event_results['threshold']:.5f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "warning_time.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'warning_time.png'}")
    plt.close(fig)


def plot_surprise_vs_features(
    holdout_mse, holdout_frames, speeds, maneuvers, inv_map, output_dir, frameskip=5
):
    """Plot surprise correlation with speed, maneuver type, temporal position."""
    mse = holdout_mse[1]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Surprise vs Speed scatter
    ax = fig.add_subplot(gs[0, 0])
    valid = ~np.isnan(speeds)
    if valid.sum() > 0:
        ax.scatter(speeds[valid], mse[valid], alpha=0.3, s=10, c="#3498db")
        # Fit line
        z = np.polyfit(speeds[valid], mse[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(speeds[valid].min(), speeds[valid].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)
        corr = np.corrcoef(speeds[valid], mse[valid])[0, 1]
        ax.set_title(f"MSE vs Speed (r={corr:.3f})")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("MSE Surprise (h=1)")
    ax.grid(alpha=0.3)

    # 2. Speed binned surprise
    ax = fig.add_subplot(gs[0, 1])
    valid = ~np.isnan(speeds)
    if valid.sum() > 0:
        spd = speeds[valid]
        m = mse[valid]
        bins = np.percentile(spd, [0, 20, 40, 60, 80, 100])
        bins = np.unique(bins)
        bin_labels = []
        bin_means = []
        bin_stds = []
        for j in range(len(bins) - 1):
            mask = (spd >= bins[j]) & (spd < bins[j + 1])
            if j == len(bins) - 2:
                mask = (spd >= bins[j]) & (spd <= bins[j + 1])
            if mask.sum() > 0:
                bin_labels.append(f"{bins[j]:.1f}-{bins[j+1]:.1f}")
                bin_means.append(m[mask].mean())
                bin_stds.append(m[mask].std())
        ax.bar(range(len(bin_labels)), bin_means, yerr=bin_stds,
               color="#2ecc71", edgecolor="black", alpha=0.7, capsize=4)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=20)
    ax.set_xlabel("Speed Bin (m/s)")
    ax.set_ylabel("Mean MSE Surprise")
    ax.set_title("Surprise by Speed Quintile")
    ax.grid(alpha=0.3)

    # 3. Surprise by maneuver (violin-like box)
    ax = fig.add_subplot(gs[0, 2])
    maneuver_colors = {
        "left": "#3498db", "right": "#e67e22", "straight": "#2ecc71",
        "stop": "#95a5a6", "accel": "#e74c3c", "decel": "#9b59b6",
    }
    data_by_man = {}
    for lbl_val, lbl_name in inv_map.items():
        mask = maneuvers == lbl_val
        if mask.sum() > 3:
            data_by_man[lbl_name] = mse[mask]
    if data_by_man:
        bp = ax.boxplot(
            [data_by_man[k] for k in data_by_man],
            tick_labels=list(data_by_man.keys()),
            patch_artist=True, showfliers=False,
        )
        for patch, name in zip(bp["boxes"], data_by_man.keys()):
            patch.set_facecolor(maneuver_colors.get(name, "#cccccc"))
            patch.set_alpha(0.7)
    ax.set_ylabel("MSE Surprise")
    ax.set_title("Surprise by Maneuver")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)

    # 4. Temporal: surprise + speed on dual y-axis
    ax1 = fig.add_subplot(gs[1, :2])
    time_sec = holdout_frames[:len(mse)] / 10.0
    ax1.plot(time_sec, mse, color="#e74c3c", alpha=0.6, linewidth=0.8, label="MSE Surprise")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("MSE Surprise (h=1)", color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    ax2 = ax1.twinx()
    valid = ~np.isnan(speeds)
    ax2.plot(time_sec[valid], speeds[valid], color="#3498db", alpha=0.5, linewidth=0.8,
             label="Speed")
    ax2.set_ylabel("Speed (m/s)", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax1.set_title("Surprise + Speed over Time (Holdout)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # 5. Maneuver proportion: top 5% vs bottom 5% vs overall
    ax = fig.add_subplot(gs[1, 2])
    top5_mask = mse >= np.percentile(mse, 95)
    bot5_mask = mse <= np.percentile(mse, 5)
    all_names = sorted(inv_map.values())
    top5_pcts = []
    bot5_pcts = []
    all_pcts = []
    for name in all_names:
        lbl_val = [k for k, v in inv_map.items() if v == name][0]
        top5_pcts.append((maneuvers[top5_mask] == lbl_val).mean() * 100 if top5_mask.sum() > 0 else 0)
        bot5_pcts.append((maneuvers[bot5_mask] == lbl_val).mean() * 100 if bot5_mask.sum() > 0 else 0)
        all_pcts.append((maneuvers == lbl_val).mean() * 100)

    x = np.arange(len(all_names))
    w = 0.25
    ax.bar(x - w, top5_pcts, w, label="Top 5% surprise", color="#e74c3c", alpha=0.7)
    ax.bar(x, all_pcts, w, label="Overall", color="#95a5a6", alpha=0.7)
    ax.bar(x + w, bot5_pcts, w, label="Bottom 5% surprise", color="#2ecc71", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=8, rotation=30)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Maneuver Distribution:\nHigh vs Low Surprise")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Surprise Score Feature Analysis (Holdout: 8014dd)", fontsize=14, y=1.01)
    fig.savefig(output_dir / "surprise_vs_features.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'surprise_vs_features.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANOMALY THRESHOLD ANALYSIS (ROC + Pre-event Detection)")
    print("=" * 70)

    # Load data
    print("\nLoading surprise scores...")
    (holdout_mse, holdout_cos, train_mse, train_cos,
     holdout_frames, train_frames) = load_surprise_scores()

    print(f"  Holdout frames: {len(holdout_mse[1])}")
    print(f"  Train frames: {len(train_mse[1])}")

    print("\nLoading maneuver labels...")
    holdout_labels, label_map, inv_map = load_labels("8014dd")
    train_labels, _, _ = load_labels("736fcb")

    print("\nLoading proprio data...")
    holdout_proprio = load_proprio(
        str(STABLEWM_HOME / "rtb_occany" / "Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5")
    )
    train_proprio = load_proprio(
        str(STABLEWM_HOME / "rtb_occany" / "Livlab-Rt-C-5_JT_2025-09-22_06-57-30_2111_736fcb.h5")
    )

    # ===================================================================
    # 1. ROC Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("1. ROC ANALYSIS (Binary: Train=Normal vs Holdout=Anomalous)")
    print("=" * 70)

    roc_results = roc_analysis(holdout_mse, train_mse)

    for h in sorted(roc_results.keys()):
        r = roc_results[h]
        print(f"\n  Horizon {h}:")
        print(f"    AUC-ROC        = {r['auc']:.4f}")
        print(f"    PR-AUC         = {r['pr_auc']:.4f}")
        print(f"    Optimal Thresh = {r['best_threshold']:.6f} (Youden J={r['best_j']:.4f})")
        print(f"    TPR @ optimal  = {r['best_tpr']:.4f}")
        print(f"    FPR @ optimal  = {r['best_fpr']:.4f}")
        print(f"    Precision      = {r['precision']:.4f}")
        print(f"    Recall         = {r['recall']:.4f}")
        print(f"    F1             = {r['f1']:.4f}")

    # Also try cosine surprise
    print("\n  --- Cosine Surprise ROC ---")
    roc_cos = roc_analysis(holdout_cos, train_cos)
    for h in sorted(roc_cos.keys()):
        r = roc_cos[h]
        print(f"    h={h}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}, "
              f"Thresh={r['best_threshold']:.6f}")

    # ===================================================================
    # 2. Within-holdout analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("2. WITHIN-HOLDOUT ANALYSIS (Top 5% vs Bottom 5%)")
    print("=" * 70)

    within_results, speeds, maneuvers = within_holdout_analysis(
        holdout_mse, holdout_frames, holdout_labels, inv_map, holdout_proprio
    )

    print(f"\n  Speed (m/s):")
    print(f"    Top 5% surprise: mean={within_results['top5_speed_mean']:.3f}, "
          f"std={within_results['top5_speed_std']:.3f}")
    print(f"    Bot 5% surprise: mean={within_results['bot5_speed_mean']:.3f}, "
          f"std={within_results['bot5_speed_std']:.3f}")

    print(f"\n  Speed-Surprise Correlation: r={within_results['speed_surprise_correlation']:.4f}")

    print(f"\n  Temporal Position (seconds):")
    print(f"    Top 5%: mean={within_results['top5_time_mean']:.1f}s, "
          f"std={within_results['top5_time_std']:.1f}s")
    print(f"    Bot 5%: mean={within_results['bot5_time_mean']:.1f}s, "
          f"std={within_results['bot5_time_std']:.1f}s")

    print(f"\n  Maneuver Distribution (Top 5% / Bot 5%):")
    for name in sorted(inv_map.values()):
        t = within_results["top5_maneuver_dist"].get(name, 0)
        b = within_results["bot5_maneuver_dist"].get(name, 0)
        print(f"    {name:10s}: top5={t:3d}, bot5={b:3d}")

    # ===================================================================
    # 3. Pre-event detection
    # ===================================================================
    print("\n" + "=" * 70)
    print("3. PRE-EVENT DETECTION (Warning Time Analysis)")
    print("=" * 70)

    pre_event = pre_event_detection(holdout_mse, holdout_frames)

    print(f"\n  Threshold: mean + 1 sigma = {pre_event['threshold']:.6f}")
    print(f"    (mean = {pre_event['mean_mse']:.6f}, std = {pre_event['std_mse']:.6f})")
    print(f"\n  Warning times for top-20 peaks:")
    print(f"    Mean  = {pre_event['mean_warning_sec']:.2f} seconds")
    print(f"    Median = {pre_event['median_warning_sec']:.2f} seconds")

    print(f"\n  Per-peak details:")
    for i, p in enumerate(pre_event["peaks"]):
        print(f"    Peak #{i+1:2d}: frame={p['peak_frame']:5d}, "
              f"MSE={p['peak_mse']:.6f}, "
              f"lead={p['lead_time_sec']:.1f}s ({p['lead_frames']} frames)")

    # Count peaks with >0 lead time
    n_with_lead = sum(1 for p in pre_event["peaks"] if p["lead_frames"] > 0)
    print(f"\n  Peaks with early warning (lead > 0): {n_with_lead}/{len(pre_event['peaks'])}")

    # ===================================================================
    # 4. Create visualizations
    # ===================================================================
    print("\n" + "=" * 70)
    print("4. CREATING VISUALIZATIONS")
    print("=" * 70)

    plot_roc_curves(roc_results, output_dir)
    plot_precision_recall(roc_results, output_dir)
    plot_warning_time(pre_event, output_dir)
    plot_surprise_vs_features(
        holdout_mse, holdout_frames, speeds, maneuvers, inv_map, output_dir
    )

    # ===================================================================
    # 5. Save report
    # ===================================================================
    report = {
        "description": "Anomaly detection threshold analysis via ROC curves. "
                       "Train (736fcb) = normal, Holdout (8014dd) = anomalous.",
        "note": "This is route deviation detection, not true anomaly detection. "
                "Holdout is normal driving on a different route.",
        "roc_mse": {},
        "roc_cosine": {},
        "within_holdout": within_results,
        "pre_event_detection": {
            "threshold": pre_event["threshold"],
            "mean_warning_sec": pre_event["mean_warning_sec"],
            "median_warning_sec": pre_event["median_warning_sec"],
            "peaks_with_lead": n_with_lead,
            "total_peaks": len(pre_event["peaks"]),
            "peak_details": pre_event["peaks"],
        },
    }

    for h in sorted(roc_results.keys()):
        r = roc_results[h]
        report["roc_mse"][f"h{h}"] = {
            "auc": r["auc"],
            "pr_auc": r["pr_auc"],
            "optimal_threshold": r["best_threshold"],
            "youden_j": r["best_j"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "tpr_at_optimal": r["best_tpr"],
            "fpr_at_optimal": r["best_fpr"],
        }
        rc = roc_cos[h]
        report["roc_cosine"][f"h{h}"] = {
            "auc": rc["auc"],
            "pr_auc": rc["pr_auc"],
            "optimal_threshold": rc["best_threshold"],
            "f1": rc["f1"],
        }

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
