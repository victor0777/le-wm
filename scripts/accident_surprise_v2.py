#!/usr/bin/env python3
"""Accident Surprise Analysis v2 — using A1 model.

Applies the JEPA world model surprise score to dashcam accident videos to test
whether surprise rises BEFORE the collision moment.

Two complementary metrics:
  1. Prediction surprise: ||predicted_emb - actual_emb||^2  (zero-action prediction)
  2. Embedding change rate: ||emb(t+1) - emb(t)||^2  (no model prediction needed)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ktl/projects/le-wm \
        python scripts/accident_surprise_v2.py
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from matplotlib.gridspec import GridSpec

# Import VP decoders so torch.load can unpickle them
try:
    from train_vp import LaneDecoder, DepthDecoder  # noqa: F401
except ImportError:
    pass

STABLEWM_HOME = Path.home() / ".stable_worldmodel"
ACCIDENT_DATA = Path("/data2/accident_data")
REVIEW_RESULTS = Path("/home/ktl/projects/accident_analysis/review_results.json")


def load_model(ckpt_path: str, device: str = "cuda"):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def get_img_transform(img_size: int = 224):
    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
    resize = spt.data.transforms.Resize(img_size, source="pixels", target="pixels")
    return spt.data.transforms.Compose(to_image, resize)


def extract_frames(video_path: Path, fps_target: int = 10, img_size: int = 224):
    """Extract frames from mp4 at target fps. Returns (N, 3, H, W) uint8 and original fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        cap.release()
        return None, 0

    skip = max(1, round(orig_fps / fps_target))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.transpose(2, 0, 1))  # (3, H, W)
        frame_idx += 1

    cap.release()
    if not frames:
        return None, orig_fps

    return np.stack(frames, axis=0), orig_fps


def encode_frames(model, frames: np.ndarray, transform, device: str,
                  batch_size: int = 32) -> torch.Tensor:
    """Encode all frames to embeddings. Returns (N, D) tensor on CPU."""
    batch_data = {"pixels": torch.from_numpy(frames)}
    batch_data = transform(batch_data)
    pixels = batch_data["pixels"]  # (N, 3, 224, 224)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(pixels), batch_size):
            px = pixels[i:i + batch_size].to(device)
            info = {"pixels": px.unsqueeze(1)}  # (B, 1, 3, H, W)
            output = model.encode(info)
            emb = output["emb"].squeeze(1).cpu()  # (B, D)
            embeddings.append(emb)

    return torch.cat(embeddings, dim=0)  # (N, D)


def compute_prediction_surprise(model, embeddings: torch.Tensor, device: str,
                                history_size: int = 3) -> np.ndarray:
    """Compute per-frame prediction surprise with zero action.
    Returns array of length N with NaN for first history_size frames.
    """
    N = len(embeddings)
    embed_dim = embeddings.shape[1]
    surprise = np.full(N, np.nan)

    with torch.no_grad():
        for t in range(history_size, N):
            ctx = embeddings[t - history_size:t].unsqueeze(0).to(device)  # (1, hs, D)
            zero_act = torch.zeros(1, history_size, embed_dim, device=device)
            pred = model.predict(ctx, zero_act)  # (1, hs, D)
            pred_last = pred[0, -1]
            actual = embeddings[t].to(device)
            surprise[t] = (pred_last - actual).pow(2).sum().item()

    return surprise


def compute_embedding_change_rate(embeddings: torch.Tensor) -> np.ndarray:
    """Compute ||emb(t+1) - emb(t)||^2. Returns array of length N with NaN at index 0."""
    diffs = (embeddings[1:] - embeddings[:-1]).pow(2).sum(dim=-1).numpy()
    return np.concatenate([[np.nan], diffs])


def analyze_video(model, video_path: Path, label_info: dict, transform, device: str,
                  fps_target: int = 10) -> dict | None:
    """Analyze a single video. Returns dict with both metrics and metadata."""
    frames, orig_fps = extract_frames(video_path, fps_target=fps_target)
    if frames is None or len(frames) < 10:
        return None

    embeddings = encode_frames(model, frames, transform, device)
    pred_surprise = compute_prediction_surprise(model, embeddings, device)
    change_rate = compute_embedding_change_rate(embeddings)

    result = {
        "video": video_path.stem,
        "label": label_info["label"],
        "num_frames": len(frames),
        "duration": len(frames) / fps_target,
        "pred_surprise": pred_surprise,
        "change_rate": change_rate,
        "fps_target": fps_target,
    }

    if label_info.get("collisions"):
        collision_times = [c["time_sec"] for c in label_info["collisions"] if "time_sec" in c]
        result["collision_times"] = collision_times
        if label_info["collisions"][0].get("collision_type"):
            result["collision_type"] = label_info["collisions"][0]["collision_type"]

    return result


def compute_pre_crash_stats(results: list, metric_key: str = "pred_surprise"):
    """Compute pre-crash surprise elevation for results with collision times."""
    ratios = []
    elevations = []  # (video_name, ratio, pre_mean, early_mean)

    for r in results:
        if "collision_times" not in r or not r["collision_times"]:
            continue
        t_col = r["collision_times"][0]
        fps = r["fps_target"]
        s = r[metric_key]
        times = np.arange(len(s)) / fps

        # Pre-crash: last 2 seconds before collision
        pre_2s = s[(times >= t_col - 2) & (times < t_col) & ~np.isnan(s)]
        # Early: from 1s into video to 3s before collision
        early = s[(times >= 1) & (times < t_col - 3) & ~np.isnan(s)]

        if len(pre_2s) > 2 and len(early) > 5:
            pre_mean = float(np.mean(pre_2s))
            early_mean = float(np.mean(early))
            ratio = pre_mean / max(early_mean, 1e-8)
            ratios.append(ratio)
            elevations.append((r["video"], ratio, pre_mean, early_mean))

    return ratios, elevations


def plot_aligned_surprise(results: list, metric_key: str, ax, title: str,
                          pre_window: float = 5.0, post_window: float = 3.0):
    """Plot surprise curves aligned to collision time."""
    aligned = []
    for r in results:
        if "collision_times" not in r or not r["collision_times"]:
            continue
        t_col = r["collision_times"][0]
        fps = r["fps_target"]
        s = r[metric_key]
        times = np.arange(len(s)) / fps
        rel_times = times - t_col
        mask = (rel_times >= -pre_window) & (rel_times <= post_window)
        if mask.sum() < 5:
            continue
        aligned.append({"rel_time": rel_times[mask], "surprise": s[mask]})

    if not aligned:
        ax.set_title(f"{title} (no data)")
        return 0

    # Individual curves
    for a in aligned[:40]:
        ax.plot(a["rel_time"], a["surprise"], alpha=0.15, color="red", linewidth=0.6)

    # Average curve
    bins = np.linspace(-pre_window, post_window, 80)
    avg = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1)
    for a in aligned:
        for j in range(len(bins) - 1):
            m = (a["rel_time"] >= bins[j]) & (a["rel_time"] < bins[j + 1])
            vals = a["surprise"][m]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                avg[j] += vals.mean()
                counts[j] += 1
    avg = np.where(counts > 0, avg / counts, np.nan)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(centers, avg, color="darkred", linewidth=3, label="Average")
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Collision")
    ax.set_xlabel("Time relative to collision (s)")
    ax.set_ylabel("Score")
    ax.set_title(f"{title} ({len(aligned)} videos)")
    ax.legend(fontsize=8)

    return len(aligned)


def plot_individual_timelines(results: list, output_dir: Path, n_examples: int = 12):
    """Save individual timeline plots for the most interesting accident videos."""
    # Select videos that have collision times and reasonable data
    candidates = [r for r in results
                  if "collision_times" in r and r["collision_times"]
                  and r["duration"] > 5]

    if not candidates:
        return

    # Sort by pre-crash surprise ratio (most elevated first)
    ratios_map = {}
    for r in candidates:
        t_col = r["collision_times"][0]
        fps = r["fps_target"]
        s = r["pred_surprise"]
        times = np.arange(len(s)) / fps
        pre = s[(times >= t_col - 2) & (times < t_col) & ~np.isnan(s)]
        early = s[(times >= 1) & (times < t_col - 3) & ~np.isnan(s)]
        if len(pre) > 2 and len(early) > 5:
            ratios_map[r["video"]] = np.mean(pre) / max(np.mean(early), 1e-8)

    # Pick top elevated + some low ones for comparison
    sorted_candidates = sorted(candidates, key=lambda r: ratios_map.get(r["video"], 0), reverse=True)
    selected = sorted_candidates[:n_examples // 2]  # top elevated
    if len(sorted_candidates) > n_examples:
        selected += sorted_candidates[-(n_examples // 2):]  # least elevated
    else:
        selected = sorted_candidates[:n_examples]

    ind_dir = output_dir / "individual_timelines"
    ind_dir.mkdir(parents=True, exist_ok=True)

    for r in selected:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        times = np.arange(r["num_frames"]) / r["fps_target"]

        # Prediction surprise
        ax1.plot(times, r["pred_surprise"], color="steelblue", linewidth=1, label="Pred. surprise")
        # Smoothed
        valid = ~np.isnan(r["pred_surprise"])
        if valid.sum() > 10:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(np.nan_to_num(r["pred_surprise"], 0), kernel, mode="same")
            ax1.plot(times, smoothed, color="darkblue", linewidth=2, alpha=0.7, label="Smoothed (5pt)")
        ax1.set_ylabel("Prediction Surprise")
        ax1.legend(fontsize=8)

        # Embedding change rate
        ax2.plot(times, r["change_rate"], color="darkorange", linewidth=1, label="Emb. change rate")
        if valid.sum() > 10:
            smoothed_cr = np.convolve(np.nan_to_num(r["change_rate"], 0), kernel, mode="same")
            ax2.plot(times, smoothed_cr, color="red", linewidth=2, alpha=0.7, label="Smoothed (5pt)")
        ax2.set_ylabel("Embedding Change Rate")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=8)

        # Mark collisions
        for ct in r.get("collision_times", []):
            for ax in (ax1, ax2):
                ax.axvline(ct, color="red", linestyle="--", linewidth=2)

        ratio_str = f"ratio={ratios_map.get(r['video'], 0):.2f}" if r["video"] in ratios_map else ""
        ctype = r.get("collision_type", "unknown")
        fig.suptitle(f"{r['video'][:50]}  |  {ctype}  |  {ratio_str}", fontsize=11)
        fig.tight_layout()
        fig.savefig(ind_dir / f"{r['video'][:60]}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(selected)} individual timelines to {ind_dir}")


def main():
    parser = argparse.ArgumentParser(description="Accident Surprise Analysis v2 (A1 model)")
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--max-videos", type=int, default=100,
                        help="Max accident videos to process")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/accident_surprise_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    with open(REVIEW_RESULTS) as f:
        labels = json.load(f)

    # Collect videos
    accident_videos = []
    non_accident_videos = []
    for vid_name, info in labels.items():
        mp4_name = vid_name if vid_name.endswith(".mp4") else vid_name + ".mp4"
        mp4_path = ACCIDENT_DATA / mp4_name
        if not mp4_path.exists():
            continue
        vid_label = info.get("label", "unknown")
        if vid_label == "accident":
            accident_videos.append((mp4_path, info))
        elif vid_label == "non_accident":
            non_accident_videos.append((mp4_path, info))

    print(f"Found: {len(accident_videos)} accident, {len(non_accident_videos)} non_accident videos")

    # Sample
    random.shuffle(accident_videos)
    random.shuffle(non_accident_videos)
    max_acc = min(args.max_videos, len(accident_videos))
    max_nonacc = min(args.max_videos // 4, len(non_accident_videos))
    accident_videos = accident_videos[:max_acc]
    non_accident_videos = non_accident_videos[:max_nonacc]
    print(f"Processing: {len(accident_videos)} accident, {len(non_accident_videos)} non_accident")

    # Load model
    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Process videos
    accident_results = []
    non_accident_results = []

    for i, (path, info) in enumerate(accident_videos):
        print(f"  Accident [{i + 1}/{len(accident_videos)}]: {path.stem}")
        result = analyze_video(model, path, info, transform, args.device, args.fps)
        if result:
            accident_results.append(result)

    for i, (path, info) in enumerate(non_accident_videos):
        print(f"  Non-accident [{i + 1}/{len(non_accident_videos)}]: {path.stem}")
        result = analyze_video(model, path, info, transform, args.device, args.fps)
        if result:
            non_accident_results.append(result)

    print(f"\nProcessed: {len(accident_results)} accident, {len(non_accident_results)} non_accident")

    # =========================================================================
    # Analysis
    # =========================================================================

    # Pre-crash stats for both metrics
    pred_ratios, pred_elev = compute_pre_crash_stats(accident_results, "pred_surprise")
    cr_ratios, cr_elev = compute_pre_crash_stats(accident_results, "change_rate")

    # Mean surprise per video
    acc_pred_means = [float(np.nanmean(r["pred_surprise"])) for r in accident_results]
    nonacc_pred_means = [float(np.nanmean(r["pred_surprise"])) for r in non_accident_results]
    acc_cr_means = [float(np.nanmean(r["change_rate"])) for r in accident_results]
    nonacc_cr_means = [float(np.nanmean(r["change_rate"])) for r in non_accident_results]

    # Temporal window analysis: surprise in different time windows before collision
    window_analysis = {}
    for window_sec in [1, 2, 3, 5]:
        window_pred = []
        window_cr = []
        for r in accident_results:
            if "collision_times" not in r or not r["collision_times"]:
                continue
            t_col = r["collision_times"][0]
            fps = r["fps_target"]
            times = np.arange(r["num_frames"]) / fps

            ps = r["pred_surprise"]
            cr = r["change_rate"]

            pre_mask = (times >= t_col - window_sec) & (times < t_col) & ~np.isnan(ps)
            early_mask = (times >= 1) & (times < t_col - 5) & ~np.isnan(ps)

            if pre_mask.sum() > 1 and early_mask.sum() > 3:
                window_pred.append(float(np.mean(ps[pre_mask]) / max(np.mean(ps[early_mask]), 1e-8)))
                window_cr.append(float(np.mean(cr[pre_mask & ~np.isnan(cr)]) /
                                       max(np.mean(cr[early_mask & ~np.isnan(cr)]), 1e-8)))

        window_analysis[window_sec] = {
            "pred_ratio_median": float(np.median(window_pred)) if window_pred else None,
            "pred_ratio_mean": float(np.mean(window_pred)) if window_pred else None,
            "pred_pct_above_1": float(np.mean(np.array(window_pred) > 1.0) * 100) if window_pred else None,
            "cr_ratio_median": float(np.median(window_cr)) if window_cr else None,
            "cr_ratio_mean": float(np.mean(window_cr)) if window_cr else None,
            "cr_pct_above_1": float(np.mean(np.array(window_cr) > 1.0) * 100) if window_cr else None,
            "n_videos": len(window_pred),
        }

    # =========================================================================
    # Plotting
    # =========================================================================
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Row 0: Aligned surprise curves (both metrics)
    ax0a = fig.add_subplot(gs[0, :2])
    n_aligned = plot_aligned_surprise(accident_results, "pred_surprise", ax0a,
                                      "Prediction Surprise (zero-action) aligned to collision")

    ax0b = fig.add_subplot(gs[0, 2])
    plot_aligned_surprise(accident_results, "change_rate", ax0b,
                          "Embedding Change Rate aligned to collision")

    # Row 1: Pre-crash ratio histograms
    ax1a = fig.add_subplot(gs[1, 0])
    if pred_ratios:
        ax1a.hist(pred_ratios, bins=20, color="coral", edgecolor="white")
        ax1a.axvline(1.0, color="black", linestyle="--", label="No change")
        med = np.median(pred_ratios)
        ax1a.axvline(med, color="red", linewidth=2, label=f"Median: {med:.2f}")
        pct = np.mean(np.array(pred_ratios) > 1.0) * 100
        ax1a.set_title(f"Pred. Surprise Ratio\n(last 2s / earlier, {pct:.0f}% > 1.0)")
        ax1a.legend(fontsize=8)
    ax1a.set_xlabel("Surprise Ratio")

    ax1b = fig.add_subplot(gs[1, 1])
    if cr_ratios:
        ax1b.hist(cr_ratios, bins=20, color="lightskyblue", edgecolor="white")
        ax1b.axvline(1.0, color="black", linestyle="--", label="No change")
        med = np.median(cr_ratios)
        ax1b.axvline(med, color="blue", linewidth=2, label=f"Median: {med:.2f}")
        pct = np.mean(np.array(cr_ratios) > 1.0) * 100
        ax1b.set_title(f"Change Rate Ratio\n(last 2s / earlier, {pct:.0f}% > 1.0)")
        ax1b.legend(fontsize=8)
    ax1b.set_xlabel("Change Rate Ratio")

    # Time-window analysis
    ax1c = fig.add_subplot(gs[1, 2])
    windows = sorted(window_analysis.keys())
    pred_meds = [window_analysis[w]["pred_ratio_median"] or 0 for w in windows]
    cr_meds = [window_analysis[w]["cr_ratio_median"] or 0 for w in windows]
    x = np.arange(len(windows))
    ax1c.bar(x - 0.15, pred_meds, 0.3, label="Pred. surprise", color="coral", alpha=0.8)
    ax1c.bar(x + 0.15, cr_meds, 0.3, label="Change rate", color="lightskyblue", alpha=0.8)
    ax1c.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1c.set_xticks(x)
    ax1c.set_xticklabels([f"{w}s" for w in windows])
    ax1c.set_xlabel("Window before collision")
    ax1c.set_ylabel("Median ratio (pre/early)")
    ax1c.set_title("Surprise Elevation by Time Window")
    ax1c.legend(fontsize=8)

    # Row 2: Accident vs Non-accident distributions
    ax2a = fig.add_subplot(gs[2, 0])
    if acc_pred_means:
        ax2a.hist(acc_pred_means, bins=20, alpha=0.6, color="red",
                  label=f"Accident (n={len(accident_results)})", density=True)
    if nonacc_pred_means:
        ax2a.hist(nonacc_pred_means, bins=20, alpha=0.6, color="green",
                  label=f"Non-accident (n={len(non_accident_results)})", density=True)
    ax2a.set_xlabel("Mean Pred. Surprise per Video")
    ax2a.set_title("Video-level Prediction Surprise")
    ax2a.legend(fontsize=8)

    ax2b = fig.add_subplot(gs[2, 1])
    if acc_cr_means:
        ax2b.hist(acc_cr_means, bins=20, alpha=0.6, color="red",
                  label=f"Accident (n={len(accident_results)})", density=True)
    if nonacc_cr_means:
        ax2b.hist(nonacc_cr_means, bins=20, alpha=0.6, color="green",
                  label=f"Non-accident (n={len(non_accident_results)})", density=True)
    ax2b.set_xlabel("Mean Change Rate per Video")
    ax2b.set_title("Video-level Embedding Change Rate")
    ax2b.legend(fontsize=8)

    # Collision type comparison
    ax2c = fig.add_subplot(gs[2, 2])
    type_surprises = {}
    for r in accident_results:
        ct = r.get("collision_type", "unknown")
        type_surprises.setdefault(ct, []).append(float(np.nanmean(r["pred_surprise"])))
    if type_surprises:
        types = sorted(type_surprises.keys(), key=lambda k: -len(type_surprises[k]))[:6]
        means = [np.mean(type_surprises[t]) for t in types]
        counts = [len(type_surprises[t]) for t in types]
        ax2c.bar(range(len(types)), means, color="salmon", edgecolor="white")
        ax2c.set_xticks(range(len(types)))
        ax2c.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=7, rotation=15)
        ax2c.set_ylabel("Mean Pred. Surprise")
        ax2c.set_title("Surprise by Collision Type")

    # Row 3: Summary
    ax3 = fig.add_subplot(gs[3, :])
    ax3.axis("off")

    summary_lines = [
        f"Model: expA1 (OccAny depth-supervised, Livlab-only)",
        f"Videos processed: {len(accident_results)} accident, {len(non_accident_results)} non-accident",
        f"",
        f"--- Prediction Surprise (zero-action) ---",
        f"  Accident mean: {np.mean(acc_pred_means):.3f}" if acc_pred_means else "  N/A",
        f"  Non-accident mean: {np.mean(nonacc_pred_means):.3f}" if nonacc_pred_means else "  N/A",
    ]
    if acc_pred_means and nonacc_pred_means:
        summary_lines.append(
            f"  Accident/Non-accident ratio: {np.mean(acc_pred_means)/max(np.mean(nonacc_pred_means),1e-8):.2f}x")

    if pred_ratios:
        summary_lines.extend([
            f"  Pre-crash elevation (median): {np.median(pred_ratios):.2f}x",
            f"  Videos with pre-crash > 1.0: {np.mean(np.array(pred_ratios) > 1.0)*100:.0f}%",
            f"  Videos with pre-crash > 1.5: {np.mean(np.array(pred_ratios) > 1.5)*100:.0f}%",
        ])

    summary_lines.append(f"")
    summary_lines.append(f"--- Embedding Change Rate ---")
    if acc_cr_means:
        summary_lines.append(f"  Accident mean: {np.mean(acc_cr_means):.3f}")
    if nonacc_cr_means:
        summary_lines.append(f"  Non-accident mean: {np.mean(nonacc_cr_means):.3f}")
    if cr_ratios:
        summary_lines.extend([
            f"  Pre-crash elevation (median): {np.median(cr_ratios):.2f}x",
            f"  Videos with pre-crash > 1.0: {np.mean(np.array(cr_ratios) > 1.0)*100:.0f}%",
        ])

    summary_lines.append(f"")
    summary_lines.append(f"--- Time Window Analysis (pred surprise) ---")
    for w in sorted(window_analysis.keys()):
        wa = window_analysis[w]
        if wa["pred_ratio_median"] is not None:
            summary_lines.append(
                f"  {w}s before: median ratio={wa['pred_ratio_median']:.2f}, "
                f">{'>'}1.0: {wa['pred_pct_above_1']:.0f}% (n={wa['n_videos']})")

    text = "\n".join(summary_lines)
    ax3.text(0.05, 0.5, text, fontsize=11, family="monospace", va="center",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
             transform=ax3.transAxes)
    ax3.set_title("Summary Statistics", fontsize=13)

    fig.suptitle("Accident Surprise Analysis v2 — LeWM A1 (OccAny depth-supervised)", fontsize=15, y=0.99)

    fig.savefig(out_dir / "accident_surprise_v2.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_dir / 'accident_surprise_v2.png'}")
    plt.close(fig)

    # Save individual timelines
    print("\nSaving individual timelines...")
    plot_individual_timelines(accident_results, out_dir, n_examples=12)

    # Save JSON results
    results_json = {
        "model": str(args.ckpt),
        "num_accident": len(accident_results),
        "num_non_accident": len(non_accident_results),
        "pred_surprise": {
            "accident_means": acc_pred_means,
            "non_accident_means": nonacc_pred_means,
            "pre_crash_ratios_2s": pred_ratios,
        },
        "change_rate": {
            "accident_means": acc_cr_means,
            "non_accident_means": nonacc_cr_means,
            "pre_crash_ratios_2s": cr_ratios,
        },
        "window_analysis": window_analysis,
        "per_video": [
            {
                "video": r["video"],
                "label": r["label"],
                "duration": r["duration"],
                "pred_surprise_mean": float(np.nanmean(r["pred_surprise"])),
                "change_rate_mean": float(np.nanmean(r["change_rate"])),
                "collision_type": r.get("collision_type"),
                "collision_times": r.get("collision_times"),
            }
            for r in accident_results + non_accident_results
        ],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {out_dir / 'results.json'}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(text)
    print("=" * 70)

    plt.close("all")


if __name__ == "__main__":
    main()
