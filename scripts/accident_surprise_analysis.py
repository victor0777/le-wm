#!/usr/bin/env python3
"""Accident Surprise Analysis.

Computes frame-by-frame surprise scores on accident/non-accident videos
using a trained LeWM model. Analyzes:
1. Pre-crash surprise patterns (does surprise rise before collision?)
2. Accident vs non-accident surprise distributions
3. Per collision-type embedding trajectories

Usage:
    python scripts/accident_surprise_analysis.py
    python scripts/accident_surprise_analysis.py --max-videos 50 --output outputs/accident_surprise.png
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from matplotlib.gridspec import GridSpec

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


def extract_frames(video_path: Path, fps_target: int = 10, img_size: int = 224) -> tuple[np.ndarray, float]:
    """Extract frames from mp4 at target fps. Returns (N, 3, H, W) uint8 and original fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def compute_surprise_scores(model, frames: np.ndarray, transform, device: str,
                            history_size: int = 3, batch_size: int = 32) -> np.ndarray:
    """Compute per-frame surprise: ||pred_emb - actual_emb||².

    For each frame t (t >= history_size), predict embedding at t from frames [t-history_size:t],
    compare with actual embedding at t.
    Returns surprise scores array of length N, with NaN for first history_size frames.
    """
    # Encode all frames
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

    embeddings = torch.cat(embeddings, dim=0)  # (N, D)
    N = len(embeddings)

    # Compute surprise: for each frame t, use context [t-hs..t-1] to predict t
    # Since we don't have action for accident videos, use zero action
    surprise = np.full(N, np.nan)

    # Create dummy zero action embeddings
    embed_dim = embeddings.shape[1]

    with torch.no_grad():
        for t in range(history_size, N):
            ctx = embeddings[t - history_size:t].unsqueeze(0).to(device)  # (1, hs, D)
            # Zero action embedding (no motion info for accident videos)
            zero_act = torch.zeros(1, history_size, embed_dim, device=device)
            pred = model.predict(ctx, zero_act)  # (1, hs, D)
            pred_last = pred[0, -1]  # (D,)
            actual = embeddings[t].to(device)  # (D,)
            surprise[t] = (pred_last - actual).pow(2).sum().item()

    return surprise


def analyze_video(model, video_path: Path, label_info: dict, transform, device: str,
                  fps_target: int = 10) -> dict | None:
    """Analyze a single video. Returns dict with surprise scores and metadata."""
    frames, orig_fps = extract_frames(video_path, fps_target=fps_target)
    if frames is None or len(frames) < 10:
        return None

    surprise = compute_surprise_scores(model, frames, transform, device)

    result = {
        "video": video_path.stem,
        "label": label_info["label"],
        "num_frames": len(frames),
        "duration": len(frames) / fps_target,
        "surprise": surprise,
        "fps_target": fps_target,
    }

    # Add collision timing if available
    if label_info.get("collisions"):
        collision_times = [c["time_sec"] for c in label_info["collisions"] if "time_sec" in c]
        result["collision_times"] = collision_times
        if label_info["collisions"][0].get("collision_type"):
            result["collision_type"] = label_info["collisions"][0]["collision_type"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "lewm_rtb_multi_epoch_9_object.ckpt"))
    parser.add_argument("--max-videos", type=int, default=80,
                        help="Max videos to process (accident + non_accident)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=str, default="outputs/accident_surprise_analysis.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load labels
    with open(REVIEW_RESULTS) as f:
        labels = json.load(f)

    # Filter accident and non_accident videos that exist in /data2
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

    # Limit
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

    # === Analysis ===

    # 1. Pre-crash surprise pattern (aligned to collision time)
    pre_crash_window = 5  # seconds before collision
    post_crash_window = 3
    aligned_surprises = []

    for r in accident_results:
        if "collision_times" not in r or not r["collision_times"]:
            continue
        t_collision = r["collision_times"][0]
        fps = r["fps_target"]
        s = r["surprise"]

        # Create time axis relative to collision
        times = np.arange(len(s)) / fps
        rel_times = times - t_collision

        # Keep window around collision
        mask = (rel_times >= -pre_crash_window) & (rel_times <= post_crash_window)
        if mask.sum() < 5:
            continue

        aligned_surprises.append({
            "rel_time": rel_times[mask],
            "surprise": s[mask],
            "collision_type": r.get("collision_type", "unknown"),
        })

    # 2. Overall surprise distributions
    all_accident_surprise = np.concatenate(
        [r["surprise"][~np.isnan(r["surprise"])] for r in accident_results]
    ) if accident_results else np.array([])
    all_nonacc_surprise = np.concatenate(
        [r["surprise"][~np.isnan(r["surprise"])] for r in non_accident_results]
    ) if non_accident_results else np.array([])

    # 3. Mean surprise per video
    acc_means = [np.nanmean(r["surprise"]) for r in accident_results]
    nonacc_means = [np.nanmean(r["surprise"]) for r in non_accident_results]

    # 4. Pre-crash surprise ratio (last 2s before collision vs earlier)
    pre_crash_ratios = []
    for r in accident_results:
        if "collision_times" not in r or not r["collision_times"]:
            continue
        t_col = r["collision_times"][0]
        fps = r["fps_target"]
        s = r["surprise"]
        times = np.arange(len(s)) / fps

        pre_2s = s[(times >= t_col - 2) & (times < t_col) & ~np.isnan(s)]
        early = s[(times >= 1) & (times < t_col - 3) & ~np.isnan(s)]

        if len(pre_2s) > 2 and len(early) > 5:
            ratio = np.mean(pre_2s) / max(np.mean(early), 1e-8)
            pre_crash_ratios.append(ratio)

    # === Plotting ===
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Pre-crash aligned surprise curves
    ax1 = fig.add_subplot(gs[0, :2])
    if aligned_surprises:
        for a in aligned_surprises[:30]:
            ax1.plot(a["rel_time"], a["surprise"], alpha=0.2, color="red", linewidth=0.8)
        # Average curve
        bins = np.linspace(-pre_crash_window, post_crash_window, 80)
        avg_surprise = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)
        for a in aligned_surprises:
            for j in range(len(bins) - 1):
                mask = (a["rel_time"] >= bins[j]) & (a["rel_time"] < bins[j + 1])
                vals = a["surprise"][mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    avg_surprise[j] += vals.mean()
                    counts[j] += 1
        avg_surprise = np.where(counts > 0, avg_surprise / counts, np.nan)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax1.plot(bin_centers, avg_surprise, color="darkred", linewidth=3, label="Average")
        ax1.axvline(0, color="black", linestyle="--", linewidth=2, label="Collision")
    ax1.set_xlabel("Time relative to collision (s)")
    ax1.set_ylabel("Surprise Score")
    ax1.set_title(f"Pre-Crash Surprise Pattern ({len(aligned_surprises)} videos)")
    ax1.legend()

    # 2. Pre-crash ratio histogram
    ax2 = fig.add_subplot(gs[0, 2])
    if pre_crash_ratios:
        ax2.hist(pre_crash_ratios, bins=20, color="coral", edgecolor="white")
        ax2.axvline(1.0, color="black", linestyle="--", label="No change")
        median_ratio = np.median(pre_crash_ratios)
        ax2.axvline(median_ratio, color="red", linewidth=2, label=f"Median: {median_ratio:.2f}")
        pct_above = np.mean(np.array(pre_crash_ratios) > 1.0) * 100
        ax2.set_title(f"Pre-crash Surprise Ratio\n(last 2s / earlier, {pct_above:.0f}% > 1.0)")
        ax2.legend()
    ax2.set_xlabel("Surprise Ratio")

    # 3. Accident vs Non-accident distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if len(all_accident_surprise) > 0:
        ax3.hist(all_accident_surprise, bins=50, alpha=0.6, color="red",
                 label=f"Accident (n={len(accident_results)})", density=True)
    if len(all_nonacc_surprise) > 0:
        ax3.hist(all_nonacc_surprise, bins=50, alpha=0.6, color="green",
                 label=f"Non-accident (n={len(non_accident_results)})", density=True)
    ax3.set_xlabel("Surprise Score")
    ax3.set_ylabel("Density")
    ax3.set_title("Frame-level Surprise Distribution")
    ax3.legend()

    # 4. Per-video mean surprise
    ax4 = fig.add_subplot(gs[1, 1])
    if acc_means:
        ax4.hist(acc_means, bins=20, alpha=0.6, color="red", label="Accident")
    if nonacc_means:
        ax4.hist(nonacc_means, bins=20, alpha=0.6, color="green", label="Non-accident")
    ax4.set_xlabel("Mean Surprise per Video")
    ax4.set_title("Video-level Surprise Distribution")
    ax4.legend()

    # 5. Example accident timeline
    ax5 = fig.add_subplot(gs[1, 2])
    if accident_results:
        # Pick one with collision time
        example = None
        for r in accident_results:
            if "collision_times" in r and r["collision_times"]:
                example = r
                break
        if example:
            times = np.arange(len(example["surprise"])) / example["fps_target"]
            ax5.plot(times, example["surprise"], color="steelblue", linewidth=1)
            for ct in example.get("collision_times", []):
                ax5.axvline(ct, color="red", linestyle="--", linewidth=2, label=f"Collision @{ct:.1f}s")
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Surprise")
            ax5.set_title(f"Example: {example['video'][:30]}...")
            ax5.legend(fontsize=8)

    # 6. Collision type comparison
    ax6 = fig.add_subplot(gs[2, 0])
    type_surprises = {}
    for r in accident_results:
        ct = r.get("collision_type", "unknown")
        mean_s = np.nanmean(r["surprise"])
        type_surprises.setdefault(ct, []).append(mean_s)
    if type_surprises:
        types = sorted(type_surprises.keys(), key=lambda k: -len(type_surprises[k]))[:6]
        means = [np.mean(type_surprises[t]) for t in types]
        counts = [len(type_surprises[t]) for t in types]
        bars = ax6.bar(range(len(types)), means, color="salmon", edgecolor="white")
        ax6.set_xticks(range(len(types)))
        ax6.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=8, rotation=15)
        ax6.set_ylabel("Mean Surprise")
        ax6.set_title("Surprise by Collision Type")

    # 7. Summary stats
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis("off")
    summary = [
        f"Videos processed: {len(accident_results)} accident, {len(non_accident_results)} non-accident",
        f"Accident mean surprise: {np.mean(acc_means):.4f}" if acc_means else "N/A",
        f"Non-accident mean surprise: {np.mean(nonacc_means):.4f}" if nonacc_means else "N/A",
    ]
    if acc_means and nonacc_means:
        ratio = np.mean(acc_means) / max(np.mean(nonacc_means), 1e-8)
        summary.append(f"Accident/Non-accident ratio: {ratio:.2f}x")
    if pre_crash_ratios:
        summary.extend([
            f"Pre-crash surprise elevation: {np.median(pre_crash_ratios):.2f}x (median)",
            f"Videos with pre-crash elevation > 1.0: {np.mean(np.array(pre_crash_ratios) > 1.0) * 100:.0f}%",
            f"Videos with pre-crash elevation > 1.5: {np.mean(np.array(pre_crash_ratios) > 1.5) * 100:.0f}%",
        ])

    text = "\n".join(summary)
    ax7.text(0.1, 0.5, text, fontsize=13, family="monospace", va="center",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax7.set_title("Summary Statistics")

    fig.suptitle("Accident Surprise Analysis (LeWM RTB World Model)", fontsize=16, y=0.98)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")

    # Save raw results
    results_path = Path(args.output).with_suffix(".json")
    save_data = {
        "accident_means": acc_means,
        "non_accident_means": nonacc_means,
        "pre_crash_ratios": pre_crash_ratios,
        "num_accident": len(accident_results),
        "num_non_accident": len(non_accident_results),
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved results: {results_path}")

    plt.close()


if __name__ == "__main__":
    main()
