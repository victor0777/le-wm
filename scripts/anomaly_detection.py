#!/usr/bin/env python3
"""Route-specific anomaly detection using JEPA world model.

Computes "surprise scores" by comparing predicted future embeddings
against actual future embeddings. High surprise = unexpected event.

Workflow:
  1. Slide a window through the recording (history_size=3)
  2. At each position: encode actual frames, predict future embeddings
  3. Surprise = MSE or (1 - cosine_similarity) between predicted and actual
  4. Multi-horizon analysis (1, 3, 5 steps)
  5. Stratify by maneuver type, compare train vs holdout recordings

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ktl/projects/le-wm python scripts/anomaly_detection.py
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

# Import VP decoders so torch.load can unpickle them
try:
    from train_vp import LaneDecoder, DepthDecoder  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def normalize_actions(action: np.ndarray):
    """StandardScaler normalization fitted on the given data."""
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0, keepdims=True)
    std = flat[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Core: sliding-window surprise computation
# ---------------------------------------------------------------------------

def compute_surprise_scores(
    model,
    h5_path: str,
    transform,
    device: str = "cuda",
    frameskip: int = 5,
    history_size: int = 3,
    max_horizon: int = 5,
    batch_size: int = 32,
):
    """Compute per-frame surprise scores at multiple prediction horizons.

    For each position t in the recording, we use frames [t-history_size+1 .. t]
    as context and predict embeddings at horizons 1..max_horizon steps ahead.

    Returns:
        surprise_mse: dict {horizon: np.array of shape (N_windows,)}
        surprise_cos: dict {horizon: np.array of shape (N_windows,)}
        window_starts: np.array of frame indices (subsampled) for each window
    """
    with h5py.File(h5_path, "r") as f:
        total_frames = f["pixels"].shape[0]
        action_dim = f["action"].shape[-1]

        # We work with subsampled frames (every frameskip)
        # Total subsampled frames
        sub_total = total_frames  # already subsampled in rtb_occany
        # Number of raw frames needed for one sequence of length T
        # In rtb_occany, pixels are already at frameskip resolution
        # Actions need to be chunked: frameskip consecutive raw actions per step
        # But rtb_occany stores 1 action per frame (already at sub rate)

        # Load all data
        pixels_all = f["pixels"][:]      # (N, 3, H, W)
        action_all = f["action"][:]      # (N, action_dim)
        proprio_all = f["proprio"][:]    # (N, 8)

    print(f"  Total frames: {total_frames}, action_dim: {action_dim}")

    # Chunk actions by frameskip (each step = frameskip * action_dim)
    # For rtb_occany the data is already at subsampled rate but actions are per-frame
    # We need to create chunked actions matching the training format
    effective_act_dim = frameskip * action_dim
    n_chunked = total_frames // frameskip
    # Reshape actions into chunks
    action_chunked = action_all[:n_chunked * frameskip].reshape(n_chunked, effective_act_dim)

    # Pixels are subsampled every frameskip
    pixels_sub = pixels_all[::frameskip][:n_chunked]

    n_sub = min(len(pixels_sub), len(action_chunked))
    pixels_sub = pixels_sub[:n_sub]
    action_chunked = action_chunked[:n_sub]

    print(f"  Subsampled frames: {n_sub} (frameskip={frameskip})")

    # Normalize actions
    action_norm, act_mean, act_std = normalize_actions(action_chunked)

    # For each window: need history_size context + max_horizon future
    window_len = history_size + max_horizon
    n_windows = n_sub - window_len + 1

    if n_windows <= 0:
        raise ValueError(f"Not enough frames: {n_sub} < {window_len}")

    print(f"  Windows: {n_windows} (history={history_size}, max_horizon={max_horizon})")

    # Initialize result containers
    horizons = [1, 3, 5]
    horizons = [h for h in horizons if h <= max_horizon]
    surprise_mse = {h: [] for h in horizons}
    surprise_cos = {h: [] for h in horizons}
    window_indices = np.arange(n_windows)

    # Process in batches
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        b_size = batch_end - batch_start

        # Gather sequences for this batch
        px_seqs = []
        act_seqs = []
        for w in range(batch_start, batch_end):
            px_seqs.append(pixels_sub[w : w + window_len])
            act_seqs.append(action_norm[w : w + window_len])

        px_batch = np.stack(px_seqs)   # (B, window_len, 3, H, W)
        act_batch = np.stack(act_seqs) # (B, window_len, eff_act_dim)

        px_t = torch.from_numpy(px_batch)
        act_t = torch.from_numpy(act_batch)

        # Apply image transform
        batch_dict = {"pixels": px_t}
        batch_dict = transform(batch_dict)
        px_t = batch_dict["pixels"].to(device)  # (B, window_len, 3, 224, 224)
        act_t = act_t.to(device)

        with torch.no_grad():
            # Encode ALL frames to get ground truth embeddings
            info = {"pixels": px_t, "action": act_t}
            info = model.encode(info)
            all_emb = info["emb"]        # (B, window_len, D)
            all_act_emb = info["act_emb"] # (B, window_len, D)

            # Context: first history_size frames
            ctx_emb = all_emb[:, :history_size]     # (B, H, D)
            ctx_act = all_act_emb[:, :history_size]  # (B, H, D)

            # Multi-step autoregressive prediction
            emb_pred = ctx_emb.clone()
            act_for_pred = act_t[:, :history_size]

            for step in range(max_horizon):
                act_emb_step = model.action_encoder(act_for_pred)
                emb_trunc = emb_pred[:, -history_size:]
                act_trunc = act_emb_step[:, -history_size:]
                next_pred = model.predict(emb_trunc, act_trunc)[:, -1:]  # (B, 1, D)
                emb_pred = torch.cat([emb_pred, next_pred], dim=1)

                # Add next action
                next_act = act_t[:, history_size + step : history_size + step + 1]
                act_for_pred = torch.cat([act_for_pred, next_act], dim=1)

            # Compute surprise for each horizon
            for h in horizons:
                pred_at_h = emb_pred[:, history_size + h - 1]   # (B, D) — predicted
                actual_at_h = all_emb[:, history_size + h - 1]  # (B, D) — ground truth

                mse = (pred_at_h - actual_at_h).pow(2).mean(dim=-1).cpu().numpy()
                cos = F.cosine_similarity(pred_at_h, actual_at_h, dim=-1).cpu().numpy()

                surprise_mse[h].append(mse)
                surprise_cos[h].append(1.0 - cos)  # surprise = 1 - similarity

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Processed {batch_end}/{n_windows} windows")

    # Concatenate
    for h in horizons:
        surprise_mse[h] = np.concatenate(surprise_mse[h])
        surprise_cos[h] = np.concatenate(surprise_cos[h])

    # Map window indices back to original frame indices
    # window_starts[i] corresponds to the context start (subsampled frame index)
    # The "current" frame is at window_start + history_size - 1
    frame_indices = np.arange(n_windows) + history_size - 1  # center of context
    # Map to original frame indices
    original_frame_indices = frame_indices * frameskip

    return surprise_mse, surprise_cos, original_frame_indices, n_windows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_surprise(
    surprise_mse,
    surprise_cos,
    frame_indices,
    labels_path: str = None,
    name: str = "recording",
    frameskip: int = 5,
):
    """Print detailed analysis of surprise scores."""
    print(f"\n{'='*70}")
    print(f"SURPRISE ANALYSIS: {name}")
    print(f"{'='*70}")

    for h in sorted(surprise_mse.keys()):
        mse = surprise_mse[h]
        cos = surprise_cos[h]
        print(f"\n--- Horizon {h} step(s) ({h*frameskip/10:.1f}s at 10Hz sub) ---")
        print(f"  MSE  : mean={mse.mean():.6f}, std={mse.std():.6f}, "
              f"median={np.median(mse):.6f}")
        print(f"         p90={np.percentile(mse,90):.6f}, "
              f"p95={np.percentile(mse,95):.6f}, "
              f"p99={np.percentile(mse,99):.6f}, "
              f"max={mse.max():.6f}")
        print(f"  Cos  : mean={cos.mean():.6f}, std={cos.std():.6f}, "
              f"median={np.median(cos):.6f}")
        print(f"         p90={np.percentile(cos,90):.6f}, "
              f"p95={np.percentile(cos,95):.6f}, "
              f"p99={np.percentile(cos,99):.6f}, "
              f"max={cos.max():.6f}")

        # Top-20 most surprising moments
        top_idx = np.argsort(mse)[-20:][::-1]
        print(f"\n  Top-20 most surprising (MSE):")
        for rank, idx in enumerate(top_idx):
            print(f"    #{rank+1:2d}: frame={frame_indices[idx]:5d}, "
                  f"MSE={mse[idx]:.6f}, CosSurp={cos[idx]:.6f}")

    # Temporal clustering analysis (horizon=1)
    if 1 in surprise_mse:
        mse = surprise_mse[1]
        threshold = np.percentile(mse, 95)
        high_surprise = mse > threshold
        # Find clusters (consecutive high-surprise frames)
        clusters = []
        in_cluster = False
        start = 0
        for i in range(len(high_surprise)):
            if high_surprise[i] and not in_cluster:
                start = i
                in_cluster = True
            elif not high_surprise[i] and in_cluster:
                clusters.append((start, i - 1, mse[start:i].max()))
                in_cluster = False
        if in_cluster:
            clusters.append((start, len(high_surprise) - 1, mse[start:].max()))

        print(f"\n  Temporal clustering (p95 threshold={threshold:.6f}):")
        print(f"  High-surprise frames: {high_surprise.sum()} / {len(mse)} "
              f"({high_surprise.mean()*100:.1f}%)")
        print(f"  Number of clusters: {len(clusters)}")
        if clusters:
            lengths = [c[1] - c[0] + 1 for c in clusters]
            print(f"  Cluster sizes: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}")
            print(f"  Top-5 clusters by peak surprise:")
            sorted_clusters = sorted(clusters, key=lambda c: c[2], reverse=True)[:5]
            for i, (s, e, peak) in enumerate(sorted_clusters):
                print(f"    Cluster {i+1}: frames {frame_indices[s]}-{frame_indices[e]}, "
                      f"length={e-s+1}, peak_MSE={peak:.6f}")

    # Maneuver stratification
    if labels_path is not None:
        label_data = np.load(labels_path)
        label_arr = label_data["labels"]
        label_map = json.loads(str(label_data["label_map"]))
        inv_map = {v: k for k, v in label_map.items()}

        # Map window indices to label indices
        # Labels are per original frame; our windows map to subsampled frames
        # frame_indices are original frame indices
        # Labels have 1 per original frame
        n_labels = len(label_arr)

        print(f"\n  Maneuver stratification:")
        for h in sorted(surprise_mse.keys()):
            mse = surprise_mse[h]
            cos = surprise_cos[h]
            print(f"\n  Horizon {h}:")

            maneuver_stats = {}
            for lbl_val, lbl_name in sorted(inv_map.items()):
                # Map frame_indices to label indices
                valid_mask = frame_indices < n_labels
                lbl_mask = np.zeros(len(mse), dtype=bool)
                lbl_mask[valid_mask] = label_arr[frame_indices[valid_mask]] == lbl_val

                if lbl_mask.sum() > 0:
                    m = mse[lbl_mask]
                    c = cos[lbl_mask]
                    maneuver_stats[lbl_name] = {
                        "count": int(lbl_mask.sum()),
                        "mse_mean": float(m.mean()),
                        "mse_std": float(m.std()),
                        "cos_mean": float(c.mean()),
                        "cos_std": float(c.std()),
                    }
                    print(f"    {lbl_name:10s}: n={lbl_mask.sum():4d}, "
                          f"MSE={m.mean():.6f} +/- {m.std():.6f}, "
                          f"CosSurp={c.mean():.6f} +/- {c.std():.6f}")

            return maneuver_stats  # return for plotting

    return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_visualizations(
    holdout_mse, holdout_cos, holdout_frames,
    train_mse, train_cos, train_frames,
    holdout_labels_path, train_labels_path,
    output_dir: str,
    frameskip: int = 5,
):
    """Create all visualization plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_data_h = np.load(holdout_labels_path)
    label_map = json.loads(str(label_data_h["label_map"]))
    inv_map = {v: k for k, v in label_map.items()}

    # -----------------------------------------------------------------------
    # 1. Surprise timeline (holdout)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    colors_h = {1: "#e74c3c", 3: "#f39c12", 5: "#8e44ad"}
    for h in sorted(holdout_mse.keys()):
        time_sec = holdout_frames[: len(holdout_mse[h])] / 10.0  # ~10Hz original
        axes[0].plot(time_sec, holdout_mse[h], alpha=0.7, linewidth=0.5,
                     color=colors_h[h], label=f"h={h}")
        axes[1].plot(time_sec, holdout_cos[h], alpha=0.7, linewidth=0.5,
                     color=colors_h[h], label=f"h={h}")

    # Mark p95 threshold for h=1
    if 1 in holdout_mse:
        thresh = np.percentile(holdout_mse[1], 95)
        axes[0].axhline(thresh, color="red", linestyle="--", alpha=0.5, label="p95")
        # Mark high-surprise regions
        high = holdout_mse[1] > thresh
        time_sec = holdout_frames[:len(holdout_mse[1])] / 10.0
        axes[0].fill_between(time_sec, 0, axes[0].get_ylim()[1] if axes[0].get_ylim()[1] > 0 else thresh * 2,
                             where=high, alpha=0.15, color="red")

    axes[0].set_ylabel("MSE Surprise")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Surprise Score Timeline (Holdout: 8014dd)")
    axes[1].set_ylabel("Cosine Surprise")
    axes[1].legend(loc="upper right")

    # Add maneuver labels as colored background
    labels_h = label_data_h["labels"]
    maneuver_colors = {
        "left": "#3498db", "right": "#e67e22", "straight": "#2ecc71",
        "stop": "#95a5a6", "accel": "#e74c3c", "decel": "#9b59b6",
    }
    n_labels = len(labels_h)
    for i in range(n_labels - 1):
        lbl = inv_map.get(labels_h[i], "?")
        c = maneuver_colors.get(lbl, "#cccccc")
        axes[2].axvspan(i / 10.0, (i + 1) / 10.0, alpha=0.5, color=c, linewidth=0)

    # Legend for maneuvers
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=maneuver_colors[k], alpha=0.6, label=k)
                       for k in maneuver_colors]
    axes[2].legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=8)
    axes[2].set_ylabel("Maneuver")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(out / "surprise_timeline.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'surprise_timeline.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 2. Surprise distribution: train vs holdout
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, h in enumerate(sorted(holdout_mse.keys())):
        ax = axes[i]
        bins = np.linspace(0, max(holdout_mse[h].max(), train_mse[h].max()) * 0.95, 60)
        ax.hist(train_mse[h], bins=bins, alpha=0.6, color="#2ecc71",
                label=f"Train (736fcb)", density=True)
        ax.hist(holdout_mse[h], bins=bins, alpha=0.6, color="#e74c3c",
                label=f"Holdout (8014dd)", density=True)
        ax.axvline(train_mse[h].mean(), color="#27ae60", linestyle="--",
                   label=f"Train mean={train_mse[h].mean():.4f}")
        ax.axvline(holdout_mse[h].mean(), color="#c0392b", linestyle="--",
                   label=f"Holdout mean={holdout_mse[h].mean():.4f}")
        ax.set_xlabel("MSE Surprise")
        ax.set_ylabel("Density")
        ax.set_title(f"Horizon {h} ({h*frameskip/10:.1f}s)")
        ax.legend(fontsize=7)

    fig.suptitle("Surprise Distribution: Train vs Holdout", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "surprise_distribution.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'surprise_distribution.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 3. Surprise by maneuver type (boxplot)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, h in enumerate(sorted(holdout_mse.keys())):
        ax = axes[i]
        mse = holdout_mse[h]
        labels_arr = label_data_h["labels"]
        n = min(len(holdout_frames), len(labels_arr))

        data_by_maneuver = {}
        for lbl_val, lbl_name in sorted(inv_map.items()):
            valid = holdout_frames[:len(mse)] < n
            mask = np.zeros(len(mse), dtype=bool)
            mask[valid] = labels_arr[holdout_frames[:len(mse)][valid]] == lbl_val
            if mask.sum() > 5:
                data_by_maneuver[lbl_name] = mse[mask]

        if data_by_maneuver:
            bp = ax.boxplot(
                [data_by_maneuver[k] for k in data_by_maneuver],
                tick_labels=list(data_by_maneuver.keys()),
                patch_artist=True,
                showfliers=False,
            )
            for patch, name in zip(bp["boxes"], data_by_maneuver.keys()):
                patch.set_facecolor(maneuver_colors.get(name, "#cccccc"))
                patch.set_alpha(0.7)

        ax.set_ylabel("MSE Surprise")
        ax.set_title(f"Horizon {h} ({h*frameskip/10:.1f}s)")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Surprise by Maneuver Type (Holdout: 8014dd)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "surprise_by_maneuver.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'surprise_by_maneuver.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 4. Top-20 most surprising moments
    # -----------------------------------------------------------------------
    if 1 in holdout_mse:
        mse_h1 = holdout_mse[1]
        top_idx = np.argsort(mse_h1)[-20:][::-1]

        fig, axes = plt.subplots(4, 5, figsize=(20, 12))
        axes = axes.flatten()

        # Load pixels for visualization
        with h5py.File(
            str(STABLEWM_HOME / "rtb_occany" / "Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"),
            "r",
        ) as f:
            all_pixels = f["pixels"]
            for rank, idx in enumerate(top_idx):
                ax = axes[rank]
                frame_idx = holdout_frames[idx]
                if frame_idx < all_pixels.shape[0]:
                    img = all_pixels[frame_idx]  # (3, H, W) uint8
                    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
                    ax.imshow(img)

                # Get maneuver label
                lbl_str = "?"
                if frame_idx < len(label_data_h["labels"]):
                    lbl_val = label_data_h["labels"][frame_idx]
                    lbl_str = inv_map.get(lbl_val, "?")

                ax.set_title(f"#{rank+1} f={frame_idx}\nMSE={mse_h1[idx]:.4f}\n{lbl_str}",
                             fontsize=8)
                ax.axis("off")

        fig.suptitle("Top-20 Most Surprising Moments (h=1, Holdout: 8014dd)", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / "top_surprises.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {out / 'top_surprises.png'}")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 5. Horizon comparison
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE by horizon
    ax = axes[0]
    horizons = sorted(holdout_mse.keys())
    h_means = [holdout_mse[h].mean() for h in horizons]
    t_means = [train_mse[h].mean() for h in horizons]
    x = np.arange(len(horizons))
    ax.bar(x - 0.15, h_means, 0.3, label="Holdout (8014dd)", color="#e74c3c", alpha=0.8)
    ax.bar(x + 0.15, t_means, 0.3, label="Train (736fcb)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("Mean MSE Surprise")
    ax.set_title("MSE Surprise by Horizon")
    ax.legend()

    # Cosine by horizon
    ax = axes[1]
    h_means = [holdout_cos[h].mean() for h in horizons]
    t_means = [train_cos[h].mean() for h in horizons]
    ax.bar(x - 0.15, h_means, 0.3, label="Holdout (8014dd)", color="#e74c3c", alpha=0.8)
    ax.bar(x + 0.15, t_means, 0.3, label="Train (736fcb)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("Mean Cosine Surprise")
    ax.set_title("Cosine Surprise by Horizon")
    ax.legend()

    fig.suptitle("Train vs Holdout Surprise Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "horizon_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'horizon_comparison.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Route-specific anomaly detection")
    parser.add_argument(
        "--ckpt",
        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"),
    )
    parser.add_argument(
        "--holdout",
        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"),
    )
    parser.add_argument(
        "--train-ref",
        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-5_JT_2025-09-22_06-57-30_2111_736fcb.h5"),
    )
    parser.add_argument(
        "--holdout-labels",
        default=str(STABLEWM_HOME / "rtb_occany_labels/8014dd_labels.npz"),
    )
    parser.add_argument(
        "--train-labels",
        default=str(STABLEWM_HOME / "rtb_occany_labels/736fcb_labels.npz"),
    )
    parser.add_argument("--output-dir", default="outputs/anomaly_detection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("ROUTE-SPECIFIC ANOMALY DETECTION")
    print("=" * 70)
    print(f"Model: {args.ckpt}")
    print(f"Holdout: {args.holdout}")
    print(f"Train ref: {args.train_ref}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # -----------------------------------------------------------------------
    # Holdout recording
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Processing HOLDOUT recording (8014dd)...")
    print(f"{'='*70}")
    holdout_mse, holdout_cos, holdout_frames, n_h = compute_surprise_scores(
        model, args.holdout, transform, args.device,
        frameskip=args.frameskip,
        history_size=args.history_size,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
    )
    holdout_maneuver = analyze_surprise(
        holdout_mse, holdout_cos, holdout_frames,
        labels_path=args.holdout_labels,
        name="HOLDOUT (8014dd — Livlab overlap)",
        frameskip=args.frameskip,
    )

    # -----------------------------------------------------------------------
    # Train recording
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Processing TRAIN recording (736fcb)...")
    print(f"{'='*70}")
    train_mse, train_cos, train_frames, n_t = compute_surprise_scores(
        model, args.train_ref, transform, args.device,
        frameskip=args.frameskip,
        history_size=args.history_size,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
    )
    train_maneuver = analyze_surprise(
        train_mse, train_cos, train_frames,
        labels_path=args.train_labels,
        name="TRAIN (736fcb — Livlab seen)",
        frameskip=args.frameskip,
    )

    # -----------------------------------------------------------------------
    # Cross-recording comparison
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CROSS-RECORDING COMPARISON")
    print(f"{'='*70}")
    for h in sorted(holdout_mse.keys()):
        h_mean = holdout_mse[h].mean()
        t_mean = train_mse[h].mean()
        ratio = h_mean / t_mean if t_mean > 0 else float("inf")
        diff_pct = (h_mean - t_mean) / t_mean * 100 if t_mean > 0 else float("inf")
        print(f"  Horizon {h}: Train={t_mean:.6f}, Holdout={h_mean:.6f}, "
              f"Ratio={ratio:.3f}, Diff={diff_pct:+.1f}%")

        h_cos_mean = holdout_cos[h].mean()
        t_cos_mean = train_cos[h].mean()
        cos_ratio = h_cos_mean / t_cos_mean if t_cos_mean > 0 else float("inf")
        print(f"           Cos: Train={t_cos_mean:.6f}, Holdout={h_cos_mean:.6f}, "
              f"Ratio={cos_ratio:.3f}")

    # Statistical test (Mann-Whitney U)
    from scipy import stats
    print(f"\n  Statistical significance (Mann-Whitney U test):")
    for h in sorted(holdout_mse.keys()):
        u_stat, p_val = stats.mannwhitneyu(
            holdout_mse[h], train_mse[h], alternative="greater"
        )
        print(f"    Horizon {h}: U={u_stat:.0f}, p={p_val:.2e} "
              f"({'*' if p_val < 0.05 else 'n.s.'})")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "surprise_scores.npz",
        holdout_mse_h1=holdout_mse.get(1, np.array([])),
        holdout_mse_h3=holdout_mse.get(3, np.array([])),
        holdout_mse_h5=holdout_mse.get(5, np.array([])),
        holdout_cos_h1=holdout_cos.get(1, np.array([])),
        holdout_cos_h3=holdout_cos.get(3, np.array([])),
        holdout_cos_h5=holdout_cos.get(5, np.array([])),
        holdout_frames=holdout_frames,
        train_mse_h1=train_mse.get(1, np.array([])),
        train_mse_h3=train_mse.get(3, np.array([])),
        train_mse_h5=train_mse.get(5, np.array([])),
        train_cos_h1=train_cos.get(1, np.array([])),
        train_cos_h3=train_cos.get(3, np.array([])),
        train_cos_h5=train_cos.get(5, np.array([])),
        train_frames=train_frames,
    )
    print(f"\nSaved scores: {out_dir / 'surprise_scores.npz'}")

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    print(f"\nCreating visualizations...")
    create_visualizations(
        holdout_mse, holdout_cos, holdout_frames,
        train_mse, train_cos, train_frames,
        args.holdout_labels, args.train_labels,
        args.output_dir,
        frameskip=args.frameskip,
    )

    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
