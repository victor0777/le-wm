#!/usr/bin/env python3
"""C3: Scene-type stratified motion ablation.

Tests hypothesis H4: action meaning depends on road structure context.
Classifies sequences by scene type using proprio/action signals, then
runs E0 ablation (correct/shuffled/zeroed) per scene type.

Scene types (from action/proprio signals):
  - "stop": very low speed (< 0.5 m/s)
  - "straight": low yaw rate, stable speed
  - "curve": sustained yaw rate
  - "accel/decel": large speed change over the sequence

Also runs multi-step rollout (3 history + 5 rollout) per scene type.

Outputs:
  - Per-scene-type shuffled gap and zeroed gap
  - Grouped bar chart: outputs/c3_scene_stratified.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

try:
    from train_vp import LaneDecoder, DepthDecoder  # noqa: F401
except ImportError:
    pass


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


def load_sequences(h5_path: str, n_sequences: int = 500, frameskip: int = 5,
                   num_steps: int = 4, seed: int = 42):
    """Load sequences with pixels, action, proprio."""
    rng = np.random.RandomState(seed)
    raw_len = num_steps * frameskip

    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        max_start = total - raw_len
        starts = rng.randint(0, max_start, size=n_sequences)

        pixels_list, action_list, proprio_list = [], [], []
        for s in starts:
            px_indices = list(range(s, s + raw_len, frameskip))
            pixels_list.append(f["pixels"][px_indices])
            action_raw = f["action"][s: s + raw_len]
            action_dim = action_raw.shape[-1]
            action_list.append(action_raw.reshape(num_steps, frameskip * action_dim))
            proprio_list.append(f["proprio"][px_indices])

    pixels = np.stack(pixels_list)
    action = np.stack(action_list)
    proprio = np.stack(proprio_list)
    return pixels, action, proprio


def load_long_sequences(h5_path: str, n_sequences: int = 300, frameskip: int = 5,
                        num_steps: int = 8, seed: int = 42):
    """Load longer sequences for rollout evaluation."""
    rng = np.random.RandomState(seed)
    raw_len = num_steps * frameskip

    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        max_start = total - raw_len
        starts = rng.randint(0, max_start, size=n_sequences)

        pixels_list, action_list, proprio_list = [], [], []
        for s in starts:
            px_indices = list(range(s, s + raw_len, frameskip))
            pixels_list.append(f["pixels"][px_indices])
            action_raw = f["action"][s: s + raw_len]
            action_dim = action_raw.shape[-1]
            action_list.append(action_raw.reshape(num_steps, frameskip * action_dim))
            proprio_list.append(f["proprio"][px_indices])

    pixels = np.stack(pixels_list)
    action = np.stack(action_list)
    proprio = np.stack(proprio_list)
    return pixels, action, proprio


def normalize_actions(action: np.ndarray):
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0, keepdims=True)
    std = flat[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32)


def classify_scene_type(proprio: np.ndarray, action: np.ndarray,
                        speed_stop_thresh: float = 0.5,
                        yaw_curve_thresh: float = 0.03,
                        accel_thresh: float = 0.15):
    """Classify each sequence into a scene type.

    Args:
        proprio: (N, T, 8) — proprio[:,t,0] = speed
        action: (N, T, action_dim) — raw (unnormalized) action;
                for chunked action (frameskip*action_dim), yaw_rate is at indices 2,5,8,...
        speed_stop_thresh: m/s threshold for 'stop'
        yaw_curve_thresh: rad/s threshold for 'curve'
        accel_thresh: m/s^2-ish threshold for speed change fraction

    Returns:
        labels: list of str, one per sequence
    """
    N = proprio.shape[0]
    labels = []

    for i in range(N):
        # Speed across timesteps
        speeds = proprio[i, :, 0]  # (T,)
        mean_speed = np.mean(np.abs(speeds))

        # Yaw rate: action is chunked (frameskip*action_dim).
        # For 3D action [vx, vy, yaw], chunked = [vx0,vy0,yaw0, vx1,vy1,yaw1, ...]
        # yaw indices within each chunk: 2, 5, 8, 11, 14
        act_row = action[i]  # (T, chunked_dim)
        chunk_dim = act_row.shape[-1]
        # Infer action_dim from chunk: frameskip=5, so action_dim = chunk_dim / 5
        action_dim = chunk_dim // 5
        # Extract yaw_rate values (index 2 within each sub-action)
        yaw_idx = 2 if action_dim >= 3 else (action_dim - 1)
        yaw_values = []
        for t in range(act_row.shape[0]):
            for k in range(5):  # frameskip
                idx = k * action_dim + yaw_idx
                if idx < chunk_dim:
                    yaw_values.append(act_row[t, idx])
        mean_abs_yaw = np.mean(np.abs(yaw_values))

        # Speed change: max - min relative to mean
        speed_range = np.max(np.abs(speeds)) - np.min(np.abs(speeds))
        speed_change_ratio = speed_range / (mean_speed + 1e-6)

        # Classification priority: stop > curve > accel/decel > straight
        if mean_speed < speed_stop_thresh:
            labels.append("stop")
        elif mean_abs_yaw > yaw_curve_thresh:
            labels.append("curve")
        elif speed_change_ratio > accel_thresh:
            labels.append("accel/decel")
        else:
            labels.append("straight")

    return labels


def run_prediction(model, pixels_t, action_t, transform, device, history_size=3):
    batch = {"pixels": pixels_t}
    batch = transform(batch)
    px = batch["pixels"].to(device)
    act = action_t.to(device)

    with torch.no_grad():
        info = {"pixels": px, "action": act}
        info = model.encode(info)
        emb = info["emb"]
        act_emb = info["act_emb"]

        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        tgt_emb = emb[:, 1:]

        pred_emb = model.predict(ctx_emb, ctx_act)

    return pred_emb.cpu(), tgt_emb.cpu()


def compute_metrics(pred_emb, tgt_emb):
    mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1).mean(dim=-1)  # (B,)
    cos = torch.nn.functional.cosine_similarity(
        pred_emb.reshape(-1, pred_emb.shape[-1]),
        tgt_emb.reshape(-1, tgt_emb.shape[-1]),
        dim=-1,
    )
    return mse.numpy(), cos.numpy()


def encode_all_frames(model, pixels_t, transform, device, batch_size=32):
    all_emb = []
    for i in range(0, len(pixels_t), batch_size):
        px_batch = pixels_t[i: i + batch_size]
        batch = {"pixels": px_batch}
        batch = transform(batch)
        px = batch["pixels"].to(device)
        with torch.no_grad():
            info = {"pixels": px, "action": torch.zeros(px.size(0), px.size(1), 15, device=device)}
            info = model.encode(info)
        all_emb.append(info["emb"].cpu())
    return torch.cat(all_emb, dim=0)


def rollout_predict(model, emb_history, action_seq, device, history_size=3):
    total_steps = action_seq.size(1)
    n_rollout = total_steps - history_size

    emb = emb_history.clone().to(device)
    action_seq = action_seq.to(device)

    pred_list = []
    with torch.no_grad():
        for t in range(n_rollout):
            act_window = action_seq[:, t: t + history_size]
            act_emb = model.action_encoder(act_window)
            pred = model.predict(emb[:, -history_size:], act_emb)[:, -1:]
            pred_list.append(pred.cpu())
            emb = torch.cat([emb, pred], dim=1)

    return torch.cat(pred_list, dim=1)


# ─── PART 1: Single-step E0 ablation per scene type ───────────────────────


def run_e0_stratified(model, transform, h5_path, device, n_sequences=500):
    print("\n" + "=" * 70)
    print("PART 1: Single-step E0 Ablation — Stratified by Scene Type")
    print("=" * 70)

    pixels, action_raw, proprio = load_sequences(h5_path, n_sequences=n_sequences, num_steps=4)
    print(f"  pixels: {pixels.shape}, action: {action_raw.shape}, proprio: {proprio.shape}")

    # Classify scenes
    scene_labels = classify_scene_type(proprio, action_raw)
    scene_labels = np.array(scene_labels)
    unique_types = sorted(set(scene_labels))
    print(f"  Scene type distribution:")
    for st in unique_types:
        cnt = (scene_labels == st).sum()
        print(f"    {st:12s}: {cnt:4d} ({cnt / len(scene_labels) * 100:.1f}%)")

    # Normalize actions
    action_norm = normalize_actions(action_raw)
    pixels_t = torch.from_numpy(pixels)
    action_correct = torch.from_numpy(action_norm)

    rng = np.random.RandomState(123)
    shuffle_idx = rng.permutation(len(action_norm))
    action_shuffled = torch.from_numpy(action_norm[shuffle_idx])
    action_zeroed = torch.zeros_like(action_correct)

    conditions = {"correct": action_correct, "shuffled": action_shuffled, "zeroed": action_zeroed}
    batch_size = 32

    # Run all conditions
    results = {}
    for cond_name, act_tensor in conditions.items():
        print(f"  Running condition: {cond_name}")
        all_mse, all_cos = [], []
        for i in range(0, len(pixels_t), batch_size):
            px_batch = pixels_t[i: i + batch_size]
            act_batch = act_tensor[i: i + batch_size]
            pred_emb, tgt_emb = run_prediction(model, px_batch, act_batch, transform, device)
            mse, cos = compute_metrics(pred_emb, tgt_emb)
            all_mse.append(mse)
            all_cos.append(cos)
        results[cond_name] = {
            "mse": np.concatenate(all_mse),
            "cos": np.concatenate(all_cos),
        }

    # Per-scene-type results
    scene_results = {}
    print(f"\n{'Scene Type':>12s} | {'N':>4s} | {'Correct MSE':>12s} | {'Shuf MSE':>12s} | {'Zero MSE':>12s} | {'Shuf Gap':>9s} | {'Zero Gap':>9s}")
    print("-" * 85)

    for st in unique_types:
        mask = scene_labels == st
        n = mask.sum()
        c_mse = results["correct"]["mse"][mask].mean()
        s_mse = results["shuffled"]["mse"][mask].mean()
        z_mse = results["zeroed"]["mse"][mask].mean()
        sg = (s_mse - c_mse) / c_mse * 100 if c_mse > 0 else 0
        zg = (z_mse - c_mse) / c_mse * 100 if c_mse > 0 else 0

        scene_results[st] = {
            "n": n, "correct_mse": c_mse, "shuffled_mse": s_mse, "zeroed_mse": z_mse,
            "shuffled_gap": sg, "zeroed_gap": zg,
        }
        print(f"  {st:>10s} | {n:4d} | {c_mse:12.6f} | {s_mse:12.6f} | {z_mse:12.6f} | {sg:>+8.1f}% | {zg:>+8.1f}%")

    # Overall
    c_all = results["correct"]["mse"].mean()
    s_all = results["shuffled"]["mse"].mean()
    z_all = results["zeroed"]["mse"].mean()
    sg_all = (s_all - c_all) / c_all * 100
    zg_all = (z_all - c_all) / c_all * 100
    print(f"  {'OVERALL':>10s} | {len(scene_labels):4d} | {c_all:12.6f} | {s_all:12.6f} | {z_all:12.6f} | {sg_all:>+8.1f}% | {zg_all:>+8.1f}%")

    return scene_results, unique_types


# ─── PART 2: Multi-step rollout per scene type ────────────────────────────


def run_rollout_stratified(model, transform, h5_path, device, n_sequences=300,
                           num_steps=8, history_size=3):
    n_rollout = num_steps - history_size
    print(f"\n{'=' * 70}")
    print(f"PART 2: Multi-step Rollout — Stratified by Scene Type")
    print(f"  History: {history_size} frames, Rollout: {n_rollout} steps")
    print("=" * 70)

    pixels, action_raw, proprio = load_long_sequences(
        h5_path, n_sequences=n_sequences, num_steps=num_steps)
    print(f"  pixels: {pixels.shape}, action: {action_raw.shape}, proprio: {proprio.shape}")

    scene_labels = classify_scene_type(proprio, action_raw)
    scene_labels = np.array(scene_labels)
    unique_types = sorted(set(scene_labels))
    print(f"  Scene type distribution:")
    for st in unique_types:
        cnt = (scene_labels == st).sum()
        print(f"    {st:12s}: {cnt:4d} ({cnt / len(scene_labels) * 100:.1f}%)")

    action_norm = normalize_actions(action_raw)
    pixels_t = torch.from_numpy(pixels)
    action_correct = torch.from_numpy(action_norm)

    rng = np.random.RandomState(123)
    shuffle_idx = rng.permutation(len(action_norm))
    action_shuffled = torch.from_numpy(action_norm[shuffle_idx])
    action_zeroed = torch.zeros_like(action_correct)

    # Encode all frames
    print("  Encoding all frames...")
    gt_emb = encode_all_frames(model, pixels_t, transform, device)
    emb_history = gt_emb[:, :history_size]

    conditions = {"correct": action_correct, "shuffled": action_shuffled, "zeroed": action_zeroed}
    batch_size = 64

    # Per-condition rollout
    rollout_results = {}
    for cond_name, act_tensor in conditions.items():
        print(f"  Rollout condition: {cond_name}")
        all_preds = []
        for i in range(0, len(emb_history), batch_size):
            emb_batch = emb_history[i: i + batch_size]
            act_batch = act_tensor[i: i + batch_size]
            preds = rollout_predict(model, emb_batch, act_batch, device, history_size)
            all_preds.append(preds)
        pred_emb = torch.cat(all_preds, dim=0)
        tgt_emb = gt_emb[:, history_size:]

        # Per-sample, per-step MSE: (N, n_rollout)
        per_sample_step_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1).numpy()
        rollout_results[cond_name] = per_sample_step_mse

    # Per-scene-type, per-step results
    rollout_scene = {}
    for st in unique_types:
        mask = scene_labels == st
        n = mask.sum()
        rollout_scene[st] = {"n": n}
        for cond_name in conditions:
            rollout_scene[st][cond_name] = rollout_results[cond_name][mask].mean(axis=0)

    # Print per-step results by scene type
    print(f"\nPer-step results by scene type:")
    for st in unique_types:
        n = rollout_scene[st]["n"]
        print(f"\n  [{st}] (n={n})")
        print(f"    {'Step':>4s} | {'Time':>6s} | {'Correct':>10s} | {'Shuffled':>10s} | {'Zeroed':>10s} | {'Shuf Gap':>9s} | {'Zero Gap':>9s}")
        print(f"    {'-' * 65}")
        for t in range(n_rollout):
            c = rollout_scene[st]["correct"][t]
            s = rollout_scene[st]["shuffled"][t]
            z = rollout_scene[st]["zeroed"][t]
            sg = (s - c) / c * 100 if c > 0 else 0
            zg = (z - c) / c * 100 if c > 0 else 0
            print(f"    {t + 1:>4d} | {(t + 1) * 0.5:>5.1f}s | {c:>10.6f} | {s:>10.6f} | {z:>10.6f} | {sg:>+8.1f}% | {zg:>+8.1f}%")

    return rollout_scene, unique_types, n_rollout


# ─── PLOT ──────────────────────────────────────────────────────────────────


def make_plot(scene_results_e0, unique_e0, rollout_scene, unique_rollout, n_rollout, output_path):
    fig = plt.figure(figsize=(20, 14))

    # Layout: 3 rows
    # Row 1: E0 grouped bar (shuffled gap, zeroed gap) + E0 MSE by scene
    # Row 2: Rollout MSE per step per scene type (correct vs shuffled vs zeroed)
    # Row 3: Rollout gap per step per scene type

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    colors_cond = {"correct": "#2ecc71", "shuffled": "#e74c3c", "zeroed": "#3498db"}

    # ── Row 1, Left: Shuffled & Zeroed Gap by Scene Type ──
    ax1 = fig.add_subplot(gs[0, 0])
    types = [st for st in unique_e0 if scene_results_e0[st]["n"] >= 5]
    x = np.arange(len(types))
    width = 0.35
    shuf_gaps = [scene_results_e0[st]["shuffled_gap"] for st in types]
    zero_gaps = [scene_results_e0[st]["zeroed_gap"] for st in types]

    bars1 = ax1.bar(x - width / 2, shuf_gaps, width, label="shuffled gap", color="#e74c3c", edgecolor="white")
    bars2 = ax1.bar(x + width / 2, zero_gaps, width, label="zeroed gap", color="#3498db", edgecolor="white")
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{st}\n(n={scene_results_e0[st]['n']})" for st in types], fontsize=9)
    ax1.set_ylabel("Gap vs Correct (%)")
    ax1.set_title("E0: Motion Sensitivity Gap by Scene Type")
    ax1.legend()
    for bar, val in zip(bars1, shuf_gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:+.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, zero_gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:+.1f}%", ha="center", va="bottom", fontsize=8)

    # ── Row 1, Right: Absolute MSE by Scene Type ──
    ax2 = fig.add_subplot(gs[0, 1])
    for ci, cond in enumerate(["correct", "shuffled", "zeroed"]):
        vals = [scene_results_e0[st][f"{cond}_mse"] for st in types]
        offset = (ci - 1) * 0.25
        ax2.bar(x + offset, vals, width=0.22, color=colors_cond[cond], label=cond, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{st}\n(n={scene_results_e0[st]['n']})" for st in types], fontsize=9)
    ax2.set_ylabel("MSE")
    ax2.set_title("E0: Absolute MSE by Scene Type & Condition")
    ax2.legend(fontsize=8)

    # ── Row 2: Rollout MSE per step, one line per scene type ──
    rollout_types = [st for st in unique_rollout if rollout_scene[st]["n"] >= 5]
    scene_colors = {"straight": "#2ecc71", "curve": "#e74c3c", "accel/decel": "#f39c12",
                    "stop": "#9b59b6"}
    times = np.arange(1, n_rollout + 1) * 0.5

    ax3 = fig.add_subplot(gs[1, 0])
    for st in rollout_types:
        c = scene_colors.get(st, "#333333")
        ax3.plot(times, rollout_scene[st]["correct"], "o-", color=c, label=f"{st} (correct)", linewidth=2)
        ax3.plot(times, rollout_scene[st]["shuffled"], "s--", color=c, label=f"{st} (shuffled)",
                 linewidth=1.5, alpha=0.7)
    ax3.set_xlabel("Horizon (seconds)")
    ax3.set_ylabel("MSE")
    ax3.set_title("Rollout MSE: Correct vs Shuffled by Scene Type")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── Row 2, Right: Rollout shuffled gap per step per scene type ──
    ax4 = fig.add_subplot(gs[1, 1])
    for st in rollout_types:
        c = scene_colors.get(st, "#333333")
        gap = (rollout_scene[st]["shuffled"] - rollout_scene[st]["correct"]) / \
              rollout_scene[st]["correct"] * 100
        ax4.plot(times, gap, "o-", color=c, label=f"{st}", linewidth=2)
    ax4.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Horizon (seconds)")
    ax4.set_ylabel("Shuffled Gap (%)")
    ax4.set_title("Rollout: Shuffled Gap by Scene Type & Horizon")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Row 3, Left: Rollout correct vs zeroed ──
    ax5 = fig.add_subplot(gs[2, 0])
    for st in rollout_types:
        c = scene_colors.get(st, "#333333")
        ax5.plot(times, rollout_scene[st]["correct"], "o-", color=c, label=f"{st} (correct)", linewidth=2)
        ax5.plot(times, rollout_scene[st]["zeroed"], "s--", color=c, label=f"{st} (zeroed)",
                 linewidth=1.5, alpha=0.7)
    ax5.set_xlabel("Horizon (seconds)")
    ax5.set_ylabel("MSE")
    ax5.set_title("Rollout MSE: Correct vs Zeroed by Scene Type")
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # ── Row 3, Right: Rollout zeroed gap per step per scene type ──
    ax6 = fig.add_subplot(gs[2, 1])
    for st in rollout_types:
        c = scene_colors.get(st, "#333333")
        gap = (rollout_scene[st]["zeroed"] - rollout_scene[st]["correct"]) / \
              rollout_scene[st]["correct"] * 100
        ax6.plot(times, gap, "o-", color=c, label=f"{st}", linewidth=2)
    ax6.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax6.set_xlabel("Horizon (seconds)")
    ax6.set_ylabel("Zeroed Gap (%)")
    ax6.set_title("Rollout: Zeroed Gap by Scene Type & Horizon")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle("C3: Scene-Type Stratified Motion Ablation (H4: Action meaning depends on road structure)",
                 fontsize=14, y=0.98)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--data", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument("--n-sequences-rollout", type=int, default=300)
    parser.add_argument("--num-steps-rollout", type=int, default=8)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/c3_scene_stratified.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("C3: Scene-Type Stratified Motion Ablation")
    print(f"  Hypothesis H4: action meaning depends on road structure context")
    print(f"  Model: {args.ckpt}")
    print(f"  Data:  {args.data}")

    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Part 1: Single-step E0 ablation per scene type
    scene_results_e0, unique_e0 = run_e0_stratified(
        model, transform, args.data, args.device, n_sequences=args.n_sequences)

    # Part 2: Multi-step rollout per scene type
    rollout_scene, unique_rollout, n_rollout = run_rollout_stratified(
        model, transform, args.data, args.device,
        n_sequences=args.n_sequences_rollout,
        num_steps=args.num_steps_rollout,
        history_size=args.history_size)

    # Plot
    make_plot(scene_results_e0, unique_e0, rollout_scene, unique_rollout, n_rollout, args.output)

    # Final summary
    print("\n" + "=" * 70)
    print("C3 SUMMARY: Scene-Type Effect on Motion Sensitivity")
    print("=" * 70)
    print("\nE0 (single-step):")
    for st in unique_e0:
        r = scene_results_e0[st]
        print(f"  {st:>12s} (n={r['n']:3d}): shuffled gap = {r['shuffled_gap']:+.1f}%, "
              f"zeroed gap = {r['zeroed_gap']:+.1f}%")

    print("\nRollout (mean across steps):")
    for st in unique_rollout:
        if rollout_scene[st]["n"] >= 5:
            c_mean = rollout_scene[st]["correct"].mean()
            s_mean = rollout_scene[st]["shuffled"].mean()
            z_mean = rollout_scene[st]["zeroed"].mean()
            sg = (s_mean - c_mean) / c_mean * 100
            zg = (z_mean - c_mean) / c_mean * 100
            print(f"  {st:>12s} (n={rollout_scene[st]['n']:3d}): "
                  f"shuffled gap = {sg:+.1f}%, zeroed gap = {zg:+.1f}%")


if __name__ == "__main__":
    main()
