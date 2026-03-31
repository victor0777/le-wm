#!/usr/bin/env python3
"""C2: Future action hint experiment (ADR-007).

Tests whether the model can leverage action information over multi-step
autoregressive rollouts. If the gap between correct and shuffled actions
GROWS over horizon steps, the model IS using action and the effect accumulates.
If it stays flat, action is ignored at longer horizons.

This tests: "Does the model have the CAPACITY to use action, but lacks CONTEXT?"

No retraining needed — uses existing model's rollout capability.
"""

import argparse
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


def load_long_sequences(h5_path: str, n_sequences: int = 300, frameskip: int = 5,
                        num_steps: int = 8, seed: int = 42):
    """Load longer sequences for multi-step rollout evaluation."""
    rng = np.random.RandomState(seed)
    raw_len = num_steps * frameskip

    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        max_start = total - raw_len
        if max_start <= 0:
            raise ValueError(f"Not enough frames: {total} < {raw_len}")
        starts = rng.randint(0, max_start, size=n_sequences)

        pixels_list, action_list = [], []
        for s in starts:
            px_indices = list(range(s, s + raw_len, frameskip))
            pixels_list.append(f["pixels"][px_indices])
            action_raw = f["action"][s : s + raw_len]
            action_dim = action_raw.shape[-1]
            action_list.append(action_raw.reshape(num_steps, frameskip * action_dim))

    pixels = np.stack(pixels_list)
    action = np.stack(action_list)
    return pixels, action


def normalize_actions(action: np.ndarray):
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0, keepdims=True)
    std = flat[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32)


def encode_all_frames(model, pixels_t, transform, device, batch_size=32):
    """Encode all frames to get ground truth embeddings."""
    all_emb = []
    for i in range(0, len(pixels_t), batch_size):
        px_batch = pixels_t[i : i + batch_size]
        batch = {"pixels": px_batch}
        batch = transform(batch)
        px = batch["pixels"].to(device)
        with torch.no_grad():
            info = {"pixels": px, "action": torch.zeros(px.size(0), px.size(1), 15, device=device)}
            info = model.encode(info)
        all_emb.append(info["emb"].cpu())
    return torch.cat(all_emb, dim=0)  # (N, T, D)


def rollout_predict(model, emb_history, action_seq, device, history_size=3):
    """Autoregressive rollout from history embeddings using given action sequence.

    Args:
        emb_history: (B, history_size, D) — ground truth embeddings for context
        action_seq: (B, total_steps, action_dim) — actions for full sequence
        history_size: number of context frames

    Returns:
        pred_embs: (B, n_rollout_steps, D) — predicted embeddings at each future step
    """
    B = emb_history.size(0)
    total_steps = action_seq.size(1)
    n_rollout = total_steps - history_size

    emb = emb_history.clone().to(device)  # (B, HS, D)
    action_seq = action_seq.to(device)

    pred_list = []
    with torch.no_grad():
        for t in range(n_rollout):
            # Action for current context window
            act_window = action_seq[:, t : t + history_size]  # (B, HS, action_dim)
            act_emb = model.action_encoder(act_window)

            # Predict next
            pred = model.predict(emb[:, -history_size:], act_emb)[:, -1:]  # (B, 1, D)
            pred_list.append(pred.cpu())

            # Append prediction for next step (autoregressive)
            emb = torch.cat([emb, pred], dim=1)

    return torch.cat(pred_list, dim=1)  # (B, n_rollout, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--data", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--n-sequences", type=int, default=300)
    parser.add_argument("--num-steps", type=int, default=8,
                        help="Total sequence length (history + rollout)")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/c2_future_action_hint.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    n_rollout = args.num_steps - args.history_size
    print(f"C2: Future Action Hint Experiment")
    print(f"  History: {args.history_size} frames, Rollout: {n_rollout} steps")
    print(f"  Each step = {5 * 0.1:.1f}s (frameskip=5, 10Hz)")
    print(f"  Total horizon: {n_rollout * 0.5:.1f}s")

    print(f"\nLoading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    print(f"Loading {args.n_sequences} sequences ({args.num_steps} steps) from: {args.data}")
    pixels, action = load_long_sequences(args.data, args.n_sequences, num_steps=args.num_steps)
    print(f"  pixels: {pixels.shape}, action: {action.shape}")

    action_norm = normalize_actions(action)
    pixels_t = torch.from_numpy(pixels)
    action_correct = torch.from_numpy(action_norm)

    # Shuffled and zeroed conditions
    rng = np.random.RandomState(123)
    shuffle_idx = rng.permutation(len(action_norm))
    action_shuffled = torch.from_numpy(action_norm[shuffle_idx])
    action_zeroed = torch.zeros_like(action_correct)

    # Encode all frames to get ground truth embeddings
    print("\nEncoding all frames...")
    gt_emb = encode_all_frames(model, pixels_t, transform, args.device)
    print(f"  gt_emb: {gt_emb.shape}")

    # Extract history embeddings (ground truth)
    emb_history = gt_emb[:, :args.history_size]  # (N, HS, D)

    # Run rollout for each condition
    conditions = {
        "correct": action_correct,
        "shuffled": action_shuffled,
        "zeroed": action_zeroed,
    }

    # Per-step MSE results
    results = {}
    batch_size = 64

    for cond_name, act_tensor in conditions.items():
        print(f"\nRollout condition: {cond_name}")
        all_preds = []
        for i in range(0, len(emb_history), batch_size):
            emb_batch = emb_history[i : i + batch_size]
            act_batch = act_tensor[i : i + batch_size]
            preds = rollout_predict(model, emb_batch, act_batch, args.device, args.history_size)
            all_preds.append(preds)
        pred_emb = torch.cat(all_preds, dim=0)  # (N, n_rollout, D)

        # Ground truth for rollout steps
        tgt_emb = gt_emb[:, args.history_size:]  # (N, n_rollout, D)

        # Per-step MSE
        per_step_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1).mean(dim=0)  # (n_rollout,)
        results[cond_name] = per_step_mse.numpy()

    # Print results
    print("\n" + "=" * 70)
    print("C2: Future Action Hint — Per-Step Results")
    print("=" * 70)
    print(f"{'Step':>4s} | {'Time (s)':>8s} | {'Correct':>10s} | {'Shuffled':>10s} | {'Zeroed':>10s} | {'Shuf Gap':>9s} | {'Zero Gap':>9s}")
    print("-" * 70)

    for t in range(n_rollout):
        time_s = (t + 1) * 0.5
        c = results["correct"][t]
        s = results["shuffled"][t]
        z = results["zeroed"][t]
        sg = (s - c) / c * 100 if c > 0 else 0
        zg = (z - c) / c * 100 if c > 0 else 0
        print(f"  {t+1:>2d}  | {time_s:>7.1f}s | {c:>10.6f} | {s:>10.6f} | {z:>10.6f} | {sg:>+8.1f}% | {zg:>+8.1f}%")

    # Overall summary
    print("\nOverall (mean across steps):")
    for cond_name in ["correct", "shuffled", "zeroed"]:
        mean_mse = results[cond_name].mean()
        print(f"  {cond_name:10s}: MSE = {mean_mse:.6f}")

    c_mean = results["correct"].mean()
    for cond in ["shuffled", "zeroed"]:
        gap = (results[cond].mean() - c_mean) / c_mean * 100
        print(f"  {cond} vs correct: MSE {gap:+.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    steps = np.arange(1, n_rollout + 1)
    times = steps * 0.5
    colors = {"correct": "#2ecc71", "shuffled": "#e74c3c", "zeroed": "#3498db"}

    # 1. Absolute MSE per step
    ax = axes[0]
    for cond_name in ["correct", "shuffled", "zeroed"]:
        ax.plot(times, results[cond_name], "o-", color=colors[cond_name], label=cond_name, linewidth=2)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("MSE")
    ax.set_title("Prediction MSE per Horizon Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Shuffled gap per step
    ax = axes[1]
    shuf_gap = (results["shuffled"] - results["correct"]) / results["correct"] * 100
    zero_gap = (results["zeroed"] - results["correct"]) / results["correct"] * 100
    ax.plot(times, shuf_gap, "o-", color="#e74c3c", label="shuffled gap", linewidth=2)
    ax.plot(times, zero_gap, "s--", color="#3498db", label="zeroed gap", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Gap vs Correct (%)")
    ax.set_title("Motion Sensitivity by Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative MSE
    ax = axes[2]
    for cond_name in ["correct", "shuffled", "zeroed"]:
        cum_mse = np.cumsum(results[cond_name])
        ax.plot(times, cum_mse, "o-", color=colors[cond_name], label=cond_name, linewidth=2)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Cumulative MSE")
    ax.set_title("Cumulative Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("C2: Future Action Hint — Multi-Step Rollout (ADR-007)", fontsize=14, y=1.02)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
