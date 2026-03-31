#!/usr/bin/env python3
"""P0-D: Counterfactual sanity check on the world model.

Verifies the model produces PLAUSIBLE counterfactuals: if we change the action,
does the predicted future change in a consistent, physically meaningful way?

Tests:
- "logged" rollout should be closest to ground truth
- "brake" should diverge increasingly from logged (car decelerates)
- "steer_left" and "steer_right" should diverge in OPPOSITE directions
- "faster" should diverge from logged (car accelerates)
- Divergence should grow with horizon (not be random)
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


def load_long_sequences(h5_path: str, n_sequences: int = 200, frameskip: int = 5,
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
    """Normalize actions with per-column z-score, return normalized array + stats."""
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0, keepdims=True)
    std = flat[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32), mean, std


def create_counterfactual_actions(action_raw: np.ndarray, frameskip: int = 5, action_dim: int = 3):
    """Create counterfactual action variants from raw (unnormalized) actions.

    action_raw: (N, T, frameskip*action_dim) — raw actions before normalization
    Returns dict of counterfactual raw actions (same shape).

    Action format per sub-step: [vx, vy, yaw_rate]
    """
    N, T, flat_dim = action_raw.shape
    # Reshape to (N, T, frameskip, action_dim) for manipulation
    act = action_raw.reshape(N, T, frameskip, action_dim).copy()

    counterfactuals = {}

    # 1. Logged (original) — no change
    counterfactuals["logged"] = action_raw.copy()

    # 2. Brake — reduce vx to 30%, zero vy
    brake = act.copy()
    brake[..., 0] *= 0.3   # vx *= 0.3
    brake[..., 1] = 0.0    # vy = 0
    counterfactuals["brake"] = brake.reshape(N, T, flat_dim)

    # 3. Steer left — add negative yaw offset
    # Compute yaw_rate statistics to determine meaningful offset
    yaw_rates = act[..., 2]  # (N, T, frameskip)
    yaw_std = np.nanstd(yaw_rates)
    yaw_offset = max(yaw_std * 1.5, 0.05)  # at least 0.05 rad/s
    print(f"  Yaw rate std: {yaw_std:.4f}, using offset: {yaw_offset:.4f} rad/s")

    steer_left = act.copy()
    steer_left[..., 2] -= yaw_offset  # negative yaw = left turn
    counterfactuals["steer_left"] = steer_left.reshape(N, T, flat_dim)

    # 4. Steer right — add positive yaw offset
    steer_right = act.copy()
    steer_right[..., 2] += yaw_offset  # positive yaw = right turn
    counterfactuals["steer_right"] = steer_right.reshape(N, T, flat_dim)

    # 5. Faster — increase vx by 1.5x
    faster = act.copy()
    faster[..., 0] *= 1.5   # vx *= 1.5
    counterfactuals["faster"] = faster.reshape(N, T, flat_dim)

    return counterfactuals


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
    """Autoregressive rollout from history embeddings using given action sequence."""
    B = emb_history.size(0)
    total_steps = action_seq.size(1)
    n_rollout = total_steps - history_size

    emb = emb_history.clone().to(device)
    action_seq = action_seq.to(device)

    pred_list = []
    with torch.no_grad():
        for t in range(n_rollout):
            act_window = action_seq[:, t : t + history_size]
            act_emb = model.action_encoder(act_window)
            pred = model.predict(emb[:, -history_size:], act_emb)[:, -1:]
            pred_list.append(pred.cpu())
            emb = torch.cat([emb, pred], dim=1)

    return torch.cat(pred_list, dim=1)  # (B, n_rollout, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--data", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/p0d_counterfactual.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    n_rollout = args.num_steps - args.history_size
    print("=" * 70)
    print("P0-D: Counterfactual Sanity Check")
    print("=" * 70)
    print(f"  History: {args.history_size} frames, Rollout: {n_rollout} steps")
    print(f"  Each step = {5 * 0.1:.1f}s (frameskip=5, 10Hz)")
    print(f"  Total horizon: {n_rollout * 0.5:.1f}s")
    print(f"  Conditions: logged, brake, steer_left, steer_right, faster")

    # Load model
    print(f"\nLoading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Load data
    print(f"Loading {args.n_sequences} sequences ({args.num_steps} steps) from: {args.data}")
    pixels, action_raw = load_long_sequences(args.data, args.n_sequences, num_steps=args.num_steps)
    print(f"  pixels: {pixels.shape}, action: {action_raw.shape}")

    # Print action statistics (raw)
    act_reshaped = action_raw.reshape(-1, 5, 3)  # (N*T, frameskip, action_dim)
    print(f"\n  Raw action statistics (per sub-step):")
    print(f"    vx:       mean={act_reshaped[...,0].mean():.4f}, std={act_reshaped[...,0].std():.4f}")
    print(f"    vy:       mean={act_reshaped[...,1].mean():.4f}, std={act_reshaped[...,1].std():.4f}")
    print(f"    yaw_rate: mean={act_reshaped[...,2].mean():.4f}, std={act_reshaped[...,2].std():.4f}")

    # Create counterfactual actions (on raw, before normalization)
    print("\nCreating counterfactual actions...")
    counterfactuals_raw = create_counterfactual_actions(action_raw)

    # Normalize all conditions with the SAME statistics (from logged)
    _, act_mean, act_std = normalize_actions(action_raw)

    counterfactuals_norm = {}
    for name, act in counterfactuals_raw.items():
        norm = (act - act_mean) / act_std
        counterfactuals_norm[name] = torch.from_numpy(
            np.nan_to_num(norm, 0.0).astype(np.float32)
        )

    # Encode all frames
    print("\nEncoding all frames...")
    pixels_t = torch.from_numpy(pixels)
    gt_emb = encode_all_frames(model, pixels_t, transform, args.device)
    print(f"  gt_emb: {gt_emb.shape}")

    emb_history = gt_emb[:, :args.history_size]
    gt_future = gt_emb[:, args.history_size:]  # (N, n_rollout, D)

    # Run rollouts for each condition
    batch_size = 64
    rollout_embs = {}  # condition -> (N, n_rollout, D)

    for cond_name, act_tensor in counterfactuals_norm.items():
        print(f"\nRollout condition: {cond_name}")
        all_preds = []
        for i in range(0, len(emb_history), batch_size):
            emb_batch = emb_history[i : i + batch_size]
            act_batch = act_tensor[i : i + batch_size]
            preds = rollout_predict(model, emb_batch, act_batch, args.device, args.history_size)
            all_preds.append(preds)
        rollout_embs[cond_name] = torch.cat(all_preds, dim=0)

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # 1. MSE vs ground truth for each condition
    print("\n--- 1. MSE vs Ground Truth (per step) ---")
    mse_vs_gt = {}
    for cond in ["logged", "brake", "steer_left", "steer_right", "faster"]:
        per_step = (rollout_embs[cond] - gt_future).pow(2).mean(dim=-1).mean(dim=0)
        mse_vs_gt[cond] = per_step.numpy()

    header = f"{'Step':>4s} | {'Time':>5s}"
    for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
        header += f" | {c:>12s}"
    print(header)
    print("-" * len(header))
    for t in range(n_rollout):
        row = f"  {t+1:>2d} | {(t+1)*0.5:>4.1f}s"
        for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
            row += f" | {mse_vs_gt[c][t]:>12.6f}"
        print(row)

    print("\nMean MSE vs GT:")
    for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
        mean_val = mse_vs_gt[c].mean()
        ratio = mean_val / mse_vs_gt["logged"].mean() if mse_vs_gt["logged"].mean() > 0 else 0
        print(f"  {c:>12s}: {mean_val:.6f}  (ratio vs logged: {ratio:.2f}x)")

    # Check: logged should be lowest
    logged_mean = mse_vs_gt["logged"].mean()
    logged_wins = all(mse_vs_gt[c].mean() >= logged_mean for c in ["brake", "steer_left", "steer_right", "faster"])
    print(f"\n  SANITY CHECK: Logged has lowest MSE vs GT? {'PASS' if logged_wins else 'FAIL'}")

    # 2. Divergence from logged trajectory
    print("\n--- 2. Divergence from Logged Trajectory (per step) ---")
    div_from_logged = {}
    for cond in ["brake", "steer_left", "steer_right", "faster"]:
        per_step = (rollout_embs[cond] - rollout_embs["logged"]).pow(2).mean(dim=-1).mean(dim=0)
        div_from_logged[cond] = per_step.numpy()

    header = f"{'Step':>4s} | {'Time':>5s}"
    for c in ["brake", "steer_left", "steer_right", "faster"]:
        header += f" | {c:>12s}"
    print(header)
    print("-" * len(header))
    for t in range(n_rollout):
        row = f"  {t+1:>2d} | {(t+1)*0.5:>4.1f}s"
        for c in ["brake", "steer_left", "steer_right", "faster"]:
            row += f" | {div_from_logged[c][t]:>12.6f}"
        print(row)

    # Check: divergence should grow with horizon
    print("\n  SANITY CHECK: Divergence grows with horizon?")
    for cond in ["brake", "steer_left", "steer_right", "faster"]:
        vals = div_from_logged[cond]
        # Check monotonic increase (allow small deviations)
        diffs = np.diff(vals)
        n_increasing = (diffs > 0).sum()
        grows = n_increasing >= len(diffs) * 0.6  # at least 60% increasing
        first_half = vals[:n_rollout // 2].mean()
        second_half = vals[n_rollout // 2:].mean()
        ratio = second_half / first_half if first_half > 0 else 0
        print(f"    {cond:>12s}: {'PASS' if grows else 'FAIL'} "
              f"(increasing steps: {n_increasing}/{len(diffs)}, "
              f"2nd/1st half ratio: {ratio:.2f}x)")

    # 3. Directional consistency: steer_left vs steer_right should be opposite
    print("\n--- 3. Directional Consistency (steer_left vs steer_right) ---")

    # Compute per-sample displacement vectors from logged
    left_disp = rollout_embs["steer_left"] - rollout_embs["logged"]   # (N, n_rollout, D)
    right_disp = rollout_embs["steer_right"] - rollout_embs["logged"]  # (N, n_rollout, D)

    # Cosine similarity between left and right displacements per step
    cos_per_step = []
    for t in range(n_rollout):
        left_v = left_disp[:, t, :]   # (N, D)
        right_v = right_disp[:, t, :]  # (N, D)

        # Per-sample cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(left_v, right_v, dim=-1)  # (N,)
        cos_per_step.append(cos_sim.mean().item())

    print(f"  Cosine similarity between steer_left and steer_right displacements:")
    for t in range(n_rollout):
        print(f"    Step {t+1} ({(t+1)*0.5:.1f}s): cos_sim = {cos_per_step[t]:+.4f}")

    mean_cos = np.mean(cos_per_step)
    print(f"\n  Mean cosine similarity: {mean_cos:+.4f}")
    print(f"  SANITY CHECK: Left/Right in opposite directions (cos < 0)? "
          f"{'PASS' if mean_cos < 0 else 'FAIL'}")

    # Also check: how much of the time are they anti-correlated per sample?
    all_cos = []
    for t in range(n_rollout):
        left_v = left_disp[:, t, :]
        right_v = right_disp[:, t, :]
        cos_sim = torch.nn.functional.cosine_similarity(left_v, right_v, dim=-1)
        all_cos.append(cos_sim)
    all_cos_cat = torch.cat(all_cos)
    frac_negative = (all_cos_cat < 0).float().mean().item()
    print(f"  Fraction of (sample, step) pairs with negative cosine: {frac_negative:.1%}")

    # 4. Ranking: can we use the model to distinguish correct actions?
    print("\n--- 4. Planning Ranking (does logged action win?) ---")
    # For each sequence, rank conditions by MSE vs ground truth at last step
    last_step = n_rollout - 1
    rankings = {c: [] for c in ["logged", "brake", "steer_left", "steer_right", "faster"]}
    logged_rank_1_count = 0

    for i in range(args.n_sequences):
        costs = {}
        for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
            cost = (rollout_embs[c][i, last_step] - gt_future[i, last_step]).pow(2).mean().item()
            costs[c] = cost
        sorted_conds = sorted(costs.keys(), key=lambda x: costs[x])
        for rank, c in enumerate(sorted_conds):
            rankings[c].append(rank + 1)
        if sorted_conds[0] == "logged":
            logged_rank_1_count += 1

    print(f"  At final step ({(last_step+1)*0.5:.1f}s horizon):")
    for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
        avg_rank = np.mean(rankings[c])
        rank1_pct = sum(1 for r in rankings[c] if r == 1) / len(rankings[c]) * 100
        print(f"    {c:>12s}: avg_rank={avg_rank:.2f}, rank_1={rank1_pct:.1f}%")

    logged_rank1_pct = logged_rank_1_count / args.n_sequences * 100
    print(f"\n  SANITY CHECK: Logged action ranked #1? {logged_rank1_pct:.1f}% of sequences "
          f"({'PASS' if logged_rank1_pct > 30 else 'MARGINAL' if logged_rank1_pct > 20 else 'FAIL'}, "
          f"random=20%)")

    # 5. Effect magnitude summary
    print("\n--- 5. Effect Magnitude Summary ---")
    for cond in ["brake", "steer_left", "steer_right", "faster"]:
        total_div = div_from_logged[cond].sum()
        growth_rate = div_from_logged[cond][-1] / div_from_logged[cond][0] if div_from_logged[cond][0] > 0 else 0
        print(f"  {cond:>12s}: total_div={total_div:.6f}, growth_rate={growth_rate:.2f}x (last/first)")

    # ===== PLOTTING =====
    print("\n\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    steps = np.arange(1, n_rollout + 1)
    times = steps * 0.5

    colors = {
        "logged": "#2ecc71",
        "brake": "#e74c3c",
        "steer_left": "#3498db",
        "steer_right": "#e67e22",
        "faster": "#9b59b6",
    }

    # 1. MSE vs ground truth
    ax = axes[0, 0]
    for c in ["logged", "brake", "steer_left", "steer_right", "faster"]:
        ax.plot(times, mse_vs_gt[c], "o-", color=colors[c], label=c, linewidth=2)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("MSE vs Ground Truth")
    ax.set_title("Prediction Error vs Ground Truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Divergence from logged
    ax = axes[0, 1]
    for c in ["brake", "steer_left", "steer_right", "faster"]:
        ax.plot(times, div_from_logged[c], "o-", color=colors[c], label=c, linewidth=2)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("MSE vs Logged Rollout")
    ax.set_title("Counterfactual Divergence from Logged")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Directional consistency (cosine similarity)
    ax = axes[0, 2]
    ax.plot(times, cos_per_step, "o-", color="#2c3e50", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7)
    ax.fill_between(times, cos_per_step, 0, alpha=0.2,
                     color="#e74c3c" if mean_cos < 0 else "#3498db")
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Left vs Right Displacement Direction\n(negative = opposite = good)")
    ax.grid(True, alpha=0.3)

    # 4. MSE gap vs logged (%)
    ax = axes[1, 0]
    for c in ["brake", "steer_left", "steer_right", "faster"]:
        gap = (mse_vs_gt[c] - mse_vs_gt["logged"]) / mse_vs_gt["logged"] * 100
        ax.plot(times, gap, "o-", color=colors[c], label=c, linewidth=2)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("MSE Gap vs Logged (%)")
    ax.set_title("Counterfactual Penalty (higher = worse than logged)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Ranking distribution
    ax = axes[1, 1]
    conds_list = ["logged", "brake", "steer_left", "steer_right", "faster"]
    avg_ranks = [np.mean(rankings[c]) for c in conds_list]
    bars = ax.bar(range(len(conds_list)), avg_ranks, color=[colors[c] for c in conds_list])
    ax.set_xticks(range(len(conds_list)))
    ax.set_xticklabels(conds_list, rotation=30, ha="right")
    ax.set_ylabel("Average Rank (lower = better)")
    ax.set_title(f"Planning Ranking at {n_rollout*0.5:.1f}s Horizon")
    ax.axhline(y=3, color="gray", linestyle=":", alpha=0.5, label="random=3.0")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Divergence growth rate
    ax = axes[1, 2]
    for c in ["brake", "steer_left", "steer_right", "faster"]:
        cum_div = np.cumsum(div_from_logged[c])
        ax.plot(times, cum_div, "o-", color=colors[c], label=c, linewidth=2)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Cumulative Divergence")
    ax.set_title("Cumulative Counterfactual Divergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("P0-D: Counterfactual Sanity Check — World Model Plausibility", fontsize=14, y=1.02)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("P0-D FINAL SUMMARY")
    print("=" * 70)
    checks = {
        "Logged closest to GT": logged_wins,
        "Divergence grows with horizon": all(
            div_from_logged[c][-1] > div_from_logged[c][0]
            for c in ["brake", "steer_left", "steer_right", "faster"]
        ),
        "Left/Right opposite direction": mean_cos < 0,
        "Logged ranked #1 > random (20%)": logged_rank1_pct > 20,
    }
    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check_name}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
