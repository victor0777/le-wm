#!/usr/bin/env python3
"""Track 1: Corridor Planning proof-of-concept (ADR-008).

Proves that the LeWM world model can be used for short-horizon planning
on known routes via MPC/CEM. Given a goal frame ~2.5s ahead, we generate
action candidates, rollout each through the model, and score by MSE to
the goal embedding. If the logged (ground-truth) action consistently ranks
near the top, the model has learned a useful planning signal.

Also implements CEM (Cross-Entropy Method) refinement to show iterative
improvement over random sampling.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

# For torch.load compatibility with VP-supervised checkpoints
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
    to_image = spt.data.transforms.ToImage(
        **imagenet_stats, source="pixels", target="pixels"
    )
    resize = spt.data.transforms.Resize(img_size, source="pixels", target="pixels")
    return spt.data.transforms.Compose(to_image, resize)


def load_sequences(h5_path: str, n_sequences: int = 100, frameskip: int = 5,
                   num_steps: int = 8, seed: int = 42):
    """Load sequences for planning evaluation.

    Returns pixels (N, num_steps, H, W, C) and action (N, num_steps, frameskip*action_dim).
    """
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
            action_raw = f["action"][s: s + raw_len]
            action_dim = action_raw.shape[-1]
            action_list.append(action_raw.reshape(num_steps, frameskip * action_dim))

    pixels = np.stack(pixels_list)
    action = np.stack(action_list)
    return pixels, action


def normalize_actions(action: np.ndarray):
    """Z-normalize actions, return (normalized, mean, std) for generation."""
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0)
    std = flat[mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32), mean, std


def encode_frames(model, pixels_t, transform, device, batch_size=32):
    """Encode all frames to get ground truth embeddings."""
    all_emb = []
    for i in range(0, len(pixels_t), batch_size):
        px_batch = pixels_t[i: i + batch_size]
        batch = {"pixels": px_batch}
        batch = transform(batch)
        px = batch["pixels"].to(device)
        with torch.no_grad():
            dummy_act = torch.zeros(px.size(0), px.size(1), 15, device=device)
            info = {"pixels": px, "action": dummy_act}
            info = model.encode(info)
        all_emb.append(info["emb"].cpu())
    return torch.cat(all_emb, dim=0)  # (N, T, D)


def rollout_predict(model, emb_history, action_seq, device, history_size=3):
    """Autoregressive rollout from history embeddings.

    Args:
        emb_history: (B, history_size, D)
        action_seq:  (B, total_steps, action_dim)

    Returns:
        pred_embs: (B, n_rollout_steps, D)
    """
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


def generate_candidates(logged_action, n_candidates=64, noise_std=0.5, seed=None):
    """Generate action candidates around logged action.

    Args:
        logged_action: (T, action_dim) — normalized logged action for full sequence
        n_candidates: total number of candidates (including logged)

    Returns:
        candidates: (n_candidates, T, action_dim)
        labels: list of str describing each candidate
    """
    rng = np.random.RandomState(seed)
    T, D = logged_action.shape
    candidates = []
    labels = []

    # 0: logged action (ground truth)
    candidates.append(logged_action.copy())
    labels.append("logged")

    # 1-31: gaussian perturbations of logged action (small noise)
    n_small = 15
    for i in range(n_small):
        noise = rng.randn(T, D).astype(np.float32) * noise_std * 0.5
        candidates.append(logged_action + noise)
        labels.append(f"gauss_small_{i}")

    # 32-47: larger perturbations
    n_large = 16
    for i in range(n_large):
        noise = rng.randn(T, D).astype(np.float32) * noise_std * 1.5
        candidates.append(logged_action + noise)
        labels.append(f"gauss_large_{i}")

    # Structured alternatives
    # Brake: reduce vx components (indices 0,3,6,9,12 for frameskip=5, action_dim=3)
    brake = logged_action.copy()
    # vx is at indices 0, 3, 6, 9, 12 in the 15-dim chunked action
    for idx in range(0, D, 3):
        brake[:, idx] *= 0.3  # reduce forward speed
    candidates.append(brake)
    labels.append("brake")

    # Hard brake
    hard_brake = logged_action.copy()
    for idx in range(0, D, 3):
        hard_brake[:, idx] = 0.0
    candidates.append(hard_brake)
    labels.append("hard_brake")

    # Steer left: increase yaw rate
    steer_left = logged_action.copy()
    for idx in range(2, D, 3):
        steer_left[:, idx] += 0.5
    candidates.append(steer_left)
    labels.append("steer_left")

    # Steer right
    steer_right = logged_action.copy()
    for idx in range(2, D, 3):
        steer_right[:, idx] -= 0.5
    candidates.append(steer_right)
    labels.append("steer_right")

    # Accelerate
    accel = logged_action.copy()
    for idx in range(0, D, 3):
        accel[:, idx] *= 1.5
    candidates.append(accel)
    labels.append("accelerate")

    # Zero action (stop)
    candidates.append(np.zeros_like(logged_action))
    labels.append("zero")

    # Random actions to fill up to n_candidates
    n_remaining = n_candidates - len(candidates)
    for i in range(n_remaining):
        random_act = rng.randn(T, D).astype(np.float32) * noise_std
        candidates.append(random_act)
        labels.append(f"random_{i}")

    candidates = np.stack(candidates[:n_candidates])
    labels = labels[:n_candidates]
    return candidates, labels


def score_candidates(model, emb_history, goal_emb, action_candidates, device,
                     history_size=3):
    """Score action candidates by MSE to goal embedding.

    Args:
        emb_history: (history_size, D)
        goal_emb: (D,)
        action_candidates: (N_cand, T, action_dim)

    Returns:
        scores: (N_cand,) — MSE to goal (lower is better)
    """
    N_cand = action_candidates.shape[0]
    emb_h = emb_history.unsqueeze(0).expand(N_cand, -1, -1)  # (N_cand, HS, D)
    act_t = torch.from_numpy(action_candidates).float()

    # Rollout all candidates
    pred_emb = rollout_predict(model, emb_h, act_t, device, history_size)
    # pred_emb: (N_cand, n_rollout, D)

    # Score: MSE of final predicted embedding vs goal
    final_pred = pred_emb[:, -1, :]  # (N_cand, D)
    goal = goal_emb.unsqueeze(0).expand(N_cand, -1)  # (N_cand, D)
    scores = (final_pred - goal).pow(2).mean(dim=-1)  # (N_cand,)
    return scores.numpy()


def cem_optimize(model, emb_history, goal_emb, initial_action, device,
                 history_size=3, n_candidates=64, n_elite=8, n_iterations=5,
                 noise_std=0.5):
    """CEM optimization starting from logged action distribution.

    Returns:
        best_action: (T, action_dim)
        convergence: list of best scores per iteration
        all_scores: list of all scores per iteration
    """
    T, D = initial_action.shape
    mean = initial_action.copy()
    std = np.ones_like(initial_action) * noise_std

    convergence = []
    all_iter_scores = []

    for it in range(n_iterations):
        # Sample candidates
        rng = np.random.RandomState(it * 1000)
        candidates = []
        for _ in range(n_candidates):
            sample = mean + rng.randn(T, D).astype(np.float32) * std
            candidates.append(sample)
        candidates = np.stack(candidates)

        # Score
        scores = score_candidates(model, emb_history, goal_emb, candidates, device,
                                  history_size)
        all_iter_scores.append(scores.copy())

        # Select elite
        elite_idx = np.argsort(scores)[:n_elite]
        elite = candidates[elite_idx]

        # Refit distribution
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6  # prevent collapse

        convergence.append(float(scores[elite_idx[0]]))

    return mean, convergence, all_iter_scores


def main():
    parser = argparse.ArgumentParser(description="Track 1: Corridor Planning")
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--data", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=8,
                        help="Total sequence length (history + rollout)")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-elite", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="outputs/track1_corridor_planning.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    n_rollout = args.num_steps - args.history_size
    goal_step = n_rollout - 1  # last rollout step = goal
    print("=" * 70)
    print("Track 1: Corridor Planning Proof-of-Concept (ADR-008)")
    print("=" * 70)
    print(f"  History: {args.history_size} frames")
    print(f"  Rollout: {n_rollout} steps ({n_rollout * 0.5:.1f}s)")
    print(f"  Goal: frame {args.history_size + n_rollout} = step {n_rollout} ({n_rollout * 0.5:.1f}s ahead)")
    print(f"  Candidates: {args.n_candidates}")
    print(f"  CEM: {args.cem_iterations} iterations, top-{args.cem_elite} elite")
    print()

    # Load model
    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Load data
    print(f"Loading {args.n_sequences} sequences from: {args.data}")
    pixels, action = load_sequences(args.data, args.n_sequences,
                                    num_steps=args.num_steps)
    print(f"  pixels: {pixels.shape}, action: {action.shape}")

    # Normalize actions
    action_norm, act_mean, act_std = normalize_actions(action)
    pixels_t = torch.from_numpy(pixels)
    action_t = torch.from_numpy(action_norm)

    # Encode all frames
    print("\nEncoding all frames...")
    gt_emb = encode_frames(model, pixels_t, transform, args.device)
    print(f"  gt_emb: {gt_emb.shape}")  # (N, T, D)

    # ================================================================
    # Phase 1: Action Candidate Ranking
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Action Candidate Ranking (N={})".format(args.n_candidates))
    print("=" * 70)

    logged_ranks = []
    logged_scores = []
    best_scores = []
    top5_count = 0
    top10_count = 0
    top1_count = 0
    all_score_distributions = []

    for seq_idx in range(args.n_sequences):
        if (seq_idx + 1) % 20 == 0 or seq_idx == 0:
            print(f"  Sequence {seq_idx + 1}/{args.n_sequences}...")

        emb_history = gt_emb[seq_idx, :args.history_size]  # (HS, D)
        goal_emb = gt_emb[seq_idx, -1]  # (D,) — last frame as goal
        logged_act = action_norm[seq_idx]  # (T, action_dim)

        # Generate candidates
        candidates, labels = generate_candidates(
            logged_act, n_candidates=args.n_candidates,
            noise_std=args.noise_std, seed=seq_idx
        )

        # Score all candidates
        scores = score_candidates(model, emb_history, goal_emb, candidates,
                                  args.device, args.history_size)

        # Find logged action rank (index 0 = logged)
        logged_score = scores[0]
        rank = (scores < logged_score).sum() + 1  # 1-indexed rank
        logged_ranks.append(rank)
        logged_scores.append(logged_score)
        best_scores.append(scores.min())
        all_score_distributions.append(scores)

        if rank == 1:
            top1_count += 1
        if rank <= 5:
            top5_count += 1
        if rank <= 10:
            top10_count += 1

    logged_ranks = np.array(logged_ranks)
    logged_scores = np.array(logged_scores)
    best_scores = np.array(best_scores)

    print("\n--- Phase 1 Results ---")
    print(f"  Logged action rank:  mean={logged_ranks.mean():.1f}, "
          f"median={np.median(logged_ranks):.1f}, "
          f"std={logged_ranks.std():.1f}")
    print(f"  Top-1 accuracy:      {top1_count}/{args.n_sequences} "
          f"({top1_count / args.n_sequences * 100:.1f}%)")
    print(f"  Top-5 accuracy:      {top5_count}/{args.n_sequences} "
          f"({top5_count / args.n_sequences * 100:.1f}%)")
    print(f"  Top-10 accuracy:     {top10_count}/{args.n_sequences} "
          f"({top10_count / args.n_sequences * 100:.1f}%)")
    print(f"  Logged action MSE:   mean={logged_scores.mean():.6f}")
    print(f"  Best candidate MSE:  mean={best_scores.mean():.6f}")
    regret = logged_scores - best_scores
    print(f"  Regret (logged-best): mean={regret.mean():.6f}, "
          f"median={np.median(regret):.6f}")
    print(f"  Regret positive (logged worse): {(regret > 0).sum()}/{args.n_sequences}")

    # ================================================================
    # Phase 2: CEM Refinement
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 2: CEM Refinement")
    print("=" * 70)

    cem_best_scores = []
    cem_convergences = []
    cem_vs_logged = []
    cem_vs_random = []

    for seq_idx in range(args.n_sequences):
        if (seq_idx + 1) % 20 == 0 or seq_idx == 0:
            print(f"  CEM sequence {seq_idx + 1}/{args.n_sequences}...")

        emb_history = gt_emb[seq_idx, :args.history_size]
        goal_emb = gt_emb[seq_idx, -1]
        logged_act = action_norm[seq_idx]

        # Run CEM starting from logged action distribution
        best_act, convergence, iter_scores = cem_optimize(
            model, emb_history, goal_emb, logged_act, args.device,
            history_size=args.history_size,
            n_candidates=args.n_candidates,
            n_elite=args.cem_elite,
            n_iterations=args.cem_iterations,
            noise_std=args.noise_std,
        )

        cem_best_scores.append(convergence[-1])
        cem_convergences.append(convergence)

        # Compare CEM best vs logged
        cem_vs_logged.append(convergence[-1] - logged_scores[seq_idx])

        # Compare CEM best vs random sampling best
        cem_vs_random.append(convergence[-1] - convergence[0])

    cem_best_scores = np.array(cem_best_scores)
    cem_convergences = np.array(cem_convergences)  # (N, n_iter)
    cem_vs_logged = np.array(cem_vs_logged)
    cem_vs_random = np.array(cem_vs_random)

    print("\n--- Phase 2 Results ---")
    print(f"  CEM final MSE:       mean={cem_best_scores.mean():.6f}")
    print(f"  Logged action MSE:   mean={logged_scores.mean():.6f}")
    print(f"  CEM vs Logged:       mean={cem_vs_logged.mean():.6f} "
          f"({'better' if cem_vs_logged.mean() < 0 else 'worse'})")
    print(f"  CEM beats logged:    {(cem_vs_logged < 0).sum()}/{args.n_sequences} "
          f"({(cem_vs_logged < 0).sum() / args.n_sequences * 100:.1f}%)")
    print(f"  CEM improvement over iter0: mean={-cem_vs_random.mean():.6f}")
    print(f"  Mean convergence per iteration:")
    mean_conv = cem_convergences.mean(axis=0)
    for it_idx, val in enumerate(mean_conv):
        improvement = (mean_conv[0] - val) / mean_conv[0] * 100 if mean_conv[0] > 0 else 0
        print(f"    Iter {it_idx}: MSE={val:.6f} ({improvement:+.1f}% from iter 0)")

    # ================================================================
    # Phase 3: Visualization
    # ================================================================
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Logged action rank histogram
    ax = axes[0, 0]
    ax.hist(logged_ranks, bins=np.arange(0.5, args.n_candidates + 1.5, 1),
            color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(x=np.median(logged_ranks), color="#e74c3c", linestyle="--",
               linewidth=2, label=f"median={np.median(logged_ranks):.0f}")
    ax.axvline(x=logged_ranks.mean(), color="#2ecc71", linestyle="--",
               linewidth=2, label=f"mean={logged_ranks.mean():.1f}")
    ax.set_xlabel("Rank of Logged Action")
    ax.set_ylabel("Count")
    ax.set_title(f"Logged Action Rank Distribution (N={args.n_candidates} candidates)")
    ax.legend()
    ax.set_xlim(0, min(args.n_candidates + 1, max(logged_ranks) + 5))

    # 2. Score distribution (aggregated)
    ax = axes[0, 1]
    all_scores_flat = np.concatenate(all_score_distributions)
    ax.hist(all_scores_flat, bins=50, color="#95a5a6", alpha=0.5, label="all candidates", density=True)
    ax.hist(logged_scores, bins=30, color="#2ecc71", alpha=0.7, label="logged action", density=True)
    ax.hist(best_scores, bins=30, color="#e74c3c", alpha=0.7, label="best candidate", density=True)
    ax.set_xlabel("MSE to Goal Embedding")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions")
    ax.legend()

    # 3. CEM convergence
    ax = axes[1, 0]
    iters = np.arange(args.cem_iterations)
    mean_conv = cem_convergences.mean(axis=0)
    std_conv = cem_convergences.std(axis=0)
    ax.plot(iters, mean_conv, "o-", color="#e74c3c", linewidth=2, label="CEM best")
    ax.fill_between(iters, mean_conv - std_conv, mean_conv + std_conv,
                    color="#e74c3c", alpha=0.15)
    ax.axhline(y=logged_scores.mean(), color="#2ecc71", linestyle="--",
               linewidth=2, label=f"logged mean={logged_scores.mean():.4f}")
    ax.set_xlabel("CEM Iteration")
    ax.set_ylabel("Best MSE")
    ax.set_title("CEM Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CEM vs Logged scatter
    ax = axes[1, 1]
    ax.scatter(logged_scores, cem_best_scores, alpha=0.5, s=20, c="#3498db")
    lim_max = max(logged_scores.max(), cem_best_scores.max()) * 1.05
    lim_min = 0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Logged Action MSE")
    ax.set_ylabel("CEM Best MSE")
    ax.set_title(f"CEM vs Logged (below line = CEM wins, "
                 f"{(cem_vs_logged < 0).sum()}/{args.n_sequences})")
    ax.legend()
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Track 1: Corridor Planning — MPC/CEM over LeWM (ADR-008)\n"
                 f"Model: A1 (Livlab overlap) | Holdout: Livlab-Rt-C-7 | "
                 f"Candidates: {args.n_candidates} | CEM: {args.cem_iterations} iters",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Track 1 Corridor Planning")
    print("=" * 70)
    print(f"  Sequences evaluated:     {args.n_sequences}")
    print(f"  Action candidates:       {args.n_candidates}")
    print(f"  Planning horizon:        {n_rollout * 0.5:.1f}s ({n_rollout} steps)")
    print()
    print("  [Ranking]")
    print(f"    Logged rank mean:      {logged_ranks.mean():.1f} / {args.n_candidates}")
    print(f"    Logged rank median:    {np.median(logged_ranks):.1f} / {args.n_candidates}")
    print(f"    Top-1:                 {top1_count / args.n_sequences * 100:.1f}%")
    print(f"    Top-5:                 {top5_count / args.n_sequences * 100:.1f}%")
    print(f"    Top-10:                {top10_count / args.n_sequences * 100:.1f}%")
    print()
    print("  [Regret]")
    print(f"    Mean regret:           {regret.mean():.6f}")
    print(f"    Median regret:         {np.median(regret):.6f}")
    print()
    print("  [CEM]")
    print(f"    CEM final MSE:         {cem_best_scores.mean():.6f}")
    print(f"    Logged MSE:            {logged_scores.mean():.6f}")
    print(f"    CEM beats logged:      {(cem_vs_logged < 0).sum() / args.n_sequences * 100:.1f}%")
    print(f"    CEM improvement:       {-cem_vs_random.mean():.6f} over iter 0")
    print()

    # Verdict
    if logged_ranks.mean() <= args.n_candidates * 0.25:
        verdict = "PASS — logged action ranks in top quartile"
    elif logged_ranks.mean() <= args.n_candidates * 0.5:
        verdict = "PARTIAL — logged action ranks above median"
    else:
        verdict = "FAIL — logged action does not rank well"

    if (cem_vs_logged < 0).sum() / args.n_sequences > 0.5:
        cem_verdict = "CEM finds better actions than logged in majority of cases"
    else:
        cem_verdict = "CEM does not consistently beat logged action"

    print(f"  VERDICT: {verdict}")
    print(f"  CEM:     {cem_verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
