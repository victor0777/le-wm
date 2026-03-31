#!/usr/bin/env python3
"""Track 1 v2: Improved Corridor Planning with Composite Cost (ADR-008).

Instead of raw MSE-to-goal, uses a composite cost:
  1. Progress cost: cosine distance to goal embedding
  2. Smoothness cost: MSE between consecutive predicted embeddings
  3. Consistency cost: variance of predicted embeddings
  4. Action regularization: penalize extreme actions

Also tests a relative ranking approach using embedding trajectory shape.
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

try:
    from train_vp import LaneDecoder, DepthDecoder  # noqa: F401
except ImportError:
    pass


# ================================================================
# Shared utilities (same as v1)
# ================================================================

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
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0)
    std = flat[mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32), mean, std


def encode_frames(model, pixels_t, transform, device, batch_size=32):
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


def generate_candidates(logged_action, n_candidates=64, noise_std=0.5, seed=None):
    rng = np.random.RandomState(seed)
    T, D = logged_action.shape
    candidates = []
    labels = []

    # 0: logged action (ground truth)
    candidates.append(logged_action.copy())
    labels.append("logged")

    # 1-15: small gaussian perturbations
    for i in range(15):
        noise = rng.randn(T, D).astype(np.float32) * noise_std * 0.5
        candidates.append(logged_action + noise)
        labels.append(f"gauss_small_{i}")

    # 16-31: larger perturbations
    for i in range(16):
        noise = rng.randn(T, D).astype(np.float32) * noise_std * 1.5
        candidates.append(logged_action + noise)
        labels.append(f"gauss_large_{i}")

    # Structured alternatives
    brake = logged_action.copy()
    for idx in range(0, D, 3):
        brake[:, idx] *= 0.3
    candidates.append(brake)
    labels.append("brake")

    hard_brake = logged_action.copy()
    for idx in range(0, D, 3):
        hard_brake[:, idx] = 0.0
    candidates.append(hard_brake)
    labels.append("hard_brake")

    steer_left = logged_action.copy()
    for idx in range(2, D, 3):
        steer_left[:, idx] += 0.5
    candidates.append(steer_left)
    labels.append("steer_left")

    steer_right = logged_action.copy()
    for idx in range(2, D, 3):
        steer_right[:, idx] -= 0.5
    candidates.append(steer_right)
    labels.append("steer_right")

    accel = logged_action.copy()
    for idx in range(0, D, 3):
        accel[:, idx] *= 1.5
    candidates.append(accel)
    labels.append("accelerate")

    candidates.append(np.zeros_like(logged_action))
    labels.append("zero")

    n_remaining = n_candidates - len(candidates)
    for i in range(n_remaining):
        random_act = rng.randn(T, D).astype(np.float32) * noise_std
        candidates.append(random_act)
        labels.append(f"random_{i}")

    candidates = np.stack(candidates[:n_candidates])
    labels = labels[:n_candidates]
    return candidates, labels


# ================================================================
# V2: Composite cost functions
# ================================================================

def compute_composite_cost(pred_emb, goal_emb, emb_history, action_candidates,
                           weights, gt_trajectory_emb=None):
    """Compute composite cost for action candidates.

    Args:
        pred_emb: (N_cand, n_rollout, D) — predicted embeddings from rollout
        goal_emb: (D,) — goal embedding (last frame)
        emb_history: (history_size, D) — history embeddings
        action_candidates: (N_cand, T, action_dim) — action candidates (numpy)
        weights: dict with w_progress, w_smoothness, w_consistency, w_action_reg
        gt_trajectory_emb: (n_rollout, D) — ground truth trajectory embeddings (optional, for ranking)

    Returns:
        total_cost: (N_cand,) numpy
        cost_components: dict of (N_cand,) numpy arrays
    """
    N_cand, n_rollout, D = pred_emb.shape
    goal = goal_emb.unsqueeze(0).expand(N_cand, -1)  # (N_cand, D)

    components = {}

    # 1. Progress cost: cosine distance between final predicted embedding and goal
    final_pred = pred_emb[:, -1, :]  # (N_cand, D)
    cos_sim = F.cosine_similarity(final_pred, goal, dim=-1)  # (N_cand,)
    progress_cost = 1.0 - cos_sim  # cosine distance: 0 = identical, 2 = opposite
    components["progress"] = progress_cost.numpy()

    # Also compute per-step progress: are we consistently moving toward the goal?
    # Weighted more heavily for later steps (should be closer to goal)
    step_weights = torch.linspace(0.2, 1.0, n_rollout)  # later steps matter more
    step_weights = step_weights / step_weights.sum()
    for t in range(n_rollout):
        step_cos = F.cosine_similarity(pred_emb[:, t, :], goal, dim=-1)
        if t == 0:
            weighted_progress = step_weights[t] * (1.0 - step_cos)
        else:
            weighted_progress = weighted_progress + step_weights[t] * (1.0 - step_cos)
    # Blend final-step and weighted progress
    progress_cost = 0.5 * progress_cost + 0.5 * weighted_progress
    components["progress"] = progress_cost.numpy()

    # 2. Smoothness cost: MSE between consecutive predicted embeddings
    # Penalizes "jumpy" trajectories
    if n_rollout > 1:
        diffs = pred_emb[:, 1:, :] - pred_emb[:, :-1, :]  # (N_cand, n_rollout-1, D)
        smoothness_cost = diffs.pow(2).mean(dim=-1).mean(dim=-1)  # (N_cand,)
    else:
        smoothness_cost = torch.zeros(N_cand)
    components["smoothness"] = smoothness_cost.numpy()

    # 3. Consistency cost: variance of embedding trajectory
    # Low variance = stable predictions, high variance = erratic
    emb_mean = pred_emb.mean(dim=1, keepdim=True)  # (N_cand, 1, D)
    consistency_cost = (pred_emb - emb_mean).pow(2).mean(dim=-1).mean(dim=-1)  # (N_cand,)
    components["consistency"] = consistency_cost.numpy()

    # 4. Action regularization: penalize extreme actions
    act_t = torch.from_numpy(action_candidates).float()  # (N_cand, T, action_dim)
    # Penalize large magnitude
    action_mag = act_t.pow(2).mean(dim=-1).mean(dim=-1)  # (N_cand,)
    # Penalize large changes between steps
    if act_t.size(1) > 1:
        act_diffs = act_t[:, 1:, :] - act_t[:, :-1, :]
        action_jerk = act_diffs.pow(2).mean(dim=-1).mean(dim=-1)
    else:
        action_jerk = torch.zeros(N_cand)
    action_reg = 0.5 * action_mag + 0.5 * action_jerk
    components["action_reg"] = action_reg.numpy()

    # Composite cost
    total = (weights["w_progress"] * progress_cost +
             weights["w_smoothness"] * smoothness_cost +
             weights["w_consistency"] * consistency_cost +
             weights["w_action_reg"] * action_reg)
    components["total"] = total.numpy()

    return total.numpy(), components


def compute_trajectory_shape_cost(pred_emb, gt_trajectory_emb, emb_history):
    """Relative ranking: compare trajectory shape (direction of change) rather than absolute position.

    Computes cosine similarity between the delta-sequence of the candidate
    and the delta-sequence of the ground truth trajectory.

    Args:
        pred_emb: (N_cand, n_rollout, D)
        gt_trajectory_emb: (n_rollout, D) — ground truth embeddings for rollout steps
        emb_history: (history_size, D)

    Returns:
        shape_cost: (N_cand,) — lower is better (more similar trajectory shape)
    """
    N_cand, n_rollout, D = pred_emb.shape

    # Compute deltas for ground truth: transition from last history frame to each rollout step
    last_history = emb_history[-1:, :]  # (1, D)

    # GT deltas: frame-to-frame changes
    gt_full = torch.cat([last_history, gt_trajectory_emb], dim=0)  # (n_rollout+1, D)
    gt_deltas = gt_full[1:] - gt_full[:-1]  # (n_rollout, D)

    # Candidate deltas
    last_h_expanded = last_history.unsqueeze(0).expand(N_cand, -1, -1)  # (N_cand, 1, D)
    pred_full = torch.cat([last_h_expanded, pred_emb], dim=1)  # (N_cand, n_rollout+1, D)
    pred_deltas = pred_full[:, 1:] - pred_full[:, :-1]  # (N_cand, n_rollout, D)

    # Cosine similarity between delta sequences at each step
    gt_deltas_exp = gt_deltas.unsqueeze(0).expand(N_cand, -1, -1)  # (N_cand, n_rollout, D)

    # Per-step cosine similarity
    cos_per_step = F.cosine_similarity(pred_deltas, gt_deltas_exp, dim=-1)  # (N_cand, n_rollout)

    # Weight later steps more
    step_weights = torch.linspace(0.5, 1.5, n_rollout)
    step_weights = step_weights / step_weights.sum()
    weighted_cos = (cos_per_step * step_weights.unsqueeze(0)).sum(dim=-1)  # (N_cand,)

    # Also compare magnitude ratio (are deltas similar in size?)
    gt_norms = gt_deltas.norm(dim=-1, keepdim=True).unsqueeze(0).expand(N_cand, -1, -1)  # (N_cand, n_rollout, 1)
    pred_norms = pred_deltas.norm(dim=-1, keepdim=True)  # (N_cand, n_rollout, 1)
    # Ratio penalty: |log(pred_norm / gt_norm)|, clamped
    eps = 1e-8
    norm_ratio = (pred_norms / (gt_norms + eps) + eps).log().abs().squeeze(-1)  # (N_cand, n_rollout)
    magnitude_penalty = (norm_ratio * step_weights.unsqueeze(0)).sum(dim=-1)  # (N_cand,)

    # Shape cost: low cosine sim + magnitude mismatch = bad
    shape_cost = (1.0 - weighted_cos) + 0.3 * magnitude_penalty

    return shape_cost.numpy()


def score_candidates_v2(model, emb_history, goal_emb, action_candidates, device,
                        history_size=3, weights=None, gt_trajectory_emb=None,
                        cost_mode="composite"):
    """Score candidates with composite or trajectory-shape cost.

    Args:
        cost_mode: "composite", "shape", or "combined"

    Returns:
        scores: (N_cand,) — lower is better
        components: dict of per-component scores
    """
    if weights is None:
        weights = {"w_progress": 1.0, "w_smoothness": 0.3,
                   "w_consistency": 0.1, "w_action_reg": 0.05}

    N_cand = action_candidates.shape[0]
    emb_h = emb_history.unsqueeze(0).expand(N_cand, -1, -1)
    act_t = torch.from_numpy(action_candidates).float()

    pred_emb = rollout_predict(model, emb_h, act_t, device, history_size)
    # pred_emb: (N_cand, n_rollout, D)

    components = {}

    if cost_mode in ("composite", "combined"):
        composite_scores, comp = compute_composite_cost(
            pred_emb, goal_emb, emb_history, action_candidates, weights, gt_trajectory_emb
        )
        components.update({f"c_{k}": v for k, v in comp.items()})

    if cost_mode in ("shape", "combined") and gt_trajectory_emb is not None:
        shape_scores = compute_trajectory_shape_cost(pred_emb, gt_trajectory_emb, emb_history)
        components["shape"] = shape_scores

    if cost_mode == "composite":
        scores = composite_scores
    elif cost_mode == "shape":
        scores = shape_scores
    elif cost_mode == "combined":
        # Normalize each to [0,1] range before combining
        c_norm = (composite_scores - composite_scores.min()) / (composite_scores.max() - composite_scores.min() + 1e-8)
        s_norm = (shape_scores - shape_scores.min()) / (shape_scores.max() - shape_scores.min() + 1e-8)
        scores = 0.6 * c_norm + 0.4 * s_norm
        components["combined_composite_norm"] = c_norm
        components["combined_shape_norm"] = s_norm
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")

    # Also compute v1 MSE for comparison
    final_pred = pred_emb[:, -1, :]
    goal_exp = goal_emb.unsqueeze(0).expand(N_cand, -1)
    mse_scores = (final_pred - goal_exp).pow(2).mean(dim=-1).numpy()
    components["v1_mse"] = mse_scores

    return scores, components


def cem_optimize_v2(model, emb_history, goal_emb, initial_action, device,
                    history_size=3, n_candidates=64, n_elite=8, n_iterations=5,
                    noise_std=0.5, weights=None, gt_trajectory_emb=None,
                    cost_mode="composite"):
    """CEM with composite cost."""
    T, D = initial_action.shape
    mean = initial_action.copy()
    std = np.ones_like(initial_action) * noise_std

    convergence = []
    all_iter_scores = []

    for it in range(n_iterations):
        rng = np.random.RandomState(it * 1000)
        candidates = []
        for _ in range(n_candidates):
            sample = mean + rng.randn(T, D).astype(np.float32) * std
            candidates.append(sample)
        candidates = np.stack(candidates)

        scores, _ = score_candidates_v2(
            model, emb_history, goal_emb, candidates, device,
            history_size, weights, gt_trajectory_emb, cost_mode
        )
        all_iter_scores.append(scores.copy())

        elite_idx = np.argsort(scores)[:n_elite]
        elite = candidates[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6

        convergence.append(float(scores[elite_idx[0]]))

    return mean, convergence, all_iter_scores


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Track 1 v2: Improved Corridor Planning")
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--data", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-elite", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="outputs/track1_planning_v2.png")
    parser.add_argument("--device", type=str, default="cuda")
    # Cost weights
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-smoothness", type=float, default=0.3)
    parser.add_argument("--w-consistency", type=float, default=0.1)
    parser.add_argument("--w-action-reg", type=float, default=0.05)
    args = parser.parse_args()

    weights = {
        "w_progress": args.w_progress,
        "w_smoothness": args.w_smoothness,
        "w_consistency": args.w_consistency,
        "w_action_reg": args.w_action_reg,
    }

    n_rollout = args.num_steps - args.history_size
    print("=" * 70)
    print("Track 1 v2: Improved Corridor Planning (Composite Cost)")
    print("=" * 70)
    print(f"  History: {args.history_size} frames")
    print(f"  Rollout: {n_rollout} steps ({n_rollout * 0.5:.1f}s)")
    print(f"  Candidates: {args.n_candidates}")
    print(f"  CEM: {args.cem_iterations} iterations, top-{args.cem_elite} elite")
    print(f"  Weights: progress={weights['w_progress']}, smoothness={weights['w_smoothness']}, "
          f"consistency={weights['w_consistency']}, action_reg={weights['w_action_reg']}")
    print()

    # Load model
    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Load data
    print(f"Loading {args.n_sequences} sequences from: {args.data}")
    pixels, action = load_sequences(args.data, args.n_sequences, num_steps=args.num_steps)
    print(f"  pixels: {pixels.shape}, action: {action.shape}")

    action_norm, act_mean, act_std = normalize_actions(action)
    pixels_t = torch.from_numpy(pixels)

    # Encode all frames
    print("\nEncoding all frames...")
    gt_emb = encode_frames(model, pixels_t, transform, args.device)
    print(f"  gt_emb: {gt_emb.shape}")

    # ================================================================
    # Phase 1: Compare cost functions on candidate ranking
    # ================================================================
    cost_modes = ["composite", "shape", "combined"]
    results_by_mode = {}

    for mode in cost_modes:
        print(f"\n{'=' * 70}")
        print(f"Phase 1-{mode}: Action Candidate Ranking ({mode} cost)")
        print("=" * 70)

        logged_ranks = []
        top1_count = 0
        top5_count = 0
        top10_count = 0
        logged_scores_list = []
        best_scores_list = []
        component_sums = {}

        for seq_idx in range(args.n_sequences):
            if (seq_idx + 1) % 20 == 0 or seq_idx == 0:
                print(f"  Sequence {seq_idx + 1}/{args.n_sequences}...")

            emb_history = gt_emb[seq_idx, :args.history_size]
            goal_emb = gt_emb[seq_idx, -1]
            gt_trajectory = gt_emb[seq_idx, args.history_size:]  # (n_rollout, D)
            logged_act = action_norm[seq_idx]

            candidates, labels = generate_candidates(
                logged_act, n_candidates=args.n_candidates,
                noise_std=args.noise_std, seed=seq_idx
            )

            scores, components = score_candidates_v2(
                model, emb_history, goal_emb, candidates, args.device,
                args.history_size, weights, gt_trajectory, cost_mode=mode
            )

            # Track component contributions
            for k, v in components.items():
                if k not in component_sums:
                    component_sums[k] = []
                component_sums[k].append(v)

            logged_score = scores[0]
            rank = int((scores < logged_score).sum() + 1)
            logged_ranks.append(rank)
            logged_scores_list.append(logged_score)
            best_scores_list.append(scores.min())

            if rank == 1:
                top1_count += 1
            if rank <= 5:
                top5_count += 1
            if rank <= 10:
                top10_count += 1

        logged_ranks = np.array(logged_ranks)
        logged_scores = np.array(logged_scores_list)
        best_scores = np.array(best_scores_list)

        results_by_mode[mode] = {
            "logged_ranks": logged_ranks,
            "logged_scores": logged_scores,
            "best_scores": best_scores,
            "top1": top1_count,
            "top5": top5_count,
            "top10": top10_count,
            "component_sums": component_sums,
        }

        print(f"\n--- {mode} cost results ---")
        print(f"  Logged rank:   mean={logged_ranks.mean():.1f}, median={np.median(logged_ranks):.1f}")
        print(f"  Top-1:         {top1_count}/{args.n_sequences} ({top1_count/args.n_sequences*100:.1f}%)")
        print(f"  Top-5:         {top5_count}/{args.n_sequences} ({top5_count/args.n_sequences*100:.1f}%)")
        print(f"  Top-10:        {top10_count}/{args.n_sequences} ({top10_count/args.n_sequences*100:.1f}%)")

    # ================================================================
    # Phase 2: CEM with best cost mode
    # ================================================================
    # Pick the mode with lowest mean logged rank
    best_mode = min(cost_modes, key=lambda m: results_by_mode[m]["logged_ranks"].mean())
    print(f"\n{'=' * 70}")
    print(f"Phase 2: CEM Refinement (using {best_mode} cost)")
    print("=" * 70)

    cem_convergences = []
    cem_best_scores = []
    cem_vs_logged = []
    logged_scores_for_cem = results_by_mode[best_mode]["logged_scores"]

    for seq_idx in range(args.n_sequences):
        if (seq_idx + 1) % 20 == 0 or seq_idx == 0:
            print(f"  CEM sequence {seq_idx + 1}/{args.n_sequences}...")

        emb_history = gt_emb[seq_idx, :args.history_size]
        goal_emb = gt_emb[seq_idx, -1]
        gt_trajectory = gt_emb[seq_idx, args.history_size:]
        logged_act = action_norm[seq_idx]

        best_act, convergence, _ = cem_optimize_v2(
            model, emb_history, goal_emb, logged_act, args.device,
            history_size=args.history_size,
            n_candidates=args.n_candidates,
            n_elite=args.cem_elite,
            n_iterations=args.cem_iterations,
            noise_std=args.noise_std,
            weights=weights,
            gt_trajectory_emb=gt_trajectory,
            cost_mode=best_mode,
        )

        cem_best_scores.append(convergence[-1])
        cem_convergences.append(convergence)
        cem_vs_logged.append(convergence[-1] - logged_scores_for_cem[seq_idx])

    cem_best_scores = np.array(cem_best_scores)
    cem_convergences = np.array(cem_convergences)
    cem_vs_logged = np.array(cem_vs_logged)

    print(f"\n--- CEM Results ({best_mode} cost) ---")
    print(f"  CEM final cost:      mean={cem_best_scores.mean():.6f}")
    print(f"  Logged action cost:  mean={logged_scores_for_cem.mean():.6f}")
    print(f"  CEM vs Logged:       mean={cem_vs_logged.mean():.6f} "
          f"({'better' if cem_vs_logged.mean() < 0 else 'worse'})")
    print(f"  CEM beats logged:    {(cem_vs_logged < 0).sum()}/{args.n_sequences} "
          f"({(cem_vs_logged < 0).sum()/args.n_sequences*100:.1f}%)")
    mean_conv = cem_convergences.mean(axis=0)
    for it_idx, val in enumerate(mean_conv):
        improvement = (mean_conv[0] - val) / mean_conv[0] * 100 if mean_conv[0] > 0 else 0
        print(f"    Iter {it_idx}: cost={val:.6f} ({improvement:+.1f}% from iter 0)")

    # ================================================================
    # Phase 3: Component analysis (for composite mode)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Cost Component Analysis")
    print("=" * 70)

    comp_data = results_by_mode["composite"]["component_sums"]
    # For each component, compute correlation with logged rank
    composite_ranks = results_by_mode["composite"]["logged_ranks"]

    print("\nPer-component logged-action scores (mean across sequences, candidate 0 = logged):")
    for k in sorted(comp_data.keys()):
        vals = np.array(comp_data[k])  # (n_seq, n_cand)
        logged_vals = vals[:, 0]  # score of logged action
        all_vals_mean = vals.mean(axis=1)  # mean score across all candidates
        # How often is logged in bottom half for this component?
        logged_rank_per_comp = np.array([(v < v[0]).sum() + 1 for v in vals])
        print(f"  {k:30s}: logged_mean={logged_vals.mean():.6f}, "
              f"all_mean={all_vals_mean.mean():.6f}, "
              f"logged_rank_mean={logged_rank_per_comp.mean():.1f}/{args.n_candidates}")

    # ================================================================
    # Phase 4: Visualization
    # ================================================================
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 1. Rank comparison across cost modes
    ax = axes[0, 0]
    mode_names = []
    mode_means = []
    mode_medians = []
    colors_modes = {"composite": "#3498db", "shape": "#e74c3c", "combined": "#2ecc71"}
    for mode in cost_modes:
        ranks = results_by_mode[mode]["logged_ranks"]
        mode_names.append(mode)
        mode_means.append(ranks.mean())
        mode_medians.append(np.median(ranks))
    # Add v1 MSE for reference (from composite components)
    v1_mse_data = np.array(comp_data.get("v1_mse", []))
    if len(v1_mse_data) > 0:
        v1_ranks = np.array([(v < v[0]).sum() + 1 for v in v1_mse_data])
        mode_names.append("v1_mse")
        mode_means.append(v1_ranks.mean())
        mode_medians.append(np.median(v1_ranks))
        colors_modes["v1_mse"] = "#95a5a6"

    x_pos = np.arange(len(mode_names))
    bars = ax.bar(x_pos - 0.15, mode_means, 0.3, label="mean", color=[colors_modes.get(m, "#666") for m in mode_names])
    ax.bar(x_pos + 0.15, mode_medians, 0.3, label="median", color=[colors_modes.get(m, "#666") for m in mode_names], alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_names, rotation=15)
    ax.set_ylabel("Logged Action Rank")
    ax.set_title("Logged Action Rank by Cost Function")
    ax.legend()
    ax.axhline(y=args.n_candidates / 2, color="gray", linestyle=":", alpha=0.5, label="chance")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Rank histogram for best mode
    ax = axes[0, 1]
    best_ranks = results_by_mode[best_mode]["logged_ranks"]
    ax.hist(best_ranks, bins=np.arange(0.5, args.n_candidates + 1.5, 1),
            color=colors_modes[best_mode], edgecolor="black", alpha=0.7)
    ax.axvline(x=np.median(best_ranks), color="#e74c3c", linestyle="--",
               linewidth=2, label=f"median={np.median(best_ranks):.0f}")
    ax.axvline(x=best_ranks.mean(), color="#2ecc71", linestyle="--",
               linewidth=2, label=f"mean={best_ranks.mean():.1f}")
    ax.set_xlabel("Rank of Logged Action")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank Distribution ({best_mode} cost)")
    ax.legend()

    # 3. Top-K accuracy comparison
    ax = axes[0, 2]
    for mode in cost_modes:
        r = results_by_mode[mode]
        top_ks = [1, 5, 10, 15, 20, 25, 32]
        accuracies = []
        for k in top_ks:
            acc = (r["logged_ranks"] <= k).sum() / args.n_sequences * 100
            accuracies.append(acc)
        ax.plot(top_ks, accuracies, "o-", color=colors_modes[mode], label=mode, linewidth=2)
    # v1 MSE reference
    if len(v1_mse_data) > 0:
        v1_accs = []
        for k in top_ks:
            acc = (v1_ranks <= k).sum() / args.n_sequences * 100
            v1_accs.append(acc)
        ax.plot(top_ks, v1_accs, "s--", color="#95a5a6", label="v1_mse", linewidth=2)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-K Accuracy by Cost Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CEM convergence
    ax = axes[1, 0]
    iters = np.arange(args.cem_iterations)
    mean_conv = cem_convergences.mean(axis=0)
    std_conv = cem_convergences.std(axis=0)
    ax.plot(iters, mean_conv, "o-", color="#e74c3c", linewidth=2, label="CEM best")
    ax.fill_between(iters, mean_conv - std_conv, mean_conv + std_conv,
                    color="#e74c3c", alpha=0.15)
    ax.axhline(y=logged_scores_for_cem.mean(), color="#2ecc71", linestyle="--",
               linewidth=2, label=f"logged mean={logged_scores_for_cem.mean():.4f}")
    ax.set_xlabel("CEM Iteration")
    ax.set_ylabel(f"Best Cost ({best_mode})")
    ax.set_title(f"CEM Convergence ({best_mode} cost)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Component contribution (composite mode)
    ax = axes[1, 1]
    comp_names = ["c_progress", "c_smoothness", "c_consistency", "c_action_reg"]
    comp_labels = ["progress", "smoothness", "consistency", "action_reg"]
    comp_weights_list = [weights["w_progress"], weights["w_smoothness"],
                         weights["w_consistency"], weights["w_action_reg"]]
    # For each component, compute mean rank of logged action
    comp_rank_means = []
    for cn in comp_names:
        if cn in comp_data:
            vals = np.array(comp_data[cn])
            ranks_per_comp = np.array([(v < v[0]).sum() + 1 for v in vals])
            comp_rank_means.append(ranks_per_comp.mean())
        else:
            comp_rank_means.append(args.n_candidates / 2)

    colors_comp = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    ax.barh(comp_labels, comp_rank_means, color=colors_comp, alpha=0.8)
    ax.axvline(x=args.n_candidates / 2, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"Mean Logged Rank (/{args.n_candidates})")
    ax.set_title("Per-Component: Logged Action Rank")
    for i, (v, w) in enumerate(zip(comp_rank_means, comp_weights_list)):
        ax.text(v + 0.5, i, f"w={w}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # 6. CEM vs Logged scatter
    ax = axes[1, 2]
    ax.scatter(logged_scores_for_cem, cem_best_scores, alpha=0.5, s=20, c="#3498db")
    lim_max = max(logged_scores_for_cem.max(), cem_best_scores.max()) * 1.05
    lim_min = min(logged_scores_for_cem.min(), cem_best_scores.min()) * 0.95
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Logged Action Cost")
    ax.set_ylabel("CEM Best Cost")
    n_cem_wins = (cem_vs_logged < 0).sum()
    ax.set_title(f"CEM vs Logged (below=CEM wins: {n_cem_wins}/{args.n_sequences})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Track 1 v2: Improved Corridor Planning (Composite Cost)\n"
        f"Best mode: {best_mode} | Rank: {results_by_mode[best_mode]['logged_ranks'].mean():.1f}/{args.n_candidates} | "
        f"Top-1: {results_by_mode[best_mode]['top1']} | "
        f"Top-5: {results_by_mode[best_mode]['top5']} | "
        f"Top-10: {results_by_mode[best_mode]['top10']}",
        fontsize=13, y=1.02
    )
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
    print("SUMMARY: Track 1 v2 — Improved Corridor Planning")
    print("=" * 70)
    print(f"  Sequences: {args.n_sequences}, Candidates: {args.n_candidates}")
    print(f"  Horizon: {n_rollout * 0.5:.1f}s ({n_rollout} steps)")
    print()

    print("  [Cost Function Comparison]")
    print(f"  {'Mode':12s} | {'Mean Rank':>10s} | {'Median':>7s} | {'Top-1':>6s} | {'Top-5':>6s} | {'Top-10':>7s}")
    print("  " + "-" * 65)
    for mode in cost_modes:
        r = results_by_mode[mode]
        print(f"  {mode:12s} | {r['logged_ranks'].mean():>10.1f} | "
              f"{np.median(r['logged_ranks']):>7.1f} | "
              f"{r['top1']:>6d} | {r['top5']:>6d} | {r['top10']:>7d}")
    if len(v1_mse_data) > 0:
        print(f"  {'v1_mse':12s} | {v1_ranks.mean():>10.1f} | "
              f"{np.median(v1_ranks):>7.1f} | "
              f"{(v1_ranks <= 1).sum():>6d} | {(v1_ranks <= 5).sum():>6d} | {(v1_ranks <= 10).sum():>7d}")

    print()
    print(f"  [CEM ({best_mode} cost)]")
    print(f"    Final cost:        {cem_best_scores.mean():.6f}")
    print(f"    Logged cost:       {logged_scores_for_cem.mean():.6f}")
    print(f"    CEM beats logged:  {n_cem_wins}/{args.n_sequences} ({n_cem_wins/args.n_sequences*100:.1f}%)")
    mean_conv = cem_convergences.mean(axis=0)
    if len(mean_conv) > 1:
        total_improvement = (mean_conv[0] - mean_conv[-1]) / mean_conv[0] * 100
        print(f"    CEM improvement:   {total_improvement:.1f}% over {args.cem_iterations} iterations")

    print()
    print("  [Component Contribution (composite cost)]")
    for cn, cl in zip(comp_names, comp_labels):
        if cn in comp_data:
            vals = np.array(comp_data[cn])
            ranks_per = np.array([(v < v[0]).sum() + 1 for v in vals])
            print(f"    {cl:15s}: logged_rank={ranks_per.mean():.1f}/{args.n_candidates}")

    print()
    best_r = results_by_mode[best_mode]
    if best_r["logged_ranks"].mean() < 20:
        verdict = f"GOOD — {best_mode} cost ranks logged in top third"
    elif best_r["logged_ranks"].mean() < args.n_candidates * 0.5:
        verdict = f"IMPROVED — {best_mode} cost ranks logged above median"
    else:
        verdict = f"NEEDS WORK — cost functions need further tuning"

    # Compare vs v1
    if len(v1_mse_data) > 0:
        v1_mean = v1_ranks.mean()
        best_mean = best_r["logged_ranks"].mean()
        if best_mean < v1_mean:
            vs_v1 = f"v2 ({best_mode}) rank {best_mean:.1f} vs v1 MSE rank {v1_mean:.1f} = BETTER"
        else:
            vs_v1 = f"v2 ({best_mode}) rank {best_mean:.1f} vs v1 MSE rank {v1_mean:.1f} = WORSE"
        print(f"  vs V1: {vs_v1}")

    print(f"  VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
