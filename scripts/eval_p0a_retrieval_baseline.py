#!/usr/bin/env python3
"""P0-A: Retrieval baseline — is LeWM more than memorization?

Compares LeWM's predictor against kNN baselines:
1. Visual kNN: nearest neighbor in embedding space → use its future embedding
2. Visual+Action kNN: nearest in (embedding, action) space
3. Proprio kNN: nearest in (speed, heading_rate) space
4. LeWM predictor: actual model prediction

If kNN matches or beats LeWM, the +43.8% motion gap is mostly memorization.
If LeWM significantly beats kNN, it has learned dynamics beyond retrieval.

Uses training set as the retrieval database, evaluates on holdout.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from tqdm import tqdm

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

try:
    from train_vp import LaneDecoder, DepthDecoder  # noqa: F401
except ImportError:
    pass


def load_model(ckpt_path, device="cuda"):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def get_img_transform(img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(**stats, source="pixels", target="pixels")
    resize = spt.data.transforms.Resize(img_size, source="pixels", target="pixels")
    return spt.data.transforms.Compose(to_image, resize)


def encode_frames(model, pixels, transform, device, batch_size=64):
    """Encode raw pixel frames into embeddings."""
    all_emb = []
    for i in range(0, len(pixels), batch_size):
        px = torch.from_numpy(pixels[i:i+batch_size]).unsqueeze(1)  # (B, 1, 3, H, W)
        batch = {"pixels": px}
        batch = transform(batch)
        px_t = batch["pixels"].to(device)
        with torch.no_grad():
            info = {"pixels": px_t, "action": torch.zeros(px_t.size(0), 1, 15, device=device)}
            info = model.encode(info)
        all_emb.append(info["emb"][:, 0].cpu())  # (B, D)
    return torch.cat(all_emb, dim=0)


def load_h5_raw(h5_path, max_frames=None):
    """Load raw frames, actions, proprio from H5."""
    with h5py.File(h5_path, "r") as f:
        n = f["pixels"].shape[0] if max_frames is None else min(f["pixels"].shape[0], max_frames)
        pixels = f["pixels"][:n]
        action = f["action"][:n]
        proprio = f["proprio"][:n]
    return pixels, action, proprio


def build_retrieval_db(model, train_h5_paths, transform, device, frameskip=5, max_per_file=None):
    """Build retrieval database: (current_emb, action_chunk, proprio, future_emb) tuples."""
    all_curr_emb = []
    all_future_emb = []
    all_action = []
    all_proprio = []

    for h5_path in train_h5_paths:
        print(f"  Encoding: {Path(h5_path).stem}")
        pixels, action, proprio = load_h5_raw(h5_path, max_frames=max_per_file)
        n = len(pixels)

        # Encode all frames
        emb = encode_frames(model, pixels, transform, device)

        # Build pairs: frame t → frame t+frameskip
        valid = n - frameskip
        curr_idx = np.arange(valid)
        future_idx = curr_idx + frameskip

        all_curr_emb.append(emb[curr_idx])
        all_future_emb.append(emb[future_idx])

        # Action chunk: aggregate frameskip actions
        action_chunks = []
        for i in range(valid):
            chunk = action[i:i+frameskip].flatten()
            action_chunks.append(chunk)
        all_action.append(np.stack(action_chunks))

        # Proprio at current frame
        all_proprio.append(proprio[curr_idx])

    db = {
        "curr_emb": torch.cat(all_curr_emb, dim=0),
        "future_emb": torch.cat(all_future_emb, dim=0),
        "action": np.concatenate(all_action, axis=0),
        "proprio": np.concatenate(all_proprio, axis=0),
    }
    print(f"  DB size: {len(db['curr_emb'])} pairs")
    return db


def knn_predict(query_emb, db_emb, db_future_emb, k=5):
    """Find k nearest neighbors and average their future embeddings."""
    # Cosine similarity
    query_norm = F.normalize(query_emb, dim=-1)
    db_norm = F.normalize(db_emb, dim=-1)
    sim = query_norm @ db_norm.T  # (Q, N)
    topk_idx = sim.topk(k, dim=-1).indices  # (Q, k)

    # Average future embeddings of k neighbors
    preds = []
    for i in range(len(query_emb)):
        nn_future = db_future_emb[topk_idx[i]]  # (k, D)
        preds.append(nn_future.mean(dim=0))
    return torch.stack(preds)  # (Q, D)


def knn_predict_combined(query_emb, query_extra, db_emb, db_extra, db_future_emb,
                         k=5, alpha=0.5):
    """kNN with combined embedding + extra features."""
    # Normalize both
    q_emb_n = F.normalize(query_emb, dim=-1)
    d_emb_n = F.normalize(db_emb, dim=-1)
    sim_emb = q_emb_n @ d_emb_n.T  # (Q, N)

    # Extra feature similarity (L2 → similarity)
    q_ext = torch.from_numpy(query_extra).float()
    d_ext = torch.from_numpy(db_extra).float()
    # Standardize
    ext_mean = d_ext.mean(0, keepdim=True)
    ext_std = d_ext.std(0, keepdim=True).clamp(min=1e-6)
    q_ext_n = (q_ext - ext_mean) / ext_std
    d_ext_n = (d_ext - ext_mean) / ext_std
    # Negative L2 as similarity
    dist = torch.cdist(q_ext_n, d_ext_n)  # (Q, N)
    sim_ext = -dist

    # Normalize to same scale
    sim_ext = (sim_ext - sim_ext.mean()) / sim_ext.std().clamp(min=1e-6)
    sim_emb_n = (sim_emb - sim_emb.mean()) / sim_emb.std().clamp(min=1e-6)

    sim = alpha * sim_emb_n + (1 - alpha) * sim_ext
    topk_idx = sim.topk(k, dim=-1).indices

    preds = []
    for i in range(len(query_emb)):
        nn_future = db_future_emb[topk_idx[i]]
        preds.append(nn_future.mean(dim=0))
    return torch.stack(preds)


def lewm_predict(model, query_emb, query_action, device, history_size=3):
    """Use LeWM predictor for single-step prediction.

    We need history_size frames. Use the same frame repeated as context
    (this is a conservative baseline — the model normally sees a sequence).
    Actually, for fair comparison, we'll use the query embedding as the last
    context frame and predict 1 step ahead.
    """
    # Expand single frame to history by repeating (conservative)
    B = query_emb.size(0)
    emb = query_emb.unsqueeze(1).expand(B, history_size, -1).to(device)  # (B, HS, D)
    act = query_action.unsqueeze(1).expand(B, history_size, -1).to(device)  # (B, HS, A)

    with torch.no_grad():
        act_emb = model.action_encoder(act)
        pred = model.predict(emb, act_emb)[:, -1]  # (B, D)
    return pred.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(STABLEWM_HOME / "expA1/lewm_expA1_livlab_only_epoch_15_object.ckpt"))
    parser.add_argument("--holdout", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany/Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd.h5"))
    parser.add_argument("--train-dir", type=str,
                        default=str(STABLEWM_HOME / "rtb_occany"))
    parser.add_argument("--n-eval", type=int, default=500, help="Number of eval pairs")
    parser.add_argument("--k", type=int, default=5, help="k for kNN")
    parser.add_argument("--max-per-file", type=int, default=5000)
    parser.add_argument("--output", type=str, default="outputs/p0a_retrieval_baseline.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("P0-A: Retrieval Baseline — Memorization vs Dynamics")
    print("=" * 60)

    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    # Build retrieval DB from training recordings (exclude holdout)
    holdout_stem = Path(args.holdout).stem
    train_paths = sorted(Path(args.train_dir).glob("*.h5"))
    train_paths = [p for p in train_paths if holdout_stem not in p.stem]
    # Also exclude da8241 for A1 (Livlab-only model)
    train_paths = [p for p in train_paths if "da8241" not in p.stem]

    print(f"\nTrain recordings ({len(train_paths)}):")
    for p in train_paths:
        print(f"  {p.stem}")

    print(f"\nBuilding retrieval database...")
    db = build_retrieval_db(model, train_paths, transform, args.device,
                            max_per_file=args.max_per_file)

    # Load and encode holdout
    print(f"\nEncoding holdout: {Path(args.holdout).stem}")
    h_pixels, h_action, h_proprio = load_h5_raw(args.holdout, max_frames=args.max_per_file)
    h_emb = encode_frames(model, h_pixels, transform, args.device)

    # Build eval pairs from holdout
    frameskip = 5
    valid = len(h_emb) - frameskip
    rng = np.random.RandomState(42)
    eval_idx = rng.choice(valid, size=min(args.n_eval, valid), replace=False)

    query_emb = h_emb[eval_idx]
    target_emb = h_emb[eval_idx + frameskip]

    # Action chunks for eval
    query_action_chunks = []
    for i in eval_idx:
        chunk = h_action[i:i+frameskip].flatten()
        query_action_chunks.append(chunk)
    query_action = np.stack(query_action_chunks)

    query_proprio = h_proprio[eval_idx]

    print(f"\nEval: {len(eval_idx)} pairs from holdout")

    # Normalize action for LeWM
    all_action = np.concatenate([db["action"], query_action], axis=0)
    act_mean = all_action.mean(0, keepdims=True)
    act_std = all_action.std(0, keepdims=True)
    act_std = np.where(act_std < 1e-8, 1.0, act_std)
    query_action_norm = torch.from_numpy(
        np.nan_to_num((query_action - act_mean) / act_std, 0.0).astype(np.float32)
    )

    # Run baselines
    print("\n--- Running baselines ---")

    results = {}

    # 1. Visual kNN
    print("Visual kNN...")
    pred_visual = knn_predict(query_emb, db["curr_emb"], db["future_emb"], k=args.k)
    mse_visual = (pred_visual - target_emb).pow(2).mean(dim=-1)
    results["Visual kNN"] = mse_visual.numpy()

    # 2. Visual+Action kNN
    print("Visual+Action kNN...")
    pred_va = knn_predict_combined(
        query_emb, query_action, db["curr_emb"], db["action"], db["future_emb"],
        k=args.k, alpha=0.7
    )
    mse_va = (pred_va - target_emb).pow(2).mean(dim=-1)
    results["Visual+Action kNN"] = mse_va.numpy()

    # 3. Proprio kNN
    print("Proprio kNN...")
    pred_proprio = knn_predict_combined(
        query_emb, query_proprio, db["curr_emb"], db["proprio"], db["future_emb"],
        k=args.k, alpha=0.3  # more weight on proprio
    )
    mse_proprio = (pred_proprio - target_emb).pow(2).mean(dim=-1)
    results["Proprio kNN"] = mse_proprio.numpy()

    # 4. LeWM predictor
    print("LeWM predictor...")
    batch_size = 64
    all_preds = []
    for i in range(0, len(query_emb), batch_size):
        pred = lewm_predict(
            model, query_emb[i:i+batch_size], query_action_norm[i:i+batch_size], args.device
        )
        all_preds.append(pred)
    pred_lewm = torch.cat(all_preds, dim=0)
    mse_lewm = (pred_lewm - target_emb).pow(2).mean(dim=-1)
    results["LeWM predictor"] = mse_lewm.numpy()

    # 5. Naive baseline: predict current = future (no change)
    mse_naive = (query_emb - target_emb).pow(2).mean(dim=-1)
    results["No-change"] = mse_naive.numpy()

    # 6. LeWM with shuffled action
    print("LeWM shuffled action...")
    shuffle_idx = rng.permutation(len(query_action_norm))
    all_preds_shuf = []
    for i in range(0, len(query_emb), batch_size):
        pred = lewm_predict(
            model, query_emb[i:i+batch_size],
            query_action_norm[shuffle_idx[i:i+batch_size]], args.device
        )
        all_preds_shuf.append(pred)
    pred_lewm_shuf = torch.cat(all_preds_shuf, dim=0)
    mse_lewm_shuf = (pred_lewm_shuf - target_emb).pow(2).mean(dim=-1)
    results["LeWM shuffled"] = mse_lewm_shuf.numpy()

    # Print results
    print("\n" + "=" * 60)
    print("P0-A: Retrieval Baseline Results")
    print("=" * 60)
    print(f"{'Method':>25s} | {'MSE Mean':>10s} | {'MSE Std':>10s} | {'vs LeWM':>8s}")
    print("-" * 60)

    lewm_mean = results["LeWM predictor"].mean()
    for name in ["No-change", "Visual kNN", "Visual+Action kNN", "Proprio kNN",
                  "LeWM predictor", "LeWM shuffled"]:
        mse = results[name]
        vs = (mse.mean() - lewm_mean) / lewm_mean * 100
        print(f"  {name:>23s} | {mse.mean():>10.6f} | {mse.std():>10.6f} | {vs:>+7.1f}%")

    # Key question
    print("\n--- Key Question: Memorization vs Dynamics ---")
    knn_best = min(results["Visual kNN"].mean(), results["Visual+Action kNN"].mean())
    gap = (knn_best - lewm_mean) / lewm_mean * 100
    if gap > 10:
        print(f"  LeWM beats best kNN by {gap:.1f}% → DYNAMICS (learned more than retrieval)")
    elif gap > 0:
        print(f"  LeWM beats best kNN by only {gap:.1f}% → MOSTLY MEMORIZATION with slight dynamics")
    else:
        print(f"  kNN beats LeWM by {-gap:.1f}% → PURE MEMORIZATION (kNN is better!)")

    # Action sensitivity comparison
    print("\n--- Action Sensitivity ---")
    lewm_gap = (results["LeWM shuffled"].mean() - lewm_mean) / lewm_mean * 100
    knn_gap_va = (results["Visual kNN"].mean() - results["Visual+Action kNN"].mean()) / results["Visual+Action kNN"].mean() * 100
    print(f"  LeWM: correct vs shuffled: {lewm_gap:+.1f}% (action matters)")
    print(f"  kNN: visual-only vs visual+action: {knn_gap_va:+.1f}% (action in retrieval)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Bar chart comparison
    ax = axes[0]
    names = ["No-change", "Visual\nkNN", "V+A\nkNN", "Proprio\nkNN", "LeWM\ncorrect", "LeWM\nshuffled"]
    keys = ["No-change", "Visual kNN", "Visual+Action kNN", "Proprio kNN", "LeWM predictor", "LeWM shuffled"]
    means = [results[k].mean() for k in keys]
    stds = [results[k].std() / np.sqrt(len(results[k])) for k in keys]
    colors = ["#95a5a6", "#3498db", "#2980b9", "#8e44ad", "#2ecc71", "#e74c3c"]
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
    ax.set_ylabel("MSE")
    ax.set_title("Future Embedding Prediction MSE")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # 2. Distribution comparison (LeWM vs best kNN)
    ax = axes[1]
    ax.hist(results["Visual kNN"], bins=50, alpha=0.5, color="#3498db", label="Visual kNN", density=True)
    ax.hist(results["LeWM predictor"], bins=50, alpha=0.5, color="#2ecc71", label="LeWM correct", density=True)
    ax.hist(results["LeWM shuffled"], bins=50, alpha=0.3, color="#e74c3c", label="LeWM shuffled", density=True)
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.set_title("MSE Distribution: LeWM vs kNN")
    ax.legend(fontsize=8)

    # 3. Scatter: LeWM vs kNN per sample
    ax = axes[2]
    ax.scatter(results["LeWM predictor"], results["Visual kNN"], s=3, alpha=0.4, c="steelblue")
    lim = max(results["LeWM predictor"].max(), results["Visual kNN"].max()) * 0.8
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("LeWM MSE")
    ax.set_ylabel("Visual kNN MSE")
    ax.set_title("Per-Sample: LeWM vs Visual kNN")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.suptitle("P0-A: Retrieval Baseline — Memorization vs Dynamics (ADR-008)", fontsize=13, y=1.02)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
