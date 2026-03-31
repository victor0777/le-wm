#!/usr/bin/env python3
"""E0: Motion-conditioning ablation.

Tests whether the model actually uses the action (ego-motion) input
by comparing prediction quality with:
  - correct motion block
  - shuffled motion block (random permutation across batch)
  - zeroed motion block

If prediction quality is similar across all conditions, the action proxy
is not contributing to future prediction.

Outputs:
  - Per-condition MSE and cosine similarity
  - Stratified analysis by speed and yaw-rate bins
  - Summary plot saved to outputs/e0_motion_ablation.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from matplotlib.gridspec import GridSpec

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

# Import VP decoders so torch.load can unpickle them
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


def load_sequences(h5_path: str, n_sequences: int = 500, frameskip: int = 5, num_steps: int = 4, seed: int = 42):
    """Load sequences in the same format as HDF5Dataset.__getitem__."""
    rng = np.random.RandomState(seed)
    raw_len = num_steps * frameskip  # 20 raw frames per sequence

    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        max_start = total - raw_len
        starts = rng.randint(0, max_start, size=n_sequences)

        pixels_list = []
        action_list = []
        proprio_list = []

        for s in starts:
            # Pixels: subsampled by frameskip
            px_indices = list(range(s, s + raw_len, frameskip))
            pixels_list.append(f["pixels"][px_indices])  # (num_steps, 3, H, W)

            # Action: raw consecutive (loader reshapes to num_steps x frameskip*action_dim)
            action_raw = f["action"][s : s + raw_len]  # (raw_len, action_dim)
            action_dim = action_raw.shape[-1]
            action_list.append(action_raw.reshape(num_steps, frameskip * action_dim))

            # Proprio: subsampled
            proprio_list.append(f["proprio"][px_indices])  # (num_steps, 8)

    pixels = np.stack(pixels_list)   # (N, T, 3, H, W)
    action = np.stack(action_list)   # (N, T, 15)
    proprio = np.stack(proprio_list) # (N, T, 8)

    return pixels, action, proprio


def normalize_actions(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler normalization (fit on data)."""
    flat = action.reshape(-1, action.shape[-1])
    mask = ~np.isnan(flat).any(axis=1)
    mean = flat[mask].mean(axis=0, keepdims=True)
    std = flat[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (action - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32), mean, std


def run_prediction(model, pixels_t, action_t, transform, device, history_size=3):
    """Run encode + predict, return predicted and target embeddings."""
    batch = {"pixels": pixels_t}
    batch = transform(batch)
    px = batch["pixels"].to(device)  # (B, T, 3, 224, 224)
    act = action_t.to(device)        # (B, T, 15)

    with torch.no_grad():
        info = {"pixels": px, "action": act}
        info = model.encode(info)
        emb = info["emb"]          # (B, T, D)
        act_emb = info["act_emb"]  # (B, T, D)

        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        tgt_emb = emb[:, 1:]  # (B, history_size, D) — shifted target

        pred_emb = model.predict(ctx_emb, ctx_act)  # (B, history_size, D)

    return pred_emb.cpu(), tgt_emb.cpu()


def compute_metrics(pred_emb, tgt_emb):
    """Compute MSE and cosine similarity."""
    mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1).mean(dim=-1)  # (B,)
    cos = torch.nn.functional.cosine_similarity(
        pred_emb.reshape(-1, pred_emb.shape[-1]),
        tgt_emb.reshape(-1, tgt_emb.shape[-1]),
        dim=-1,
    )
    return mse.numpy(), cos.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(STABLEWM_HOME / "lewm_rtb_epoch_5_object.ckpt"))
    parser.add_argument("--data", type=str, default=str(STABLEWM_HOME / "rtb/Gian-Pankyo_JT_2025-08-19_05-51-29_2111_58bb5f.h5"))
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument("--output", type=str, default="outputs/e0_motion_ablation.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    transform = get_img_transform()

    print(f"Loading {args.n_sequences} sequences from: {args.data}")
    pixels, action, proprio = load_sequences(args.data, n_sequences=args.n_sequences)
    print(f"  pixels: {pixels.shape}, action: {action.shape}, proprio: {proprio.shape}")

    # Normalize actions
    action_norm, act_mean, act_std = normalize_actions(action)

    # Prepare tensors
    pixels_t = torch.from_numpy(pixels)
    action_correct = torch.from_numpy(action_norm)

    # Shuffled: permute action across batch dimension
    rng = np.random.RandomState(123)
    shuffle_idx = rng.permutation(len(action_norm))
    action_shuffled = torch.from_numpy(action_norm[shuffle_idx])

    # Zeroed: all zeros
    action_zeroed = torch.zeros_like(action_correct)

    # Run predictions in batches
    batch_size = 32
    conditions = {
        "correct": action_correct,
        "shuffled": action_shuffled,
        "zeroed": action_zeroed,
    }

    results = {}
    for cond_name, act_tensor in conditions.items():
        print(f"\nRunning condition: {cond_name}")
        all_mse, all_cos = [], []

        for i in range(0, len(pixels_t), batch_size):
            px_batch = pixels_t[i : i + batch_size]
            act_batch = act_tensor[i : i + batch_size]

            pred_emb, tgt_emb = run_prediction(model, px_batch, act_batch, transform, args.device)
            mse, cos = compute_metrics(pred_emb, tgt_emb)
            all_mse.append(mse)
            all_cos.append(cos)

        results[cond_name] = {
            "mse": np.concatenate(all_mse),
            "cos": np.concatenate(all_cos),
        }

    # Extract speed and yaw for stratification
    speed = proprio[:, 0, 0]  # speed at first frame
    yaw = np.abs(action[:, 0, 2])  # |angular_yaw| at first action (raw, before norm)

    # Print summary
    print("\n" + "=" * 60)
    print("E0: Motion-Conditioning Ablation Results")
    print("=" * 60)
    for cond_name, res in results.items():
        mse_mean = res["mse"].mean()
        mse_std = res["mse"].std()
        cos_mean = res["cos"].mean()
        cos_std = res["cos"].std()
        print(f"  {cond_name:10s}: MSE = {mse_mean:.6f} ± {mse_std:.6f}, "
              f"Cosine = {cos_mean:.4f} ± {cos_std:.4f}")

    # Relative degradation
    mse_correct = results["correct"]["mse"].mean()
    for cond in ["shuffled", "zeroed"]:
        mse_cond = results[cond]["mse"].mean()
        pct = (mse_cond - mse_correct) / mse_correct * 100
        print(f"  {cond} vs correct: MSE +{pct:.1f}%")

    # Plot
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. MSE comparison bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    cond_names = list(results.keys())
    mse_means = [results[c]["mse"].mean() for c in cond_names]
    mse_stds = [results[c]["mse"].std() / np.sqrt(len(results[c]["mse"])) for c in cond_names]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax1.bar(cond_names, mse_means, yerr=mse_stds, color=colors, capsize=5, edgecolor="white")
    ax1.set_ylabel("MSE")
    ax1.set_title("Prediction MSE by Condition")
    for bar, val in zip(bars, mse_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # 2. Cosine similarity comparison
    ax2 = fig.add_subplot(gs[0, 1])
    cos_means = [results[c]["cos"].mean() for c in cond_names]
    cos_stds = [results[c]["cos"].std() / np.sqrt(len(results[c]["cos"])) for c in cond_names]
    bars = ax2.bar(cond_names, cos_means, yerr=cos_stds, color=colors, capsize=5, edgecolor="white")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Prediction Cosine Sim by Condition")
    for bar, val in zip(bars, cos_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # 3. MSE distribution violin plot
    ax3 = fig.add_subplot(gs[0, 2])
    data_violin = [results[c]["mse"] for c in cond_names]
    vp = ax3.violinplot(data_violin, positions=[0, 1, 2], showmeans=True, showmedians=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(cond_names)
    ax3.set_ylabel("MSE")
    ax3.set_title("MSE Distribution")

    # 4. Stratified by speed
    ax4 = fig.add_subplot(gs[1, 0])
    speed_bins = np.quantile(speed, [0, 0.33, 0.66, 1.0])
    speed_labels = ["slow", "medium", "fast"]
    speed_cat = np.digitize(speed, speed_bins[1:-1])

    for ci, cond in enumerate(cond_names):
        bin_means = []
        for b in range(3):
            mask = speed_cat == b
            if mask.sum() > 0:
                bin_means.append(results[cond]["mse"][mask].mean())
            else:
                bin_means.append(0)
        x = np.arange(3) + ci * 0.25
        ax4.bar(x, bin_means, width=0.22, color=colors[ci], label=cond, edgecolor="white")

    ax4.set_xticks(np.arange(3) + 0.25)
    ax4.set_xticklabels(speed_labels)
    ax4.set_ylabel("MSE")
    ax4.set_title("MSE by Speed Bin")
    ax4.legend(fontsize=8)

    # 5. Stratified by yaw rate
    ax5 = fig.add_subplot(gs[1, 1])
    yaw_bins = np.quantile(yaw, [0, 0.5, 0.9, 1.0])
    yaw_labels = ["straight", "mild turn", "sharp turn"]
    yaw_cat = np.digitize(yaw, yaw_bins[1:-1])

    for ci, cond in enumerate(cond_names):
        bin_means = []
        for b in range(3):
            mask = yaw_cat == b
            if mask.sum() > 0:
                bin_means.append(results[cond]["mse"][mask].mean())
            else:
                bin_means.append(0)
        x = np.arange(3) + ci * 0.25
        ax5.bar(x, bin_means, width=0.22, color=colors[ci], label=cond, edgecolor="white")

    ax5.set_xticks(np.arange(3) + 0.25)
    ax5.set_xticklabels(yaw_labels)
    ax5.set_ylabel("MSE")
    ax5.set_title("MSE by Yaw Rate Bin")
    ax5.legend(fontsize=8)

    # 6. Per-sample scatter: correct vs shuffled MSE
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(results["correct"]["mse"], results["shuffled"]["mse"],
                s=3, alpha=0.5, c="steelblue")
    lim = max(results["correct"]["mse"].max(), results["shuffled"]["mse"].max()) * 1.1
    ax6.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
    ax6.set_xlabel("MSE (correct motion)")
    ax6.set_ylabel("MSE (shuffled motion)")
    ax6.set_title("Per-sample: Correct vs Shuffled")
    ax6.legend()

    fig.suptitle("E0: Motion-Conditioning Ablation (LeWM RTB, 5 epochs)", fontsize=14, y=0.98)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
