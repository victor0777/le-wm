#!/usr/bin/env python3
"""Visualize LeWM embeddings from RTB driving data.

Produces:
1. t-SNE/UMAP scatter plot colored by speed (driving dynamics)
2. t-SNE/UMAP scatter plot colored by time (temporal structure)
3. Cosine similarity matrix of sampled frames
4. Top-5 nearest neighbor retrieval examples
"""

import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

STABLEWM_HOME = Path.home() / ".stable_worldmodel"


def load_model(ckpt_path: str):
    """Load a LeWM checkpoint (object checkpoint)."""
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def get_img_transform(img_size: int = 224):
    """ImageNet normalization + resize transform."""
    import stable_pretraining as spt

    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(
        **imagenet_stats, source="pixels", target="pixels"
    )
    resize = spt.data.transforms.Resize(
        img_size, source="pixels", target="pixels"
    )
    return spt.data.transforms.Compose(to_image, resize)


def encode_dataset(model, h5_path: str, n_samples: int = 2000, img_size: int = 224, device: str = "cuda"):
    """Encode sampled frames through the model's encoder."""
    transform = get_img_transform(img_size)

    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        indices = np.linspace(0, total - 1, n_samples, dtype=int)

        pixels_all = f["pixels"][indices]  # (N, 3, H, W) uint8
        action_all = f["action"][indices]
        proprio_all = f["proprio"][indices]

    # Apply ImageNet normalization
    batch = {"pixels": torch.from_numpy(pixels_all)}
    batch = transform(batch)
    pixels_tensor = batch["pixels"]  # (N, 3, 224, 224) float32

    # Encode in batches
    embeddings = []
    batch_size = 64
    model = model.to(device)

    with torch.no_grad():
        for i in range(0, len(pixels_tensor), batch_size):
            px = pixels_tensor[i : i + batch_size].to(device)
            info = {"pixels": px.unsqueeze(1)}  # (B, 1, 3, H, W)
            output = model.encode(info)
            emb = output["emb"].squeeze(1).cpu().numpy()  # (B, D)
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, action_all, proprio_all, indices


def plot_tsne_by_speed(embeddings, proprio, ax, title="t-SNE colored by Speed"):
    """t-SNE scatter colored by vehicle speed."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    speed = proprio[:, 0]  # speed is first proprio dim
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=speed, cmap="plasma", s=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label="Speed (m/s)")


def plot_tsne_by_time(embeddings, indices, ax, title="t-SNE colored by Time"):
    """t-SNE scatter colored by frame index (temporal position)."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=indices, cmap="viridis", s=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label="Frame Index")


def plot_tsne_by_yaw(embeddings, action, ax, title="t-SNE colored by Yaw Rate"):
    """t-SNE scatter colored by angular yaw (steering)."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    yaw = action[:, 2]  # angular_yaw
    vmax = np.percentile(np.abs(yaw), 95)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=yaw, cmap="RdBu_r", s=3, alpha=0.7,
                    vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label="Yaw Rate (rad/s)")


def plot_similarity_matrix(embeddings, ax, n_show=200, title="Cosine Similarity (200 frames)"):
    """Cosine similarity heatmap."""
    step = max(1, len(embeddings) // n_show)
    sub = embeddings[::step][:n_show]
    sim = cosine_similarity(sub)
    ax.imshow(sim, cmap="hot", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")


def plot_nearest_neighbors(embeddings, indices, h5_path, ax_list, query_idx=0):
    """Show query frame and its top-5 nearest neighbors."""
    sim = cosine_similarity(embeddings[query_idx : query_idx + 1], embeddings)[0]
    sim[query_idx] = -1  # exclude self
    top5 = np.argsort(sim)[-5:][::-1]

    with h5py.File(h5_path, "r") as f:
        query_img = f["pixels"][indices[query_idx]]  # (3, H, W)
        nn_imgs = [f["pixels"][indices[i]] for i in top5]

    query_img = query_img.transpose(1, 2, 0)  # (H, W, 3)
    ax_list[0].imshow(query_img)
    ax_list[0].set_title(f"Query (frame {indices[query_idx]})")
    ax_list[0].axis("off")

    for j, (nn_idx, nn_img) in enumerate(zip(top5, nn_imgs)):
        img = nn_img.transpose(1, 2, 0)
        ax_list[j + 1].imshow(img)
        ax_list[j + 1].set_title(f"NN{j+1} (f={indices[nn_idx]}, sim={sim[nn_idx]:.3f})")
        ax_list[j + 1].axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(STABLEWM_HOME / "lewm_rtb_epoch_5_object.ckpt"))
    parser.add_argument("--data", type=str, default=str(STABLEWM_HOME / "rtb/Gian-Pankyo_JT_2025-08-19_05-51-29_2111_58bb5f.h5"))
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--output", type=str, default="outputs/rtb_embedding_analysis.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt)

    print(f"Encoding {args.n_samples} frames from: {args.data}")
    embeddings, action, proprio, indices = encode_dataset(
        model, args.data, n_samples=args.n_samples, device=args.device
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: t-SNE plots
    ax1 = fig.add_subplot(gs[0, 0])
    print("Computing t-SNE by speed...")
    plot_tsne_by_speed(embeddings, proprio, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    print("Computing t-SNE by time...")
    plot_tsne_by_time(embeddings, indices, ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    print("Computing t-SNE by yaw rate...")
    plot_tsne_by_yaw(embeddings, action, ax3)

    # Row 2: Similarity matrix + stats
    ax4 = fig.add_subplot(gs[1, :2])
    plot_similarity_matrix(embeddings, ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    # Embedding norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    ax5.hist(norms, bins=50, color="steelblue", edgecolor="white")
    ax5.set_title("Embedding L2 Norm Distribution")
    ax5.set_xlabel("L2 Norm")
    ax5.set_ylabel("Count")

    # Row 3: Nearest neighbor retrieval (3 queries)
    for q, col_start in enumerate([0, 0, 0]):
        query_positions = [0, len(embeddings) // 2, len(embeddings) - 1]
        nn_axes = [fig.add_subplot(gs[2, j]) for j in range(3)]
        # Show 3 query frames with their top-1 NN
        for j, qi in enumerate(query_positions):
            sim = cosine_similarity(embeddings[qi:qi+1], embeddings)[0]
            sim[qi] = -1
            top1 = np.argmax(sim)
            with h5py.File(args.data, "r") as f:
                q_img = f["pixels"][indices[qi]].transpose(1, 2, 0)
                n_img = f["pixels"][indices[top1]].transpose(1, 2, 0)
            combined = np.concatenate([q_img, n_img], axis=1)
            nn_axes[j].imshow(combined)
            nn_axes[j].set_title(f"Query f={indices[qi]} | NN f={indices[top1]} (sim={sim[top1]:.3f})")
            nn_axes[j].axis("off")
        break

    fig.suptitle("LeWM RTB Embedding Analysis (5 epochs)", fontsize=16, y=0.98)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
