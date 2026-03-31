#!/usr/bin/env python3
"""Auto-labeling benchmark: compare L1 vs L3 encoder for scene classification.

Uses rtb-vlm segments.json road_type labels as ground truth.
Tests NN-based auto-labeling accuracy with different encoder checkpoints.

Usage:
    python scripts/auto_labeling_benchmark.py
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

STABLEWM_HOME = Path.home() / ".stable_worldmodel"

# rtb-vlm segments.json for Gian-Pankyo da8241
SEGMENTS_PATH = Path("/home/ktl/projects/rtb-vlm/experiments/exp10_segments/results/segments.json")


def load_model(ckpt_path: str, device: str = "cuda"):
    import sys
    import __main__
    try:
        from train_vp import LaneDecoder, DepthDecoder
        # Register in __main__ so torch.load unpickler can find them
        __main__.LaneDecoder = LaneDecoder
        __main__.DepthDecoder = DepthDecoder
    except ImportError:
        pass
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def get_img_transform(img_size: int = 224):
    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
    resize = spt.data.transforms.Resize(img_size, source="pixels", target="pixels")
    return spt.data.transforms.Compose(to_image, resize)


def encode_frames(model, h5_path: str, indices: list[int], transform, device: str, batch_size: int = 64):
    """Encode specific frame indices."""
    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"][indices]  # (N, 3, H, W)

    batch = {"pixels": torch.from_numpy(pixels)}
    batch = transform(batch)
    px = batch["pixels"]

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(px), batch_size):
            chunk = px[i:i + batch_size].to(device)
            info = {"pixels": chunk.unsqueeze(1)}
            output = model.encode(info)
            emb = output["emb"].squeeze(1).cpu().numpy()
            embeddings.append(emb)

    return np.concatenate(embeddings, axis=0)


def load_road_type_labels(segments_path: Path, fps: float = 10.0) -> dict[int, str]:
    """Load road_type labels from segments.json. Returns {frame_idx: label}."""
    data = json.load(open(segments_path))
    road_segments = data["field_segments"]["road_type"]

    frame_labels = {}
    for seg in road_segments:
        start_frame = int(seg["start_s"] * fps)
        end_frame = int(seg["end_s"] * fps)
        label = seg["label"]
        for f in range(start_frame, end_frame + 1):
            frame_labels[f] = label

    return frame_labels


def nn_classify(anchor_embs: np.ndarray, anchor_labels: list[str],
                query_embs: np.ndarray) -> list[str]:
    """Classify queries by nearest neighbor in anchor set."""
    # Cosine similarity
    anchor_norm = anchor_embs / (np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    sim = query_norm @ anchor_norm.T  # (Q, A)
    nn_indices = sim.argmax(axis=1)
    return [anchor_labels[i] for i in nn_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=[
        str(STABLEWM_HOME / "lewm_rtb_multi_epoch_9_object.ckpt"),
        str(STABLEWM_HOME / "lewm_L3_occany_depth_epoch_10_object.ckpt"),
    ])
    parser.add_argument("--labels", nargs="+", default=["L1 (baseline)", "L3 (OccAny depth)"])
    parser.add_argument("--segments", type=str, default=str(SEGMENTS_PATH))
    parser.add_argument("--output", type=str, default="outputs/auto_labeling_benchmark.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load road_type labels
    frame_labels = load_road_type_labels(Path(args.segments))
    unique_labels = sorted(set(frame_labels.values()))
    print(f"Road type labels: {unique_labels}")
    print(f"Labeled frames: {len(frame_labels)}")

    # Find HDF5 that covers these frames
    # da8241 is the Gian-Pankyo recording used by rtb-vlm segments
    h5_candidates = [
        STABLEWM_HOME / "rtb/Gian-Pankyo_JT_2025-08-19_05-51-29_2111_58bb5f.h5",
        STABLEWM_HOME / "rtb/Gian-Pankyo_JT_2025-08-19_04-59-28_2111_da8241.h5",
    ]
    h5_path = None
    for p in h5_candidates:
        if p.exists():
            h5_path = str(p)
            break

    if not h5_path:
        # Use any available HDF5
        rtb_dir = STABLEWM_HOME / "rtb"
        h5_path = str(sorted(rtb_dir.glob("*.h5"))[0])

    print(f"Using HDF5: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        total_frames = f["pixels"].shape[0]
    print(f"Total frames in HDF5: {total_frames}")

    # Map frame labels to available frame range
    valid_frames = {f: l for f, l in frame_labels.items() if f < total_frames}
    print(f"Valid labeled frames: {len(valid_frames)}")

    if len(valid_frames) < 50:
        print("Not enough labeled frames for meaningful benchmark")
        return

    # Split: first 70% as anchor, last 30% as query (temporal disjoint)
    sorted_frames = sorted(valid_frames.keys())
    split_point = int(len(sorted_frames) * 0.7)
    anchor_frames = sorted_frames[:split_point]
    query_frames = sorted_frames[split_point:]

    # Subsample for speed
    max_anchors = 500
    max_queries = 300
    if len(anchor_frames) > max_anchors:
        anchor_frames = [anchor_frames[i] for i in np.linspace(0, len(anchor_frames) - 1, max_anchors, dtype=int)]
    if len(query_frames) > max_queries:
        query_frames = [query_frames[i] for i in np.linspace(0, len(query_frames) - 1, max_queries, dtype=int)]

    anchor_labels_list = [valid_frames[f] for f in anchor_frames]
    query_labels_gt = [valid_frames[f] for f in query_frames]

    print(f"Anchor frames: {len(anchor_frames)}, Query frames: {len(query_frames)}")
    print(f"Anchor label distribution: {dict(zip(*np.unique(anchor_labels_list, return_counts=True)))}")

    transform = get_img_transform()
    results = {}

    for ckpt_path, model_name in zip(args.checkpoints, args.labels):
        if not Path(ckpt_path).exists():
            print(f"  Skipping {model_name}: checkpoint not found")
            continue

        print(f"\n=== {model_name} ===")
        model = load_model(ckpt_path, args.device)

        anchor_embs = encode_frames(model, h5_path, anchor_frames, transform, args.device)
        query_embs = encode_frames(model, h5_path, query_frames, transform, args.device)

        predicted = nn_classify(anchor_embs, anchor_labels_list, query_embs)
        acc = accuracy_score(query_labels_gt, predicted)
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(query_labels_gt, predicted, zero_division=0))

        results[model_name] = {
            "accuracy": acc,
            "predicted": predicted,
            "gt": query_labels_gt,
        }

        del model
        torch.cuda.empty_cache()

    # Plot
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 5))

    # Bar chart of accuracies
    ax = axes[0]
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    colors = ["#3498db", "#e74c3c", "#2ecc71"][:len(names)]
    bars = ax.bar(names, accs, color=colors, edgecolor="white")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.set_title("NN Auto-Labeling Accuracy")
    ax.set_ylim(0, 1.1)

    # Confusion matrices
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i + 1]
        labels_unique = sorted(set(res["gt"] + res["predicted"]))
        cm = confusion_matrix(res["gt"], res["predicted"], labels=labels_unique)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels_unique)))
        ax.set_xticklabels(labels_unique, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels_unique)))
        ax.set_yticklabels(labels_unique, fontsize=8)
        ax.set_title(f"{name}\n(acc={res['accuracy']:.1%})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Auto-Labeling Benchmark: Road Type Classification", fontsize=14)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
