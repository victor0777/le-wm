#!/usr/bin/env python3
"""Convert RTB + OccAny depth data to LeWM HDF5 format.

Uses OccAny pts3d_local Z-axis as dense depth supervision.
Only includes frames that have OccAny depth coverage.

Usage:
    python scripts/convert_rtb_occany_to_hdf5.py --recordings b5b236 c48e71 736fcb 8014dd
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

DEFAULT_INGEST_DIR = "/mnt/phoenix-aap/ingest-output"
DEFAULT_OCCANY_DIR = "/data2/occany-inference"
DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")), "rtb_occany"
)

# Short ID → Full recording ID mapping
SHORT_TO_FULL = {}


def find_full_recording_id(short_id: str, ingest_dir: Path) -> str | None:
    """Find full recording ID from short ID."""
    for d in ingest_dir.iterdir():
        if d.is_dir() and d.name.endswith(short_id):
            return d.name
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-dir", type=str, default=DEFAULT_INGEST_DIR)
    parser.add_argument("--occany-dir", type=str, default=DEFAULT_OCCANY_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recordings", nargs="+", required=True, help="Short IDs (e.g. b5b236)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--depth-size", nargs=2, type=int, default=[64, 128], help="H W for depth")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression", type=str, default="gzip")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_and_resize_image(args: tuple) -> np.ndarray:
    jpg_path, img_size = args
    img = cv2.imread(str(jpg_path))
    if img is None:
        raise ValueError(f"Failed to load: {jpg_path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)


def extract_actions(batch_dir: Path) -> np.ndarray:
    fi = np.load(batch_dir / "velocity" / "frame_index.npy")
    vx = np.load(batch_dir / "velocity" / "vx.npy")
    vy = np.load(batch_dir / "velocity" / "vy.npy")
    ay = np.load(batch_dir / "velocity" / "angular_yaw.npy")
    n = len(fi)
    action = np.zeros((n, 3), dtype=np.float32)
    action[:, 0] = vx[fi]
    action[:, 1] = vy[fi]
    action[:, 2] = ay[fi]
    return action


def extract_proprio(batch_dir: Path) -> np.ndarray:
    vel_fi = np.load(batch_dir / "velocity" / "frame_index.npy")
    imu_fi = np.load(batch_dir / "imu" / "frame_index.npy")
    vx = np.load(batch_dir / "velocity" / "vx.npy")
    vy = np.load(batch_dir / "velocity" / "vy.npy")
    ay = np.load(batch_dir / "velocity" / "angular_yaw.npy")
    ax = np.load(batch_dir / "imu" / "accel_x.npy")
    a_y = np.load(batch_dir / "imu" / "accel_y.npy")
    az = np.load(batch_dir / "imu" / "accel_z.npy")
    gx = np.load(batch_dir / "imu" / "gyro_x.npy")
    gy = np.load(batch_dir / "imu" / "gyro_y.npy")
    gz = np.load(batch_dir / "imu" / "gyro_z.npy")
    n = len(vel_fi)
    speed = np.sqrt(vx[vel_fi] ** 2 + vy[vel_fi] ** 2)
    return np.column_stack([
        speed, ay[vel_fi],
        ax[imu_fi[:n]], a_y[imu_fi[:n]], az[imu_fi[:n]],
        gx[imu_fi[:n]], gy[imu_fi[:n]], gz[imu_fi[:n]],
    ]).astype(np.float32)


def convert_recording(short_id: str, args):
    ingest_dir = Path(args.ingest_dir)
    occany_dir = Path(args.occany_dir) / short_id
    output_dir = Path(args.output_dir)
    depth_h, depth_w = args.depth_size

    full_id = find_full_recording_id(short_id, ingest_dir)
    if not full_id:
        print(f"  ❌ {short_id}: no matching recording in ingest")
        return
    rec_dir = ingest_dir / full_id

    print(f"\n{'='*60}")
    print(f"Recording: {short_id} ({full_id})")

    # Discover batches with both ingest + OccAny data
    occany_batches = sorted([d.name for d in occany_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    print(f"  OccAny batches: {len(occany_batches)}")

    if not occany_batches:
        print("  No OccAny batches, skipping")
        return

    # Count total frames (OccAny-covered only)
    total_frames = 0
    batch_info_list = []
    for batch_name in occany_batches:
        occany_batch = occany_dir / batch_name
        ingest_batch = rec_dir / batch_name

        if not ingest_batch.exists():
            continue

        meta_path = occany_batch / "metadata.json"
        if not meta_path.exists():
            continue

        meta = json.load(open(meta_path))
        n_windows = meta["n_windows"]
        window_size = meta["window_size"]

        # Build frame index mapping: which camera frames have OccAny depth
        cam_jpgs = sorted((ingest_batch / "camera_front").glob("*.jpg"))
        covered_indices = []
        for scene in meta["scenes"]:
            start = scene["frame_start"]
            for i in range(window_size):
                idx = start + i
                if idx < len(cam_jpgs):
                    covered_indices.append(idx)

        covered_indices = sorted(set(covered_indices))
        batch_info_list.append({
            "batch_name": batch_name,
            "meta": meta,
            "covered_indices": covered_indices,
            "cam_jpgs": cam_jpgs,
        })
        total_frames += len(covered_indices)

    print(f"  Total OccAny-covered frames: {total_frames}")

    if args.dry_run:
        return

    # Create HDF5
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{full_id}.h5"
    tmp_path = output_path.with_suffix(".h5.tmp")

    with h5py.File(tmp_path, "w") as f:
        px_ds = f.create_dataset("pixels", shape=(total_frames, 3, args.img_size, args.img_size),
                                 maxshape=(None, 3, args.img_size, args.img_size),
                                 dtype=np.uint8, chunks=(1, 3, args.img_size, args.img_size),
                                 compression=args.compression)
        ac_ds = f.create_dataset("action", shape=(total_frames, 3), maxshape=(None, 3), dtype=np.float32)
        pr_ds = f.create_dataset("proprio", shape=(total_frames, 8), maxshape=(None, 8), dtype=np.float32)
        dp_ds = f.create_dataset("depth_maps", shape=(total_frames, depth_h, depth_w),
                                 maxshape=(None, depth_h, depth_w), dtype=np.float32,
                                 chunks=(1, depth_h, depth_w), compression=args.compression)
        cf_ds = f.create_dataset("depth_conf", shape=(total_frames, depth_h, depth_w),
                                 maxshape=(None, depth_h, depth_w), dtype=np.float32,
                                 chunks=(1, depth_h, depth_w), compression=args.compression)

        ep_offsets = []
        ep_lengths = []
        global_offset = 0

        for bi in tqdm(batch_info_list, desc=f"  {short_id}", unit="batch"):
            batch_name = bi["batch_name"]
            meta = bi["meta"]
            covered = bi["covered_indices"]
            cam_jpgs = bi["cam_jpgs"]
            ingest_batch = rec_dir / batch_name
            occany_batch = occany_dir / batch_name

            if not covered:
                continue

            ep_offsets.append(global_offset)

            # Load OccAny depth
            pts3d = np.load(occany_batch / "pts3d_local.npy", mmap_mode="r")
            conf = np.load(occany_batch / "conf.npy", mmap_mode="r")

            # Load actions/proprio for full batch
            action = extract_actions(ingest_batch)
            proprio = extract_proprio(ingest_batch)

            # Process covered frames
            n_written = 0
            for idx in covered:
                if idx >= len(cam_jpgs) or idx >= len(action):
                    continue

                # Find which OccAny window/frame this index belongs to
                window_idx = None
                frame_in_window = None
                for si, scene in enumerate(meta["scenes"]):
                    start = scene["frame_start"]
                    if start <= idx < start + meta["window_size"]:
                        window_idx = si
                        frame_in_window = idx - start
                        break

                if window_idx is None:
                    continue

                # Image
                img = load_and_resize_image((cam_jpgs[idx], args.img_size))

                # OccAny depth (Z-axis of pts3d_local)
                depth_full = pts3d[window_idx, frame_in_window, :, :, 2]  # (272, 512)
                conf_full = conf[window_idx, frame_in_window]  # (272, 512)

                # Resize depth and conf
                depth_resized = cv2.resize(depth_full, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
                conf_resized = cv2.resize(conf_full, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

                # Write
                px_ds[global_offset] = img
                ac_ds[global_offset] = action[idx]
                pr_ds[global_offset] = proprio[idx]
                dp_ds[global_offset] = depth_resized.astype(np.float32)
                cf_ds[global_offset] = conf_resized.astype(np.float32)

                global_offset += 1
                n_written += 1

            ep_lengths.append(n_written)

        # Trim
        if global_offset < total_frames:
            for ds in [px_ds, ac_ds, pr_ds, dp_ds, cf_ds]:
                ds.resize(global_offset, axis=0)

        f.create_dataset("ep_len", data=np.array(ep_lengths, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.array(ep_offsets, dtype=np.int64))

    tmp_path.rename(output_path)
    size_gb = output_path.stat().st_size / 1e9
    print(f"  Written: {output_path} ({size_gb:.2f} GB, {global_offset} frames)")


def main():
    args = parse_args()
    print(f"Ingest: {args.ingest_dir}")
    print(f"OccAny: {args.occany_dir}")
    print(f"Output: {args.output_dir}")

    for short_id in args.recordings:
        convert_recording(short_id, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
