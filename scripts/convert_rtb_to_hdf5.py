#!/usr/bin/env python3
"""Convert rosbag-ingest output to LeWM HDF5 training format.

Usage:
    # Single recording
    python scripts/convert_rtb_to_hdf5.py --recordings Gian-Pankyo_JT_2025-02-11_04-29-01_2111_2ebc9f

    # All recordings
    python scripts/convert_rtb_to_hdf5.py

    # Dry run (print stats only)
    python scripts/convert_rtb_to_hdf5.py --dry-run

    # Parallel (via GNU parallel)
    ls /mnt/phoenix-aap/ingest-output/ | parallel -j4 \
        python scripts/convert_rtb_to_hdf5.py --recordings {}
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
DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")), "rtb"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RTB ingest data to LeWM HDF5")
    parser.add_argument("--ingest-dir", type=str, default=DEFAULT_INGEST_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recordings", nargs="*", default=None, help="Recording IDs (default: all)")
    parser.add_argument("--camera", type=str, default="camera_front")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--gap-threshold", type=float, default=0.2, help="Seconds gap to split episodes")
    parser.add_argument("--workers", type=int, default=8, help="Parallel image workers")
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", None])
    parser.add_argument("--action-4d", action="store_true", help="Include Δψ as 4th action channel")
    parser.add_argument("--vp-dir", type=str, default="/data2/vp-inference", help="VP inference dir")
    parser.add_argument("--include-vp", action="store_true", help="Include VP lane_masks and depth_maps")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already converted recordings")
    return parser.parse_args()


def discover_batches(recording_dir: Path) -> list[dict]:
    """Discover and sort batches by start_time."""
    batches = []
    for batch_dir in sorted(recording_dir.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        info_path = batch_dir / "batch_info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        info["path"] = batch_dir
        batches.append(info)
    batches.sort(key=lambda b: b["start_time"])
    return batches


def detect_episodes(batches: list[dict], gap_threshold: float) -> list[list[dict]]:
    """Group batches into episodes based on temporal gaps."""
    if not batches:
        return []
    episodes = [[batches[0]]]
    for i in range(1, len(batches)):
        gap = batches[i]["start_time"] - batches[i - 1]["end_time"]
        if gap > gap_threshold:
            episodes.append([batches[i]])
        else:
            episodes[-1].append(batches[i])
    return episodes


def validate_batch(batch_dir: Path, camera: str) -> bool:
    """Check that a batch has all required sensor data."""
    required = [
        batch_dir / camera,
        batch_dir / "velocity" / "vx.npy",
        batch_dir / "velocity" / "vy.npy",
        batch_dir / "velocity" / "angular_yaw.npy",
        batch_dir / "velocity" / "frame_index.npy",
        batch_dir / "imu" / "accel_x.npy",
        batch_dir / "imu" / "frame_index.npy",
    ]
    return all(p.exists() for p in required)


def quaternion_to_yaw(w, x, y, z):
    """Extract yaw angle from quaternion. Returns yaw in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def extract_actions(batch_dir: Path, include_delta_yaw: bool = False) -> np.ndarray:
    """Extract action vectors aligned to camera frames.

    Returns (N, 3) float32 if include_delta_yaw=False,
            (N, 4) float32 if include_delta_yaw=True (adds Δψ from IMU quaternion).
    """
    frame_index = np.load(batch_dir / "velocity" / "frame_index.npy")
    vx = np.load(batch_dir / "velocity" / "vx.npy")
    vy = np.load(batch_dir / "velocity" / "vy.npy")
    angular_yaw = np.load(batch_dir / "velocity" / "angular_yaw.npy")

    n_frames = len(frame_index)
    action_dim = 4 if include_delta_yaw else 3
    action = np.zeros((n_frames, action_dim), dtype=np.float32)
    action[:, 0] = vx[frame_index]
    action[:, 1] = vy[frame_index]
    action[:, 2] = angular_yaw[frame_index]

    if include_delta_yaw:
        imu_fi = np.load(batch_dir / "imu" / "frame_index.npy")
        ori_w = np.load(batch_dir / "imu" / "ori_w.npy")
        ori_x = np.load(batch_dir / "imu" / "ori_x.npy")
        ori_y = np.load(batch_dir / "imu" / "ori_y.npy")
        ori_z = np.load(batch_dir / "imu" / "ori_z.npy")

        # Yaw at each camera frame from IMU quaternion
        yaw = quaternion_to_yaw(
            ori_w[imu_fi[:n_frames]], ori_x[imu_fi[:n_frames]],
            ori_y[imu_fi[:n_frames]], ori_z[imu_fi[:n_frames]],
        )
        # Unwrap to handle ±π discontinuity, then diff
        yaw_unwrapped = np.unwrap(yaw)
        delta_yaw = np.diff(yaw_unwrapped, prepend=yaw_unwrapped[0])
        action[:, 3] = delta_yaw.astype(np.float32)

    return action


def extract_proprio(batch_dir: Path) -> np.ndarray:
    """Extract proprioceptive state aligned to camera frames. Returns (N, 8) float32."""
    vel_fi = np.load(batch_dir / "velocity" / "frame_index.npy")
    imu_fi = np.load(batch_dir / "imu" / "frame_index.npy")

    vx = np.load(batch_dir / "velocity" / "vx.npy")
    vy = np.load(batch_dir / "velocity" / "vy.npy")
    angular_yaw = np.load(batch_dir / "velocity" / "angular_yaw.npy")

    accel_x = np.load(batch_dir / "imu" / "accel_x.npy")
    accel_y = np.load(batch_dir / "imu" / "accel_y.npy")
    accel_z = np.load(batch_dir / "imu" / "accel_z.npy")
    gyro_x = np.load(batch_dir / "imu" / "gyro_x.npy")
    gyro_y = np.load(batch_dir / "imu" / "gyro_y.npy")
    gyro_z = np.load(batch_dir / "imu" / "gyro_z.npy")

    n_frames = len(vel_fi)
    speed = np.sqrt(vx[vel_fi] ** 2 + vy[vel_fi] ** 2)
    heading_rate = angular_yaw[vel_fi]

    proprio = np.column_stack([
        speed,
        heading_rate,
        accel_x[imu_fi[:n_frames]],
        accel_y[imu_fi[:n_frames]],
        accel_z[imu_fi[:n_frames]],
        gyro_x[imu_fi[:n_frames]],
        gyro_y[imu_fi[:n_frames]],
        gyro_z[imu_fi[:n_frames]],
    ]).astype(np.float32)
    return proprio


def load_and_resize_image(args: tuple) -> np.ndarray:
    """Load a JPG, resize, return uint8 CHW array."""
    jpg_path, img_size = args
    img = cv2.imread(str(jpg_path))
    if img is None:
        raise ValueError(f"Failed to load image: {jpg_path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)  # (3, H, W)


def extract_pixels(batch_dir: Path, camera: str, img_size: int, workers: int) -> np.ndarray:
    """Load and resize all images in a batch. Returns (N, 3, H, W) uint8."""
    cam_dir = batch_dir / camera
    jpg_files = sorted(cam_dir.glob("*.jpg"))
    if not jpg_files:
        jpg_files = sorted(cam_dir.glob("*.png"))
    if not jpg_files:
        raise FileNotFoundError(f"No images found in {cam_dir}")

    tasks = [(f, img_size) for f in jpg_files]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(load_and_resize_image, tasks))

    return np.stack(results, axis=0)


def extract_vp(vp_batch_dir: Path, lane_size: tuple = (80, 160), depth_size: tuple = (64, 128)) -> dict | None:
    """Extract VP lane masks and depth maps for a batch. Returns dict or None if not available."""
    results_dir = vp_batch_dir / "inference_results"
    if not results_dir.exists():
        return None

    lane_path = results_dir / "lane_masks.npy"
    depth_path = results_dir / "depth_maps.npy"
    if not lane_path.exists() or not depth_path.exists():
        return None

    lanes = np.load(lane_path)  # (N, 3, H, W)
    depth = np.load(depth_path, mmap_mode="r")  # (N, H, W)

    # Resize lanes to fixed size for HDF5 storage
    if lanes.shape[2:] != lane_size:
        resized = np.zeros((len(lanes), lanes.shape[1], *lane_size), dtype=np.float32)
        for i in range(len(lanes)):
            for c in range(lanes.shape[1]):
                resized[i, c] = cv2.resize(lanes[i, c], (lane_size[1], lane_size[0]), interpolation=cv2.INTER_LINEAR)
        lanes = resized

    # Resize depth to compact size
    if depth.shape[1:] != depth_size:
        resized_d = np.zeros((len(depth), *depth_size), dtype=np.float32)
        for i in range(len(depth)):
            resized_d[i] = cv2.resize(depth[i], (depth_size[1], depth_size[0]), interpolation=cv2.INTER_LINEAR)
        depth = resized_d
    else:
        depth = np.array(depth)

    return {"lane_masks": lanes.astype(np.float32), "depth_maps": depth.astype(np.float32)}


def count_frames(batches: list[dict], camera: str) -> int:
    """Count total camera frames across batches."""
    total = 0
    for b in batches:
        cam_dir = b["path"] / camera
        total += len(list(cam_dir.glob("*.jpg"))) or len(list(cam_dir.glob("*.png")))
    return total


def convert_recording(
    recording_dir: Path,
    output_path: Path,
    camera: str,
    img_size: int,
    gap_threshold: float,
    workers: int,
    compression: str | None,
    dry_run: bool,
    action_4d: bool = False,
    vp_dir: Path | None = None,
):
    """Convert a single recording to HDF5."""
    recording_id = recording_dir.name
    include_vp = vp_dir is not None and (vp_dir / recording_id).is_dir()
    print(f"\n{'='*60}")
    print(f"Recording: {recording_id}")

    # Phase 1: Discovery
    batches = discover_batches(recording_dir)
    if not batches:
        print("  No batches found, skipping.")
        return

    # Validate batches
    valid_batches = [b for b in batches if validate_batch(b["path"], camera)]
    skipped = len(batches) - len(valid_batches)
    if skipped:
        print(f"  Skipping {skipped}/{len(batches)} batches (missing sensor data)")
    batches = valid_batches
    if not batches:
        print("  No valid batches, skipping.")
        return

    # Detect episodes
    episodes = detect_episodes(batches, gap_threshold)
    total_frames = count_frames(batches, camera)

    print(f"  Batches: {len(batches)}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    est_gb = total_frames * 3 * img_size * img_size / 1e9
    print(f"  Estimated raw size: {est_gb:.2f} GB")

    if dry_run:
        return

    # Phase 2: Write HDF5 batch-by-batch
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".h5.tmp")

    ep_lengths = []
    ep_offsets = []
    global_offset = 0

    with h5py.File(tmp_path, "w") as f:
        # Pre-allocate datasets
        px_ds = f.create_dataset(
            "pixels",
            shape=(total_frames, 3, img_size, img_size),
            maxshape=(None, 3, img_size, img_size),
            dtype=np.uint8,
            chunks=(1, 3, img_size, img_size),
            compression=compression,
        )
        act_dim = 4 if action_4d else 3
        ac_ds = f.create_dataset(
            "action",
            shape=(total_frames, act_dim),
            maxshape=(None, act_dim),
            dtype=np.float32,
        )
        pr_ds = f.create_dataset(
            "proprio",
            shape=(total_frames, 8),
            maxshape=(None, 8),
            dtype=np.float32,
        )

        # VP datasets (optional)
        lane_ds = None
        depth_ds = None
        if include_vp:
            lane_ds = f.create_dataset(
                "lane_masks",
                shape=(total_frames, 3, 80, 160),
                maxshape=(None, 3, 80, 160),
                dtype=np.float32,
                chunks=(1, 3, 80, 160),
                compression=compression,
            )
            depth_ds = f.create_dataset(
                "depth_maps",
                shape=(total_frames, 64, 128),
                maxshape=(None, 64, 128),
                dtype=np.float32,
                chunks=(1, 64, 128),
                compression=compression,
            )

        for ep_idx, ep_batches in enumerate(episodes):
            ep_start = global_offset
            ep_frame_count = 0

            for batch_info in tqdm(
                ep_batches,
                desc=f"  Episode {ep_idx + 1}/{len(episodes)}",
                unit="batch",
            ):
                batch_dir = batch_info["path"]

                # Extract sensor data
                action = extract_actions(batch_dir, include_delta_yaw=action_4d)
                proprio = extract_proprio(batch_dir)
                pixels = extract_pixels(batch_dir, camera, img_size, workers)

                # Extract VP data if available
                vp_data = None
                if include_vp:
                    batch_name = batch_dir.name  # e.g. "batch_000"
                    vp_batch_dir = vp_dir / recording_id / batch_name
                    vp_data = extract_vp(vp_batch_dir)

                # Align lengths (use minimum across sensors)
                n = min(len(pixels), len(action), len(proprio))
                if vp_data is not None:
                    n = min(n, len(vp_data["lane_masks"]), len(vp_data["depth_maps"]))
                if n == 0:
                    continue

                pixels = pixels[:n]
                action = action[:n]
                proprio = proprio[:n]

                # Write to HDF5
                px_ds[global_offset : global_offset + n] = pixels
                ac_ds[global_offset : global_offset + n] = action
                pr_ds[global_offset : global_offset + n] = proprio

                if vp_data is not None and lane_ds is not None:
                    lane_ds[global_offset : global_offset + n] = vp_data["lane_masks"][:n]
                    depth_ds[global_offset : global_offset + n] = vp_data["depth_maps"][:n]

                global_offset += n
                ep_frame_count += n

            ep_lengths.append(ep_frame_count)
            ep_offsets.append(ep_start)

        # Trim if fewer frames than pre-allocated (due to alignment)
        if global_offset < total_frames:
            px_ds.resize(global_offset, axis=0)
            ac_ds.resize(global_offset, axis=0)
            pr_ds.resize(global_offset, axis=0)
            if lane_ds is not None:
                lane_ds.resize(global_offset, axis=0)
                depth_ds.resize(global_offset, axis=0)

        # Write episode metadata
        f.create_dataset("ep_len", data=np.array(ep_lengths, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.array(ep_offsets, dtype=np.int64))

    # Atomic rename
    tmp_path.rename(output_path)
    size_gb = output_path.stat().st_size / 1e9
    print(f"  Written: {output_path} ({size_gb:.2f} GB, {global_offset} frames)")


def main():
    args = parse_args()
    ingest_dir = Path(args.ingest_dir)
    output_dir = Path(args.output_dir)

    # Discover recordings
    if args.recordings:
        recording_ids = args.recordings
    else:
        recording_ids = sorted(
            d.name for d in ingest_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    print(f"Input:  {ingest_dir}")
    print(f"Output: {output_dir}")
    print(f"Camera: {args.camera}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Recordings: {len(recording_ids)}")

    for rec_id in recording_ids:
        rec_dir = ingest_dir / rec_id
        if not rec_dir.is_dir():
            print(f"Warning: {rec_dir} not found, skipping")
            continue

        output_path = output_dir / f"{rec_id}.h5"
        if args.skip_existing and output_path.exists():
            print(f"Skipping {rec_id} (already exists)")
            continue

        convert_recording(
            recording_dir=rec_dir,
            output_path=output_path,
            camera=args.camera,
            img_size=args.img_size,
            gap_threshold=args.gap_threshold,
            workers=args.workers,
            compression=args.compression,
            dry_run=args.dry_run,
            action_4d=args.action_4d,
            vp_dir=Path(args.vp_dir) if args.include_vp else None,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
