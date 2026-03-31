"""Track 2: Token-conditioned training with maneuver pseudo-labels.

Tests whether maneuver tokens (left/right/straight/stop/accel/decel)
make action useful cross-domain by providing route-invariant semantic
conditioning alongside ego-motion.

The maneuver token embedding is ADDED to the action embedding before
feeding to the predictor, so the predictor sees:
    visual_embedding + (action_embedding + maneuver_embedding)

Usage:
    # V1: Livlab 3 train -> 8014dd holdout
    python train_vp_token.py data=rtb_occany \
        +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
        wandb.enabled=False trainer.devices=1 subdir=track2_v1

    # V2: da8241 train -> 8014dd holdout (cross-domain)
    python train_vp_token.py data=rtb_occany \
        +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
        +exclude='[Livlab-Rt-C-1,Livlab-Rt-C-3,Livlab-Rt-C-5]' \
        wandb.enabled=False trainer.devices=1 subdir=track2_v2

    # V3: Livlab 4 train -> da8241 holdout (cross-domain reverse)
    python train_vp_token.py data=rtb_occany \
        +holdout=Gian-Pankyo_JT_2025-08-19_04-59-28_2111_da8241 \
        wandb.enabled=False trainer.devices=1 subdir=track2_v3
"""

from functools import partial
from pathlib import Path

import h5py
import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from train_multi import load_multi_recording_datasets
from train_vp import DepthDecoder
from utils import get_img_preprocessor, ModelObjectCallBack

LABELS_DIR = Path.home() / ".stable_worldmodel" / "rtb_occany_labels"
NUM_MANEUVER_CLASSES = 6  # left=0, right=1, straight=2, stop=3, accel=4, decel=5


class ManeuverLabelWrapper(torch.utils.data.Dataset):
    """Wraps an HDF5Dataset to inject maneuver labels into each batch.

    The underlying HDF5Dataset returns sequences of length num_steps,
    with pixels subsampled by frameskip.  For a clip at (ep_idx, start),
    the pixel frames are at global indices:
        offset[ep_idx] + start + t * frameskip   for t in 0..num_steps-1

    We load the per-frame maneuver labels from .npz files and index them
    at the same global positions.
    """

    def __init__(self, hdf5_dataset: swm.data.HDF5Dataset, labels_dir: Path = LABELS_DIR):
        self.ds = hdf5_dataset
        self.num_steps = hdf5_dataset.num_steps
        self.frameskip = hdf5_dataset.frameskip

        # Extract short_id from h5 filename (last 6 chars before .h5)
        h5_stem = hdf5_dataset.h5_path.stem  # e.g. "Livlab-Rt-C-1_..._b5b236"
        short_id = h5_stem.split("_")[-1]    # e.g. "b5b236"

        label_path = labels_dir / f"{short_id}_labels.npz"
        if not label_path.exists():
            raise FileNotFoundError(f"Maneuver labels not found: {label_path}")

        data = np.load(label_path, allow_pickle=True)
        self.all_labels = data["labels"].astype(np.int64)  # (total_frames,)

        # Get total frames in HDF5 for validation
        with h5py.File(hdf5_dataset.h5_path, "r") as f:
            total_frames = f["pixels"].shape[0]

        if len(self.all_labels) != total_frames:
            raise ValueError(
                f"Label count ({len(self.all_labels)}) != HDF5 frame count ({total_frames}) "
                f"for {short_id}"
            )

        # Store offsets for global indexing
        self.offsets = hdf5_dataset.offsets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # Replicate clip_indices logic: (ep_idx, start) -> global frame indices
        ep_idx, start = self.ds.clip_indices[idx]
        g_start = self.offsets[ep_idx] + start

        # Get maneuver label for each of the num_steps pixel frames
        maneuver_ids = []
        for t in range(self.num_steps):
            frame_idx = g_start + t * self.frameskip
            if frame_idx < len(self.all_labels):
                maneuver_ids.append(self.all_labels[frame_idx])
            else:
                maneuver_ids.append(2)  # default: straight

        sample["maneuver"] = torch.tensor(maneuver_ids, dtype=torch.long)  # (T,)
        return sample

    # Delegate attribute access to underlying dataset
    def __getattr__(self, name):
        if name in ("ds", "num_steps", "frameskip", "all_labels", "offsets"):
            raise AttributeError(name)
        return getattr(self.ds, name)


def lejepa_token_forward(self, batch, stage, cfg):
    """Forward with maneuver token conditioning + depth auxiliary loss."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    lambda_depth = cfg.loss.get("depth_weight", 0.1)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]           # (B, T, D)
    act_emb = output["act_emb"]   # (B, T, D)

    # Add maneuver token embedding to action embedding
    if "maneuver" in batch:
        maneuver_emb = self.model.maneuver_embedding(batch["maneuver"])  # (B, T, D)
        act_emb = act_emb + maneuver_emb

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Main losses
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # Depth auxiliary loss
    B, T, D = emb.shape
    emb_flat = rearrange(emb, "b t d -> (b t) d")
    total_aux_loss = torch.tensor(0.0, device=emb.device)

    depth_dec = getattr(self.model, "depth_decoder", None)
    if "depth_maps" in batch and depth_dec is not None:
        depth_pred = depth_dec(emb_flat).squeeze(1)  # (B*T, 64, 128)
        depth_gt = rearrange(batch["depth_maps"].float(), "b t ... -> (b t) ...")
        valid = depth_gt > 0
        if valid.any():
            if "depth_conf" in batch:
                conf = rearrange(batch["depth_conf"].float(), "b t ... -> (b t) ...")
                conf_norm = conf / (conf.max() + 1e-8)
                weighted_diff = conf_norm[valid] * (depth_pred[valid] - depth_gt[valid]).abs()
                depth_loss = weighted_diff.mean()
            else:
                depth_loss = nn.functional.l1_loss(depth_pred[valid], depth_gt[valid])
        else:
            depth_loss = torch.tensor(0.0, device=emb.device)
        output["depth_loss"] = depth_loss
        total_aux_loss = total_aux_loss + lambda_depth * depth_loss

    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"] + total_aux_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    holdout = cfg.get("holdout", "")
    if not holdout:
        raise ValueError("Must specify holdout=RECORDING_NAME")

    with open_dict(cfg):
        if not hasattr(cfg, "data"):
            cfg.data = {"dataset": {}}
        if "depth_weight" not in cfg.loss:
            cfg.loss.depth_weight = 0.1

    #########################
    ##       dataset       ##
    #########################

    exclude = list(cfg.get("exclude", []))
    print(f"\nLoading recordings (holdout: {holdout}, exclude: {exclude}):")
    train_ds, val_ds, train_datasets_list, val_datasets_list = load_multi_recording_datasets(
        cfg, holdout, exclude_names=exclude
    )
    print(f"\nTrain: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    # Check if depth data available
    has_depth = "depth_maps" in cfg.data.dataset.keys_to_load
    print(f"VP data: depth_maps={has_depth}")

    # Wrap datasets with maneuver labels
    print("\nLoading maneuver labels:")
    wrapped_train = []
    for ds in train_ds.datasets:
        w = ManeuverLabelWrapper(ds)
        wrapped_train.append(w)
        short_id = ds.h5_path.stem.split("_")[-1]
        print(f"  {short_id}: {len(w.all_labels)} labels, {len(w)} sequences")

    wrapped_val = []
    for ds in val_ds.datasets:
        w = ManeuverLabelWrapper(ds)
        wrapped_val.append(w)
        short_id = ds.h5_path.stem.split("_")[-1]
        print(f"  {short_id}: {len(w.all_labels)} labels (val), {len(w)} sequences")

    # Fit normalizers on train data only
    ref_ds = train_datasets_list[0]
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col in ("depth_maps", "depth_conf"):
                continue

            all_col_data = []
            for ds in train_datasets_list:
                col_data = ds.get_col_data(col)
                all_col_data.append(torch.from_numpy(np.array(col_data)))

            data = torch.cat(all_col_data, dim=0)
            data = data[~torch.isnan(data).any(dim=1)]
            mean = data.mean(0, keepdim=True).clone()
            std = data.std(0, keepdim=True).clone()

            def make_norm_fn(m, s):
                def norm_fn(x):
                    return ((x - m) / s).float()
                return norm_fn

            normalizer = spt.data.transforms.WrapTorchTransform(
                make_norm_fn(mean, std), source=col, target=col
            )
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", ref_ds.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)

    # Apply transform to underlying HDF5Datasets (wrapper delegates)
    for w in wrapped_train:
        w.ds.transform = transform
    for w in wrapped_val:
        w.ds.transform = transform

    train_concat = torch.utils.data.ConcatDataset(wrapped_train)
    val_concat = torch.utils.data.ConcatDataset(wrapped_val)

    # Build loader kwargs, handling num_workers=0 compatibility
    loader_kwargs = dict(cfg.loader)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(
        train_concat, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_concat, **loader_kwargs, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)

    world_model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=predictor_proj,
    )

    # Maneuver token embedding (6 classes -> embed_dim)
    world_model.maneuver_embedding = nn.Embedding(NUM_MANEUVER_CLASSES, embed_dim)
    nn.init.normal_(world_model.maneuver_embedding.weight, std=0.02)
    print(f"\nManeuver embedding: {NUM_MANEUVER_CLASSES} classes -> {embed_dim}D")

    # Depth decoder
    depth_decoder = DepthDecoder(embed_dim) if has_depth else None
    if depth_decoder:
        world_model.depth_decoder = depth_decoder

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_token_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or "track2"
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer, callbacks=[object_dump_callback],
        num_sanity_val_steps=1, logger=logger, enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
