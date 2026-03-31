"""Training with VP auxiliary supervision (Level 2).

Adds lane mask and depth map prediction losses to force the encoder
to learn lane structure and 3D depth understanding.

Usage:
    python train_vp.py data=rtb_vp +holdout=Livlab-Rt-C-7_JT_2025-09-22_07-38-12_2111_8014dd \
        wandb.enabled=False trainer.max_epochs=10 trainer.devices=1
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
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
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


class LaneDecoder(nn.Module):
    """Lightweight decoder: embedding → lane mask (3, 80, 160)."""

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 5 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 10x20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 20x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 40x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 80x160
        )

    def forward(self, emb):
        """emb: (B, D) → (B, 3, 80, 160)"""
        x = self.fc(emb).view(-1, 256, 5, 10)
        return self.decoder(x)


class DepthDecoder(nn.Module):
    """Lightweight decoder: embedding → depth map (1, 64, 128)."""

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 4 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 64x128
        )

    def forward(self, emb):
        """emb: (B, D) → (B, 1, 64, 128)"""
        x = self.fc(emb).view(-1, 256, 4, 8)
        return self.decoder(x)


def lejepa_vp_forward(self, batch, stage, cfg):
    """Forward with VP auxiliary losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    lambda_lane = cfg.loss.get("lane_weight", 0.1)
    lambda_depth = cfg.loss.get("depth_weight", 0.1)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Main losses
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # VP auxiliary losses
    B, T, D = emb.shape
    emb_flat = rearrange(emb, "b t d -> (b t) d")

    total_aux_loss = torch.tensor(0.0, device=emb.device)

    lane_dec = getattr(self.model, 'lane_decoder', None)
    depth_dec = getattr(self.model, 'depth_decoder', None)

    if "lane_masks" in batch and lane_dec is not None:
        lane_pred = lane_dec(emb_flat)  # (B*T, 3, 80, 160)
        lane_gt = rearrange(batch["lane_masks"].float(), "b t ... -> (b t) ...")
        # Use L1 loss (lane masks have varying scales)
        lane_loss = nn.functional.l1_loss(lane_pred, lane_gt)
        output["lane_loss"] = lane_loss
        total_aux_loss = total_aux_loss + lambda_lane * lane_loss

    if "depth_maps" in batch and depth_dec is not None:
        depth_pred = depth_dec(emb_flat).squeeze(1)  # (B*T, 64, 128)
        depth_gt = rearrange(batch["depth_maps"].float(), "b t ... -> (b t) ...")
        # Mask out invalid depth (<=0)
        valid = depth_gt > 0
        if valid.any():
            # Confidence-weighted loss if available
            if "depth_conf" in batch:
                conf = rearrange(batch["depth_conf"].float(), "b t ... -> (b t) ...")
                # Normalize confidence to [0, 1] range
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
        # VP loss weights
        if "lane_weight" not in cfg.loss:
            cfg.loss.lane_weight = 0.1
        if "depth_weight" not in cfg.loss:
            cfg.loss.depth_weight = 0.1

    #########################
    ##       dataset       ##
    #########################

    exclude = list(cfg.get("exclude", []))
    print(f"\nLoading recordings (holdout: {holdout}, exclude: {exclude}):")
    train_ds, val_ds, train_datasets_list, _ = load_multi_recording_datasets(cfg, holdout, exclude_names=exclude)
    print(f"\nTrain: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    # Check if VP data available
    ref_ds = train_datasets_list[0]
    has_lane = "lane_masks" in cfg.data.dataset.keys_to_load
    has_depth = "depth_maps" in cfg.data.dataset.keys_to_load
    print(f"VP data: lane_masks={has_lane}, depth_maps={has_depth}")

    # Fit normalizers on train data only
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col in ("lane_masks", "depth_maps", "depth_conf"):
                continue

            all_col_data = []
            for ds in train_datasets_list:
                col_data = ds.get_col_data(col)
                all_col_data.append(torch.from_numpy(__import__('numpy').array(col_data)))

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
    for ds in train_ds.datasets:
        ds.transform = transform
    for ds in val_ds.datasets:
        ds.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(
        train_ds, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_ds, **cfg.loader, shuffle=False, drop_last=False
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

    # VP decoders
    lane_decoder = LaneDecoder(embed_dim) if has_lane else None
    depth_decoder = DepthDecoder(embed_dim) if has_depth else None

    # Attach decoders as sub-modules of world_model so they share the optimizer
    if lane_decoder:
        world_model.lane_decoder = lane_decoder
    if depth_decoder:
        world_model.depth_decoder = depth_decoder

    extra_modules = {}

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_vp_forward, cfg=cfg),
        optim=optimizers,
        **extra_modules,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
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
