"""Multi-recording training with whole-recording holdout split.

Usage:
    # Train on 4 recordings, validate on 1 (holdout)
    python train_multi.py data=rtb holdout=Namyang-Gian_JT_2025-09-15_07-06-35_2111_6c319d

    # Override epochs, batch size, etc.
    python train_multi.py data=rtb holdout=Namyang-Gian_JT_2025-09-15_07-06-35_2111_6c319d trainer.max_epochs=10
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def load_multi_recording_datasets(cfg, holdout_name: str, exclude_names: list[str] | None = None):
    """Load multiple recordings, split by whole-recording holdout.

    Args:
        exclude_names: list of substrings to match against recording names to skip entirely.
    """
    # Extract subdir from dataset name (e.g. "rtb4d/RECORDING_ID" -> "rtb4d")
    ds_name = cfg.data.dataset.get("name", "rtb/RECORDING_ID")
    subdir = ds_name.split("/")[0] if "/" in ds_name else "rtb"
    rtb_dir = Path(swm.data.utils.get_cache_dir()) / subdir
    h5_files = sorted(rtb_dir.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files in {rtb_dir}")

    exclude_names = exclude_names or []

    train_datasets = []
    val_datasets = []

    dataset_cfg = dict(cfg.data.dataset)
    dataset_cfg.pop("name", None)

    for h5_path in h5_files:
        rec_name = h5_path.stem

        if any(ex in rec_name for ex in exclude_names):
            print(f"  SKIP:  {rec_name}")
            continue

        ds = swm.data.HDF5Dataset(name=f"{subdir}/{rec_name}", **dataset_cfg, transform=None)

        if holdout_name and holdout_name in rec_name:
            val_datasets.append(ds)
            print(f"  VAL:   {rec_name} ({len(ds)} sequences)")
        else:
            train_datasets.append(ds)
            print(f"  TRAIN: {rec_name} ({len(ds)} sequences)")

    if not val_datasets:
        raise ValueError(f"Holdout '{holdout_name}' not found in {[f.stem for f in h5_files]}")

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)

    return train_ds, val_ds, train_datasets, val_datasets


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    holdout = cfg.get("holdout", "")
    if not holdout:
        raise ValueError("Must specify holdout=RECORDING_NAME")

    with open_dict(cfg):
        if not hasattr(cfg, "data"):
            cfg.data = {"dataset": {}}

    #########################
    ##       dataset       ##
    #########################

    print(f"\nLoading recordings (holdout: {holdout}):")
    train_ds, val_ds, train_datasets_list, _ = load_multi_recording_datasets(cfg, holdout)
    print(f"\nTrain: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    # Fit normalizers on train data only
    # Use first train dataset to get column data and dimensions
    ref_ds = train_datasets_list[0]
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            # Collect column data from ALL train datasets
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

    # Apply transform to all underlying datasets
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
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

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
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
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
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
