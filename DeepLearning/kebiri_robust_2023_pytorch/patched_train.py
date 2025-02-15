import os
from abc import ABC
import numpy as np

from warnings import filterwarnings

filterwarnings("ignore", category=FutureWarning)

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.tuner.tuning import Tuner
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import monai
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser

from patched_data_module import DWIDataPatched
from patched_model import FODNetPatched


def main(args):
    seed_everything(args.seed)

    data = DWIDataPatched(
        data_dir=args.data_dir,
        subject_list_dir=args.subject_list_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_val_split=args.train_val_split,
        cache_rate=args.cache_rate,
        mul_factor=args.mul_factor,
        x_name=args.x_name,
        y_name=args.y_name,
        mask_name=args.mask_name,
        wm_mask_name=args.wm_mask_name,
    )

    if args.ckpt_finetune_dir is not None:
        model = FODNetPatched.load_from_checkpoint(
            args.ckpt_finetune_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            backbone=args.backbone,
            critrion=args.criterion,
            optimizer=args.optimizer,
            activation=args.activation,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            spatial_dims=args.spatial_dims,
            mul_factor=args.mul_factor,
        )
    else:
        model = FODNetPatched(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            backbone=args.backbone,
            critrion=args.criterion,
            optimizer=args.optimizer,
            activation=args.activation,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            spatial_dims=args.spatial_dims,
            mul_factor=args.mul_factor,
            # ckpt_finetune_dir=args.ckpt_finetune_dir,
        )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.wandb_name}-{args.backbone}-{args.learning_rate}-{args.mul_factor}",
        log_model=False,
        offline=False,
    )
    wandb_logger.log_hyperparams(args)
    wandb.run.log_code(
        include_fn=lambda path: path.endswith(".py") or path.endswith(".sh")
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=f"{args.wandb_project}-{args.wandb_name}"
        + "-{epoch:04d}-{val_loss:.8f}-{val_loss_wm:.8f}",
        save_top_k=5,
        mode="min",
        save_weights_only=True,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=False, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=[3],
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        # precision=args.precision,
    )

    # tuner = Tuner(trainer)
    # # # tuner.scale_batch_size(model, data)
    # #
    # lr = tuner.lr_find(model, data)
    # print(lr.suggestion())

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_dir", type=str, default="/mnt/lrz/data/BCP_train_patched"
    )
    parser.add_argument(
        "--subject_list_dir",
        type=str,
        default="/mnt/lrz/data/BCP_sampled/BCP_sampled_10.txt",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--crop_size", type=int, default=16)
    parser.add_argument("--cache_rate", type=float, default=0.0)
    parser.add_argument("--mul_factor", type=float, default=10)
    parser.add_argument("--x_name", type=str, default="dwi_6_1000_orig")
    parser.add_argument("--y_name", type=str, default="wm")
    parser.add_argument("--mask_name", type=str, default="mask_bet")
    parser.add_argument("--wm_mask_name", type=str, default="HD_WM")

    parser.add_argument(
        "--ckpt_finetune_dir",
        type=str,
        default=None,
    )

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--backbone", type=str, default="davoodnet")
    parser.add_argument("--criterion", type=str, default="mse")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--activation", type=str, default="PReLU")
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--out_channels", type=int, default=45)
    parser.add_argument("--spatial_dims", type=int, default=3)

    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--wandb_project", type=str, default="FODNetPatched")
    parser.add_argument("--wandb_name", type=str, default="SS3T-6")
    parser.add_argument("--offline", type=bool, default=False)
    args = parser.parse_args()

    main(args)
