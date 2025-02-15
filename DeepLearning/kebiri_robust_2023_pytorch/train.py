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

from data import DWIData
from model import FODNet


def main(args):
    seed_everything(args.seed)

    data = DWIData(
        data_dir=args.data_dir,
        subject_list_dir=args.subject_list_dir,
        test_data_dir=args.test_data_dir,
        test_list_dir=args.test_list_dir,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_patches=args.num_patches,
        num_workers=args.num_workers,
        train_val_split=args.train_val_split,
        crop_size=args.crop_size,
        cache_rate=args.cache_rate,
        mul_factor=args.mul_factor,
        noise_std=args.noise_std,
        x_name=args.x_name,
        y_name=args.y_name,
        mask_name=args.mask_name,
        b0_name=args.b0_name,
        bvals_name=args.bvals_name,
        bvecs_name=args.bvecs_name,
        x_sh_order=args.x_sh_order,
    )

    model = FODNet(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_patches=args.num_patches,
        val_batch_size=args.val_batch_size,
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

    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_name}-{args.backbone}-{args.learning_rate}-{args.mul_factor}",
        group=f"{args.wandb_name}-{args.backbone}-{args.learning_rate}-{args.mul_factor}",
        save_code=True,
        mode="online",
    )
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.wandb_name}-{args.backbone}-{args.learning_rate}-{args.mul_factor}",
        group=f"{args.wandb_name}-{args.backbone}-{args.learning_rate}-{args.mul_factor}",
        log_model=False,  # "all" if not args.offline else False,
        offline=args.offline,
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
        save_weights_only=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, verbose=False, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        # strategy='ddp',
        max_epochs=args.max_epochs,
        min_epochs=500,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        precision=args.precision,
        # check_val_every_n_epoch=10,
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
        "--data_dir", type=str, default="/media/rizhong/Data/dHCP_train"
    )
    parser.add_argument(
        "--subject_list_dir",
        type=str,
        default="/media/rizhong/Data/dHCP_train/dhcp_sampled_new.txt",
    )
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--test_list_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_patches", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_val_split", type=int, default=70)
    parser.add_argument("--crop_size", type=int, default=16)
    parser.add_argument("--cache_rate", type=float, default=0)
    parser.add_argument("--mul_factor", type=float, default=10)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--x_name", type=str, default="dwi_45_1000_orig.nii.gz")
    parser.add_argument("--y_name", type=str, default="wm_ss3t_20_0_88_1000.nii.gz")
    parser.add_argument("--mask_name", type=str, default="mask.nii.gz")
    parser.add_argument("--b0_name", type=str, default="b0.nii.gz")
    parser.add_argument("--bvals_name", type=str, default="dwi_45_1000.bval")
    parser.add_argument("--bvecs_name", type=str, default="dwi_45_1000.bvec")

    parser.add_argument("--learning_rate", type=float, default=10e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--backbone", type=str, default="davoodnet")
    parser.add_argument("--criterion", type=str, default="mse")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--activation", type=str, default="PReLU")
    parser.add_argument("--in_channels", type=int, default=45)
    parser.add_argument("--x_sh_order", type=int, default=8)
    parser.add_argument("--out_channels", type=int, default=45)
    parser.add_argument("--spatial_dims", type=int, default=3)

    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--wandb_project", type=str, default="fodnet")
    parser.add_argument("--wandb_name", type=str, default="SS3T-28")
    parser.add_argument("--offline", type=bool, default=False)
    args = parser.parse_args()

    main(args)
