import os
from abc import ABC
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet, DynUNet
from monai.inferers import sliding_window_inference
from nets import DavoodNet
from monai.networks import icnr_init, normal_init
from monai.losses import MaskedLoss, SSIMLoss
from monai.metrics import SSIMMetric, MAEMetric
import nibabel as nib


class FODNet(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_workers=0,
        batch_size=1,
        val_batch_size=1,
        num_patches=1,
        crop_size=16,
        backbone="unet",
        critrion="mse",
        optimizer="adam",
        activation="PReLU",
        in_channels=6,
        out_channels=45,
        spatial_dims=3,
        mul_factor=1.0,
        test_data_save_dir=None,
        test_data_save_name=None,
        test_overwrite=False,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_patches = num_patches
        self.crop_size = crop_size
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.mul_factor = mul_factor
        self.activation = activation

        self.test_data_save_dir = test_data_save_dir
        self.test_data_save_name = test_data_save_name
        self.test_overwrite = test_overwrite

        if critrion == "mse":
            self.critrion = F.mse_loss
        elif critrion == "l1":
            self.critrion = F.l1_loss
        else:
            raise ValueError("Unknown loss function.")

        # self.critrion = MaskedLoss(self.critrion)

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            raise ValueError("Unknown optimizer.")

        self.metric = F.l1_loss

        if backbone.lower() == "unet":
            # the patch size is only 16, make sure the network is not too deep
            self.model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(32, 64, 128, 256),
                strides=(
                    2,
                    2,
                    2,
                ),
                num_res_units=2,
            )
        elif backbone.lower() == "dynunet":

            def get_kernels_strides():
                """
                This function is only used for decathlon datasets with the provided patch sizes.
                When referring this method for other tasks, please ensure that the patch size for each spatial dimension
                should be divisible by the product of all strides in the corresponding dimension.
                In addition, the minimal spatial size should have at least one dimension that has twice the size of
                the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
                """
                sizes, spacings = [self.crop_size] * self.spatial_dims, [
                    1.0
                ] * self.spatial_dims
                input_size = sizes
                strides, kernels = [], []
                while True:
                    spacing_ratio = [sp / min(spacings) for sp in spacings]
                    stride = [
                        2 if ratio <= 2 and size >= 8 else 1
                        for (ratio, size) in zip(spacing_ratio, sizes)
                    ]
                    kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
                    if all(s == 1 for s in stride):
                        break
                    for idx, (i, j) in enumerate(zip(sizes, stride)):
                        if i % j != 0:
                            raise ValueError(
                                f"Patch size is not supported, please try to modify the size {input_size[idx]} in the "
                                f"spatial dimension {idx}."
                            )
                    sizes = [i / j for i, j in zip(sizes, stride)]
                    spacings = [i * j for i, j in zip(spacings, stride)]
                    kernels.append(kernel)
                    strides.append(stride)

                strides.insert(0, len(spacings) * [1])
                kernels.append(len(spacings) * [3])
                return kernels, strides

            kernels, strides = get_kernels_strides()
            self.model = DynUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],  # skip the input layer
                res_block=True,
            )
        elif backbone.lower() == "davoodnet":
            self.model = DavoodNet(
                spatial_dims=spatial_dims,
                kernel_size=3,
                depth=np.log2(crop_size).astype(int) - 1,
                n_feat_0=36,
                num_channel=in_channels,
                num_class=out_channels,
                dropout=0.1,
                act=self.activation,
                norm=None,
            )
        else:
            raise ValueError("Unknown backbone.")

        self.model.apply(normal_init)

    def configure_optimizers(self):
        return self.optimizer(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def forward(self, x):
        return self.model(x)

    def prepare_batch(self, batch):
        return (
            batch["x"].float(),
            batch["y"].float(),
            batch["mask"],
            batch["wm_mask"],
            batch["x_meta_dict"],
        )

    def training_step(self, batch, batch_idx):
        x, y, mask, wm_mask, _ = self.prepare_batch(batch)
        y_hat = self(x)
        loss = self.critrion(y_hat[mask.expand_as(y_hat)], y[mask.expand_as(y_hat)])
        # loss = self.critrion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, wm_mask, _ = self.prepare_batch(batch)

        y_hat = sliding_window_inference(
            x,
            roi_size=[self.crop_size] * self.spatial_dims,
            sw_batch_size=self.batch_size * self.num_patches,
            predictor=self.model,
            overlap=0.25,
        )
        loss_unmasked = self.critrion(
            y_hat[mask.expand_as(y_hat)], y[mask.expand_as(y_hat)]
        )
        loss = self.critrion(
            y_hat[wm_mask.expand_as(y_hat)], y[wm_mask.expand_as(y_hat)]
        )
        metric = self.metric(
            y_hat[wm_mask.expand_as(y_hat)], y[wm_mask.expand_as(y_hat)]
        )
        # loss = self.critrion(y_hat, y)
        self.log(
            "val_loss",
            loss_unmasked,
            prog_bar=True,
            batch_size=self.val_batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_wm",
            loss,
            prog_bar=True,
            batch_size=self.val_batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_metric",
            metric,
            prog_bar=True,
            batch_size=self.val_batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        # log images
        if batch_idx == 0:
            wandb.log(
                {
                    f"val_x_{batch_idx}": wandb.Image(
                        x[0, 0, :, :, 30].cpu(),
                        caption=f"min: {x[wm_mask.expand_as(x)].min()}, "
                        f"max: {x[wm_mask.expand_as(x)].max()}",
                    ),
                    f"val_y_{batch_idx}": wandb.Image(
                        y[0, 0, :, :, 30].cpu(),
                        caption=f"min: {y[wm_mask.expand_as(y)].min()}, "
                        f"max: {y[wm_mask.expand_as(y)].max()}",
                    ),
                    f"val_y_hat_{batch_idx}": wandb.Image(
                        y_hat[0, 0, :, :, 30].cpu(),
                        caption=f"min: {y_hat[wm_mask.expand_as(y_hat)].min()}, "
                        f"max: {y_hat[wm_mask.expand_as(y_hat)].max()}",
                    ),
                }
            )
        return loss_unmasked

    def test_step(self, batch, batch_idx):
        x, y, mask, wm_mask, x_meta_dict = self.prepare_batch(batch)

        test_data_dir = self.test_data_save_dir
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

        affine = x_meta_dict["original_affine"][0]
        filename = x_meta_dict["filename_or_obj"][0]

        subject_name = os.path.basename(os.path.dirname(filename))
        # # get the part before the _
        # subject_name = subject_name.split('_')[0]
        subject_dir = os.path.join(test_data_dir, subject_name)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        # check if the file already exists, if so, and overwrite is False, skip the test
        if not self.test_overwrite and os.path.exists(
            os.path.join(subject_dir, self.test_data_save_name)
        ):
            return None

        y_hat = sliding_window_inference(
            x,
            roi_size=[self.crop_size] * self.spatial_dims,
            sw_batch_size=self.val_batch_size,
            predictor=self.model,
            overlap=0.25,
        )
        loss = self.critrion(
            y_hat[wm_mask.expand_as(y_hat)], y[wm_mask.expand_as(y_hat)]
        )
        y_hat /= self.mul_factor

        # multiply the y_hat with the mask
        y_hat = y_hat * mask.expand_as(y_hat)

        y_hat = y_hat.cpu().numpy()[0]
        y_hat = np.moveaxis(y_hat, 0, -1)

        # save the output
        # output_dir = os.path.join(subject_dir, f'dl_early_ss3t_U_{self.in_channels}.nii.gz')
        output_dir = os.path.join(subject_dir, self.test_data_save_name)
        y_hat_nii = nib.Nifti1Image(y_hat, affine.cpu().numpy())
        nib.save(y_hat_nii, output_dir)

        self.log("test_loss", loss, prog_bar=True, batch_size=self.val_batch_size)
        return loss


if __name__ == "__main__":
    model = FODNet(backbone="dynunet", crop_size=32)

    x = torch.randn(1, 6, 100, 100, 64)
    y = sliding_window_inference(
        x,
        roi_size=[32, 32, 32],
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
    )
    print(y.shape)
