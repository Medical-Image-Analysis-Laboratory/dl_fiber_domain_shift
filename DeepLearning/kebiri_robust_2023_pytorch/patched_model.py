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


class FODNetPatched(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_workers=0,
        batch_size=1,
        crop_size=16,
        backbone="davoodnet",
        critrion="mse",
        optimizer="adam",
        activation="PReLU",
        in_channels=6,
        out_channels=45,
        spatial_dims=3,
        mul_factor=1.0,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.mul_factor = mul_factor
        self.activation = activation

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

    @staticmethod
    def prepare_batch(batch):
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
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, wm_mask, _ = self.prepare_batch(batch)

        y_hat = self(x)
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
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )
        self.log(
            "val_loss_wm",
            loss,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )
        self.log(
            "val_metric",
            metric,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )

        return loss_unmasked

    def test_step(self, batch, batch_idx):
        x, y, mask, wm_mask, x_meta_dict = self.prepare_batch(batch)

        test_data_dir = "/media/rizhong/Data/dHCP_test_data"

        affine = x_meta_dict["original_affine"][0]
        filename = x_meta_dict["filename_or_obj"][0]

        subject_name = os.path.basename(os.path.dirname(filename))
        # get the part before the _
        subject_name = subject_name.split("_")[0]
        subject_dir = os.path.join(test_data_dir, subject_name)

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
        output_dir = os.path.join(subject_dir, "dl_ss3t_U_45.nii.gz")
        y_hat_nii = nib.Nifti1Image(y_hat, affine.cpu().numpy())
        nib.save(y_hat_nii, output_dir)

        self.log("test_loss", loss, prog_bar=True, batch_size=self.val_batch_size)
        return loss


if __name__ == "__main__":

    pass
    # model = FODNetPatched(backbone='dynunet', crop_size=16)
    #
    # model.load_from_checkpoint

    x = torch.randn(256, 6, 16, 16, 16).to("cuda")
    # y = model(x)
    # print(y.shape)
    # # check if y is normalized
    # print(y.mean(), y.std())

    model = FODNetPatched.load_from_checkpoint(
        "/home/lrz/code/UNetPyTorch/checkpoints/FODNet-MSMT-6-epoch=0624-val_loss=0.01414023-val_loss_wm=0.01075079.ckpt"
    )
    y = model(x)
    print(y.shape)
    # check if y is normalized
    print(y.mean(), y.std())

    # get the current epoch and other info from the checkpoint
    print(model.current_epoch)
