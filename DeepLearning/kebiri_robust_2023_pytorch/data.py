import os
from abc import ABC
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F
import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    RandRicianNoised,
    ScaleIntensityRanged,
    Lambdad,
    CopyItemsd,
    NormalizeIntensityd,
)
from sklearn.model_selection import train_test_split


class DWIData(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir=None,
        subject_list_dir=None,
        test_data_dir=None,
        test_list_dir=None,
        batch_size=1,
        val_batch_size=1,
        num_patches=1,
        num_workers=0,
        train_val_split=70,
        crop_size=16,
        cache_rate=0,
        mul_factor=1,
        noise_std=0.0,
        x_name="dwi_6_1000_orig.nii.gz",
        y_name="wm.nii.gz",
        mask_name="mask.nii.gz",
        wm_mask_name="HD_WM.nii.gz",
        b0_name="dwi_6_1000.nii.gz",
        wm_response_name="wm_response_6_1000.txt",
        bvals_name="dwi_6_1000.bval",
        bvecs_name="dwi_6_1000.bvec",
        is_shconv=False,
        x_sh_order=8,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.subject_list_dir = subject_list_dir
        self.subject_list = np.loadtxt(self.subject_list_dir, dtype=str)
        if __name__ == "__main__":
            print("subject_list: ", self.subject_list)
        self.test_data_dir = test_data_dir if test_data_dir is not None else data_dir
        self.test_list_dir = test_list_dir if test_list_dir is not None else None
        self.test_list = (
            np.loadtxt(self.test_list_dir, dtype=str)
            if test_list_dir is not None
            else None
        )
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_patches = num_patches
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.crop_size = crop_size
        self.cr = cache_rate
        self.mul_factor = mul_factor
        self.noise_std = noise_std
        self.x_name = x_name
        self.y_name = y_name
        self.mask_name = mask_name
        self.wm_mask_name = wm_mask_name
        self.b0_name = b0_name
        self.bvals_name = bvals_name
        self.bvecs_name = bvecs_name
        self.wm_response_name = wm_response_name
        self.is_shconv = is_shconv
        self.x_sh_order = x_sh_order

        self.subject_dict_list = self.get_subject_dict_list(
            self.data_dir,
            self.subject_list,
            self.x_name,
            self.y_name,
            self.mask_name,
            self.wm_mask_name,
            self.b0_name,
            self.bvals_name,
            self.bvecs_name,
            self.wm_response_name,
        )
        self.test_subject_dict_list = (
            self.get_subject_dict_list(
                self.test_data_dir,
                self.test_list,
                self.x_name,
                self.y_name,
                self.mask_name,
                self.wm_mask_name,
                self.b0_name,
                self.bvals_name,
                self.bvecs_name,
                self.wm_response_name,
            )
            if self.test_list is not None
            else None
        )

        self.train_subject_dict_list, self.val_subject_dict_list = train_test_split(
            self.subject_dict_list, train_size=self.train_val_split
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def get_subject_dict_list(
        data_dir,
        subject_list,
        x_name,
        y_name,
        mask_name,
        wm_mask_name,
        b0_name,
        bvals_name,
        bvecs_name,
        wm_response_name,
    ):
        return [
            {
                "x": os.path.join(data_dir, s, x_name),
                "y": os.path.join(data_dir, s, y_name),
                "mask": os.path.join(data_dir, s, mask_name),
                "wm_mask": os.path.join(data_dir, s, wm_mask_name),
                "b0": os.path.join(data_dir, s, b0_name),
                "bvals": os.path.join(data_dir, s, bvals_name),
                "bvecs": os.path.join(data_dir, s, bvecs_name),
                "wm_response": os.path.join(data_dir, s, wm_response_name),
            }
            for s in subject_list
        ]

    def get_transforms(self, aug=False, crop=True):
        return Compose(
            [
                LoadImaged(
                    keys=["x", "y"], image_only=False, ensure_channel_first=True
                ),
                LoadImaged(keys=["b0"], image_only=False, ensure_channel_first=True),
                # this is a little bit ugly above
                LoadImaged(
                    keys=["mask", "wm_mask"],
                    image_only=False,
                    dtype=bool,
                    ensure_channel_first=True,
                ),
                ReadBvalsBvecsd(bvals_key="bvals", bvecs_key="bvecs"),
                (
                    LoadResponseFunctiond(keys=["wm_response"])
                    if self.is_shconv
                    else Compose([])
                ),
                (
                    CropForegroundd(
                        keys=["x", "y", "b0", "mask", "wm_mask"],
                        source_key="mask",
                        allow_smaller=False,
                        margin=4,
                    )
                    if crop
                    else Compose([])
                ),
                Lambdad(
                    keys=["y"],
                    func=lambda x: x * self.mul_factor,
                    inv_func=lambda x: x / self.mul_factor,
                ),
                (
                    CopyItemsd(keys=["x"], times=1, names=["x_orig"])
                    if self.is_shconv
                    else Compose([])
                ),
                (
                    NormalizeByB0d(keys=["x"], b0_key="b0")
                    if not self.is_shconv
                    else Compose([])
                ),
                # RandRicianNoised(keys=["x"], prob=0.4, std=self.noise_std) if aug else Compose([]),
                # NormalizeIntensityd(keys=["x"]) if self.is_shconv else Compose([]),
                # ScaleIntensityRanged(keys=["x"], a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
                SphericalFunctionToSphericalHarmonicsMRtrixd(
                    keys=["x"],
                    bvals_key="bvals",
                    bvecs_key="bvecs",
                    sh_order=self.x_sh_order,
                ),
                # SphericalFunctionToSphericalHarmonicsd(keys=["x_orig"], bvals_key="bvals", bvecs_key="bvecs",
                #                                        sh_order=2, ) if self.is_shconv else Compose([]),
                ToTensord(
                    keys=(
                        ["x", "y", "b0", "mask", "wm_mask", "x_orig"]
                        if self.is_shconv
                        else ["x", "y", "b0", "mask", "wm_mask"]
                    )
                ),
                (
                    RandCropByPosNegLabeld(
                        keys=(
                            ["x", "y", "b0", "mask", "wm_mask", "x_orig"]
                            if self.is_shconv
                            else ["x", "y", "b0", "mask", "wm_mask"]
                        ),
                        label_key="mask",
                        spatial_size=[self.crop_size] * 3,
                        num_samples=self.num_patches,
                        neg=0,
                    )
                    if aug
                    else Compose([])
                ),
            ]
        )

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CacheDataset(
                self.train_subject_dict_list,
                transform=self.get_transforms(aug=True),
                cache_rate=self.cr,
            )
            self.val_dataset = CacheDataset(
                self.val_subject_dict_list,
                transform=self.get_transforms(aug=False),
                cache_rate=self.cr,
            )
        if (stage == "test" or stage is None) and self.test_list is not None:
            self.test_dataset = CacheDataset(
                self.test_subject_dict_list,
                transform=self.get_transforms(aug=False, crop=False),
                cache_rate=0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    seed_everything(42)
    data_dir = "/media/rizhong/Data/dHCP_train"
    subject_list_dir = "/media/rizhong/Data/dHCP_train/dhcp_sampled_new.txt"
    test_data_dir = None
    test_list_dir = None
    batch_size = 1
    num_patches = 1
    num_workers = 1
    train_val_split = 3
    crop_size = 16
    cache_rate = 0
    x_name = "dwi_6_1000_orig.nii.gz"
    y_name = "wm.nii.gz"
    b0_name = "b0.nii.gz"
    mask_name = "mask.nii.gz"
    wm_mask_name = "HD_WM.nii.gz"
    bvals_name = "dwi_6_1000.bval"
    bvecs_name = "dwi_6_1000.bvec"
    wm_response_name = "wm_response_csd_6_1000.txt"
    noise_std = 1

    data = DWIData(
        data_dir=data_dir,
        subject_list_dir=subject_list_dir,
        test_data_dir=test_data_dir,
        test_list_dir=test_list_dir,
        batch_size=batch_size,
        num_patches=num_patches,
        num_workers=num_workers,
        train_val_split=train_val_split,
        crop_size=crop_size,
        cache_rate=cache_rate,
        x_name=x_name,
        y_name=y_name,
        b0_name=b0_name,
        mask_name=mask_name,
        wm_mask_name=wm_mask_name,
        bvals_name=bvals_name,
        bvecs_name=bvecs_name,
        wm_response_name=wm_response_name,
        noise_std=noise_std,
        is_shconv=False,
    )
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    for i, batch in enumerate(train_loader):
        print(
            "train_loader: ",
            i,
            batch["x"].shape,
            batch["y"].shape,
            batch["mask"].shape,
            batch["b0"].shape,
        )
        print("train_loader: ", i, batch["x"].max(), batch["x"].min())
        print("train_loader: ", i, batch["y"].max(), batch["y"].min())
        print("train_loader: ", i, batch["b0"].max(), batch["b0"].min())
        print("train_loader: ", i, batch["mask"].max(), batch["mask"].min())
        print("train_loader: ", i, batch["wm_mask"].max(), batch["wm_mask"].min())
        # print("train_loader: ", i, batch["wm_response"].shape)

        print("train_loader: ", i, batch["x_meta_dict"].keys())
        print("train_loader: ", i, batch["x_meta_dict"]["filename_or_obj"])

    # for i, batch in enumerate(val_loader):
    #     print("val_loader: ", i, batch["x"].shape, batch["y"].shape, batch["mask"].shape, batch["b0"].shape)
    #     print("val_loader: ", i, batch["x"].max(), batch["x"].min())
    #     print("val_loader: ", i, batch["y"].max(), batch["y"].min())
    #     print("val_loader: ", i, batch["b0"].max(), batch["b0"].min())
    #     print("val_loader: ", i, batch["mask"].max(), batch["mask"].min())
    #     print("val_loader: ", i, batch["wm_mask"].max(), batch["wm_mask"].min())
    #     print("val_loader: ", i, batch["wm_response"].shape)
    #
    #     print("val_loader: ", i, batch["x_meta_dict"].keys())
    #     print("val_loader: ", i, batch["x_meta_dict"]['filename_or_obj'])
    #     print("val_loader: ", i, batch["x_meta_dict"]['affine'].shape)
    #     print("val_loader: ", i, batch["x_meta_dict"]['affine'])
    #     print("val_loader: ", i, batch["x_meta_dict"]['original_affine'])
    #
    #     from monai.transforms import SaveImage
    #     from monai.data.utils import decollate_batch
    #
    #     save_image = SaveImage(output_dir="./test/blah")
    #
    #     x_to_save = {
    #         "image": batch["y"],
    #         "meta": batch["y_meta_dict"]
    #     }
    #
    #     x_to_save = decollate_batch(x_to_save)
    #
    #     for i, d in enumerate(x_to_save):
    #         x=d["image"]
    #         meta=d["meta"]
    #         print(meta['filename_or_obj'])
    #         save_image(x, meta, f'123_{i}')
    #
    #     break
