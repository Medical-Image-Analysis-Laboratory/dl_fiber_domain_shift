import os
from abc import ABC
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F
import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged,
    ToTensord,
    Compose,
    CropForegroundd,
    Lambdad,
)
from sklearn.model_selection import train_test_split
from glob import glob


class DWIDataPatched(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir=None,
        subject_list_dir=None,
        batch_size=1,
        num_workers=0,
        train_val_split=0.8,
        cache_rate=0.0,
        mul_factor=1.0,
        x_name="dwi_6_1000_orig",
        y_name="wm",
        mask_name="mask",
        wm_mask_name="HD_WM",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.subject_list_dir = subject_list_dir
        self.subject_list = np.loadtxt(self.subject_list_dir, dtype=str)
        if self.subject_list.size == 1:
            self.subject_list = np.array([self.subject_list])
        if __name__ == "__main__":
            print("subject_list: ", self.subject_list)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.cache_rate = cache_rate
        self.mul_factor = mul_factor
        self.x_name = x_name
        self.y_name = y_name
        self.mask_name = mask_name
        self.wm_mask_name = wm_mask_name

        self.subject_dict_list = self.get_subject_dict_list(
            self.data_dir,
            self.subject_list,
            self.x_name,
            self.y_name,
            self.mask_name,
            self.wm_mask_name,
        )
        print("Total number of patches: ", len(self.subject_dict_list))

        self.train_subject_dict_list, self.val_subject_dict_list = train_test_split(
            self.subject_dict_list, train_size=self.train_val_split, random_state=42
        )

        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def get_subject_dict_list(
        data_dir, subject_list, x_name, y_name, mask_name, wm_mask_name
    ):
        def get_sorted_files(sub, pattern):
            """
            Helper function to get sorted list of files based on a pattern.
            """
            return sorted(glob(os.path.join(data_dir, sub, pattern)))

        subject_dict_list = []
        for s in subject_list:
            x = get_sorted_files(s, f"{x_name}_patch_*.nii.gz")
            y = get_sorted_files(s, f"{y_name}_patch_*.nii.gz")
            mask = get_sorted_files(s, f"{mask_name}_patch_*.nii.gz")
            wm_mask = get_sorted_files(s, f"{wm_mask_name}_patch_*.nii.gz")

            try:
                assert len(x) == len(y) == len(mask) == len(wm_mask)
            except AssertionError as e:
                print(
                    f"Subject {s}: {e} - x: {len(x)}, y: {len(y)}, mask: {len(mask)}, wm_mask: {len(wm_mask)}"
                )
                continue

            for i in range(len(x)):
                subject_dict_list.append(
                    {"x": x[i], "y": y[i], "mask": mask[i], "wm_mask": wm_mask[i]}
                )

        if __name__ == "__main__":
            print("Total number of patches: ", len(subject_dict_list))
        return subject_dict_list

    def get_transforms(self, stage=None):
        if stage == "fit" or stage is None:
            return Compose(
                [
                    LoadImaged(
                        keys=["x", "y"], image_only=False, ensure_channel_first=True
                    ),
                    LoadImaged(
                        keys=["mask", "wm_mask"],
                        image_only=False,
                        dtype=bool,
                        ensure_channel_first=True,
                    ),
                    Lambdad(
                        keys=["y"],
                        func=lambda x: x * self.mul_factor,
                        inv_func=lambda x: x / self.mul_factor,
                    ),
                    ToTensord(keys=["x", "y", "mask", "wm_mask"]),
                ]
            )
        elif stage == "predict":
            return Compose(
                [
                    LoadImaged(keys=["x"], image_only=False, ensure_channel_first=True),
                    ToTensord(keys=["x"]),
                ]
            )
        else:
            raise NotImplementedError

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CacheDataset(
                self.train_subject_dict_list,
                transform=self.get_transforms(stage),
                cache_rate=self.cache_rate,
            )
            self.val_dataset = CacheDataset(
                self.val_subject_dict_list,
                transform=self.get_transforms(stage),
                cache_rate=self.cache_rate,
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    seed_everything(42)

    data_dir = "/mnt/lrz/data/dHCP_train_patched"
    subject_list_dir = "train_val_list.txt"
    batch_size = 64
    num_workers = 0
    train_val_split = 0.8
    cache_rate = 0.0
    mul_factor = 10
    x_name = "dwi_6_1000_orig"
    y_name = "wm"
    mask_name = "mask"
    wm_mask_name = "HD_WM"

    dm = DWIDataPatched(
        data_dir=data_dir,
        subject_list_dir=subject_list_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_split=train_val_split,
        cache_rate=cache_rate,
        mul_factor=mul_factor,
        x_name=x_name,
        y_name=y_name,
        mask_name=mask_name,
        wm_mask_name=wm_mask_name,
    )

    dm.setup("fit")

    for batch in dm.train_dataloader():
        print(
            batch["x"].shape,
            batch["y"].shape,
            batch["mask"].shape,
            batch["wm_mask"].shape,
        )
        break

    for batch in dm.val_dataloader():
        print(
            batch["x"].shape,
            batch["y"].shape,
            batch["mask"].shape,
            batch["wm_mask"].shape,
        )
        break
