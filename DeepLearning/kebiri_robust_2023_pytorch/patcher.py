import os
from abc import ABC
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from pytorch_lightning import seed_everything
import torch
import monai
from monai.inferers import SlidingWindowSplitter
from monai.transforms import (
    LoadImaged,
    SaveImage,
    SaveImaged,
    Compose,
    CropForegroundd,
    RandRicianNoised,
    Lambdad,
    GridPatchd,
    ToTensord,
)
from utils import *

num_measurement_list = [12, 6]
sh_order_list = [2, 2]

# num_measurement = 45
# sh_order = 8

for num_measurement, sh_order in zip(num_measurement_list, sh_order_list):

    dataset_root_dir = os.path.join("/mnt/lrz/data/dHCP_train")
    dataset_patched_dir = os.path.join("/mnt/lrz/data/dHCP_train_patched")

    # dataset_root_dir = os.path.join("/mnt/lrz/data/BCP_sampled")
    # dataset_patched_dir = os.path.join("/mnt/lrz/data/BCP_train_patched")
    x_name = f"dwi_{num_measurement}_1000_orig.nii.gz"
    y_name = "wm.nii.gz"  # "wm.nii.gz"
    mask_name = "mask.nii.gz"
    wm_mask_name = "HD_WM.nii.gz"
    bvals_name = f"dwi_{num_measurement}_1000.bval"
    bvecs_name = f"dwi_{num_measurement}_1000.bvec"
    b0_name = "b0.nii.gz"

    # folder_list = sorted(glob(os.path.join(dataset_root_dir, "CC*")))
    # print(folder_list)

    # subject_list = np.loadtxt("/mnt/lrz/data/dHCP_train/dhcp_sampled_10.txt", dtype=str)
    # subject_list_5 = np.loadtxt("/mnt/lrz/data/BCP_sampled/BCP_sampled_5.txt", dtype=str)
    # subject_list_2 = np.loadtxt("/mnt/lrz/data/BCP_sampled/BCP_sampled_2.txt", dtype=str)
    # subject_list_1 = np.array([np.loadtxt("/mnt/lrz/data/BCP_sampled/BCP_sampled_1.txt", dtype=str)])

    subject_list_5 = np.loadtxt(
        "/mnt/lrz/data/dHCP_train/dhcp_sampled_5.txt", dtype=str
    )
    subject_list_2 = np.loadtxt(
        "/mnt/lrz/data/dHCP_train/dhcp_sampled_2.txt", dtype=str
    )
    subject_list_1 = np.array(
        [np.loadtxt("/mnt/lrz/data/dHCP_train/dhcp_sampled_1.txt", dtype=str)]
    )

    subject_list = [
        *subject_list_5,
        *subject_list_2,
        *subject_list_1,
    ]
    folder_list = [os.path.join(dataset_root_dir, subject) for subject in subject_list]

    subject_dict_list = [
        {
            "x": os.path.join(folder, x_name),
            "y": os.path.join(folder, y_name),
            "mask": os.path.join(folder, mask_name),
            "wm_mask": os.path.join(folder, wm_mask_name),
            "bvals": os.path.join(folder, bvals_name),
            "bvecs": os.path.join(folder, bvecs_name),
            "b0": os.path.join(folder, b0_name),
        }
        for folder in folder_list
    ]

    print(len(subject_dict_list))

    trans = Compose(
        [
            LoadImaged(
                keys=["x", "y", "b0"],
                image_only=False,
                ensure_channel_first=True,
                dtype=np.float64,
            ),
            LoadImaged(
                keys=["mask", "wm_mask"],
                image_only=False,
                ensure_channel_first=True,
                dtype=bool,
            ),
            ReadBvalsBvecsd(bvals_key="bvals", bvecs_key="bvecs"),
            CropForegroundd(
                keys=["x", "y", "mask", "wm_mask", "b0"],
                source_key="mask",
                allow_smaller=False,
            ),
            NormalizeByB0d(keys=["x"], b0_key="b0"),
            SphericalFunctionToSphericalHarmonicsMRtrixd(
                keys=["x"], bvals_key="bvals", bvecs_key="bvecs", sh_order=sh_order
            ),
            ToTensord(keys=["x", "y", "mask", "wm_mask"]),
            # GridPatchd(keys=["x", "y", "mask", "wm_mask"], patch_size=(16, 16, 16), overlap=0.5,
            #            pad_mode="constant", constant_values=0),
        ]
    )

    splitter = SlidingWindowSplitter(patch_size=(16, 16, 16), overlap=0.5)

    if not os.path.exists(dataset_patched_dir):
        os.makedirs(dataset_patched_dir)

    for subject_dict in tqdm(subject_dict_list):
        subject = trans(subject_dict)

        print(
            subject["x"].shape,
            subject["y"].shape,
            subject["mask"].shape,
            subject["wm_mask"].shape,
        )

        subject_patch_dir = os.path.join(
            dataset_patched_dir, os.path.basename(os.path.dirname(subject_dict["x"]))
        )
        if not os.path.exists(subject_patch_dir):
            os.makedirs(subject_patch_dir)

        for key in ["x", "y", "mask", "wm_mask"]:  # "y", "mask",

            if num_measurement == 6 and key != "x":
                continue
            patches = splitter(subject[key][None, ...])
            patches = [patch for patch in patches]
            patches = [patch[0][0] for patch in patches]

            print(patches[0].shape, len(patches))

            # # get the metadata from the original image, e.g., affine, spacing, etc.
            # print(subject[key + "_meta_dict"])

            for i, patch in enumerate(patches):

                save_image = SaveImage(
                    output_dir=dataset_patched_dir,
                    output_postfix=f"patch_{i:04d}",
                    separate_folder=False,
                    output_dtype=np.float64,
                    squeeze_end_dims=True,
                    data_root_dir=dataset_root_dir,
                )

                save_image(patch, subject[key + "_meta_dict"])
