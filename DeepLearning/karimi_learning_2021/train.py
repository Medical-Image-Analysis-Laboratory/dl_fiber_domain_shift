from data import process_data
from utils import train_generator, set_global_seed
from mlp import build_mlp_SH

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

# set the training device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

import argparse


class Train:
    def __init__(
        self,
        batch_size,
        n_epochs,
        n_steps_per_epoch,
        train_val_sep,
        n_workers,
        n_sig,
        n_tar,
        n_feat_vec,
        p_keep_hidden,
        bias_init,
        data_dir,
        ckpt_dir,
        dir_list_dir,
        model_save_name,
        target_shape=(119, 138, 96),
        x_begin=16,
        x_end=135,
        y_begin=3,
        y_end=141,
        z_begin=0,
        z_end=96,
        dwi_name="dwi_6_1000_sh.nii.gz",
        lr=0.001,
        mul_coe=10,
        train_id="00",
        task_name="test",
        ckpt_finetune_dir=None,
        y_name="wm.nii.gz",
        mask_name="mask.nii.gz",
        b0_name="b0.nii.gz",
        bvec_name=".bvec",
        N_grad=45,
        sh_order=8,
        age_group=None,
        gt_kind="SS3T",
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.train_val_sep = train_val_sep
        self.n_workers = n_workers
        self.n_sig = n_sig
        self.n_tar = n_tar
        self.n_feat_vec = n_feat_vec
        self.p_keep_hidden = p_keep_hidden
        self.bias_init = bias_init
        self.data_dir = data_dir
        self.ckpt_dir = os.path.join(ckpt_dir, task_name, "train_" + train_id)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.dir_list_dir = dir_list_dir
        self.model_save_name = (
            model_save_name + ".h5"
            if not model_save_name.endswith(".h5")
            and not model_save_name.endswith(".hdf5")
            else model_save_name
        )
        self.target_shape = target_shape
        self.x_begin = x_begin
        self.x_end = x_end
        self.y_begin = y_begin
        self.y_end = y_end
        self.z_begin = z_begin
        self.z_end = z_end
        self.dwi_name = dwi_name
        self.y_name = y_name
        self.mask_name = mask_name
        self.b0_name = b0_name
        self.bvec_name = bvec_name
        self.N_grad = N_grad
        self.sh_order = sh_order
        self.lr = lr
        self.mul_coe = mul_coe
        self.train_id = train_id
        self.task_name = task_name
        self.ckpt_finetune_dir = ckpt_finetune_dir
        self.age_group = age_group
        self.gt_kind = gt_kind

        self.read_data()
        self.n_samples = self.train_data.shape[0]
        print(self.n_samples)
        self.validation_steps = np.ceil(self.n_samples // self.batch_size).astype(int)
        self.n_steps_per_epoch = np.ceil(
            self.val_data.shape[0] // self.batch_size
        ).astype(int)
        print(self.validation_steps)
        print(self.n_steps_per_epoch)
        self.configure_model()

    def read_data(self):
        # Process the data
        train_data, train_labels, train_mask = process_data(
            self.data_dir,
            self.dir_list_dir,
            self.target_shape,
            self.x_begin,
            self.x_end,
            self.y_begin,
            self.y_end,
            self.z_begin,
            self.z_end,
            self.dwi_name,
            self.y_name,
            self.mask_name,
            self.b0_name,
            self.N_grad,
            self.n_sig,
            self.bvec_name,
            self.sh_order,
        )

        # shuffle the data, labels and mask, keeping the same order
        length = train_data.shape[0]
        index = np.arange(length)
        np.random.shuffle(index)
        train_data = train_data[index]
        train_labels = train_labels[index]
        train_mask = train_mask[index]

        self.train_data = []
        self.train_labels = []
        for i in range(train_data.shape[0]):
            data = train_data[i]
            label = train_labels[i]
            mask = train_mask[i]

            # only keep the masked voxels, and flatten the data, label and mask on the middle three dimensions, i.e. (W, H, D)
            # and append them to the corresponding lists
            self.train_data.append(data[mask].reshape((-1, data.shape[-1])))
            self.train_labels.append(label[mask].reshape((-1, label.shape[-1])))

        self.train_data_ = np.concatenate(self.train_data, axis=0).copy()
        self.train_labels_ = np.concatenate(self.train_labels, axis=0).copy()
        del self.train_data, self.train_labels
        self.train_data = self.train_data_
        self.train_labels = self.train_labels_
        print(self.train_data.shape, self.train_labels.shape)

        self.train_labels *= self.mul_coe

        self.train_val_sep = int(self.train_data.shape[0] * self.train_val_sep)
        print(self.train_val_sep)
        self.train_data, self.val_data = (
            self.train_data[: self.train_val_sep],
            self.train_data[self.train_val_sep :],
        )
        self.train_labels, self.val_labels = (
            self.train_labels[: self.train_val_sep],
            self.train_labels[self.train_val_sep :],
        )

    def configure_model(self):
        self.model = build_mlp_SH(self.n_feat_vec, self.p_keep_hidden, self.bias_init)
        if self.ckpt_finetune_dir is not None:
            self.model.load_weights(self.ckpt_finetune_dir)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=keras.losses.MeanSquaredError(),
            run_eagerly=True,
        )

        # self.gt_kind = 'SS3T' if 'ss3t' in self.y_name else 'MSMT'

        model_checkpoint = WandbModelCheckpoint(
            filepath=os.path.join(self.ckpt_dir, self.model_save_name),
            monitor="val_loss",
            # save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.ckpt_dir, "logs"),
            histogram_freq=1,
            profile_batch=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        )
        wandb_logger = WandbMetricsLogger(
            log_freq="batch",
        )

        self.callbacks = [model_checkpoint, tensorboard, wandb_logger]

    def train(self):
        self.model.fit(
            train_generator(self.train_data, self.train_labels, self.batch_size),
            steps_per_epoch=self.validation_steps,  # naming is wrong, but it is the number of steps per epoch, lol
            epochs=self.n_epochs,
            validation_data=train_generator(
                self.val_data, self.val_labels, self.batch_size
            ),
            validation_steps=self.n_steps_per_epoch,
            callbacks=self.callbacks,
            workers=self.n_workers,
            use_multiprocessing=True,
            verbose=2,
            shuffle=True,
        )


if __name__ == "__main__":
    set_global_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", type=int, default=2000)
    parser.add_argument("-n_epochs", type=int, default=10)
    parser.add_argument("-n_steps_per_epoch", type=int, default=0)
    parser.add_argument("-train_val_sep", type=float, default=0.8)
    parser.add_argument("-n_workers", type=int, default=8)
    parser.add_argument("-n_sig", type=int, default=45)
    parser.add_argument("-n_tar", type=int, default=45)
    parser.add_argument("-n_grad", type=int, default=45)
    parser.add_argument("-p_keep_hidden", type=float, default=1.0)
    parser.add_argument("-bias_init", type=float, default=0.001)
    parser.add_argument("-data_dir", type=str, default="/mnt/lrz/data/dHCP_train/")
    parser.add_argument("-ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "-dir_list_dir",
        type=str,
        default="/mnt/lrz/data/dHCP_train/dhcp_sampled_new.txt",
    )
    parser.add_argument("-gt_kind", type=str, default="SS3T")
    parser.add_argument("-sh_order", type=int, default=8)
    parser.add_argument("-lr", type=float, default=0.0002)
    parser.add_argument("-mul_coe", type=int, default=10)
    parser.add_argument("-age_group", type=str, default="early")
    parser.add_argument("-model_save_name", type=str)
    parser.add_argument("-ckpt_finetune_dir", type=str, default=None)

    args = parser.parse_args()

    model_trained_on = f"BCP--MLP_{args.gt_kind}_{args.n_grad}_{args.age_group}"
    wandb.init(
        project="FODNet",
        entity="tobiasforest",
        name=model_trained_on,
        save_code=True,
        mode="online",
    )
    wandb.run.log_code(os.getcwd())

    train = Train(
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_val_sep=args.train_val_sep,
        n_workers=args.n_workers,
        n_sig=args.n_sig,
        n_tar=args.n_tar,
        N_grad=args.n_grad,
        n_feat_vec=[args.n_sig, 300, 300, 300, 400, 500, 600, args.n_tar],
        p_keep_hidden=args.p_keep_hidden,
        bias_init=args.bias_init,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        dir_list_dir=args.dir_list_dir,  # f'/mnt/lrz/data/dHCP_age_new/{args.age_group}_train_50.txt',
        model_save_name=args.model_save_name,
        target_shape=(140, 140, 96),
        x_begin=0,
        x_end=140,
        y_begin=0,
        y_end=140,
        z_begin=0,
        z_end=96,
        dwi_name=f"dwi_{args.n_grad}_1000_orig.nii.gz",
        lr=args.lr,
        mul_coe=args.mul_coe,
        train_id="02",
        task_name="MLP",
        ckpt_finetune_dir=args.ckpt_finetune_dir,
        y_name="wm_ss3t_20_0_88_1000.nii.gz" if args.gt_kind == "SS3T" else "wm.nii.gz",
        mask_name="mask_bet.nii.gz" if "BCP" in args.data_dir else "mask.nii.gz",
        b0_name="b0.nii.gz",
        bvec_name=f"dwi_{args.n_grad}_1000.bvec",
        sh_order=args.sh_order,
        age_group=args.age_group,
        gt_kind=args.gt_kind,
    )

    train.train()
