import numpy as np
from model import build_davood_net
from utils import train_generator, val_generator, val_generator_w, train_generator_adv, val_generator_w_adv, set_global_seed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
from data import process_data
import os
from tqdm import tqdm
import nibabel as nib
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere
from itertools import product
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow.keras as keras


class Train:
    def __init__(self, batch_size, epochs, test_interval, learning_rate, train_val_sep=70, ckpt_path='checkpoints', mul_coe=10, train_id='00', task_name='test'):

        self.batch_size = batch_size
        self.epochs = epochs
        self.test_interval = test_interval
        self.learning_rate = learning_rate
        self.train_val_sep = train_val_sep
        self.ckpt_path = os.path.join(ckpt_path, task_name, 'train_' + str(train_id))
        self.mul_coe = mul_coe
        self.task_name = task_name
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.train_data = None
        self.train_labels = None
        self.train_mask = None
        self.val_data = None
        self.val_labels = None
        self.val_mask = None
        self.read_data()

        self.SX, self.SY, self.SZ, self.n_sig = self.train_data.shape[1:]
        self.n_tar = self.train_labels.shape[-1]

        self.validation_steps = self.val_data.shape[0]

    def read_data(self):

        if self.task_name == 'test':
            n_subjects = int(1.25 * self.train_val_sep)
            self.train_data = np.ones((n_subjects, 30, 30, 30, 6))
            self.train_labels = np.random.rand(n_subjects, 30, 30, 30, 45)
            self.train_mask = np.ones((n_subjects, 30, 30, 30))
        # elif self.task_name == '{YOUR_TASK_NAME}':
        #     self.train_data, self.train_labels, self.train_mask, self.val_data, self.val_labels, self.val_mask = process_data(
        #         data_dir, dir_list_dir, target_shape=(119, 138, 96), x_begin=16, x_end=135, y_begin=3, y_end=141, z_begin=0, z_end=96)
        else:
            raise NotImplementedError

        self.train_labels *= self.mul_coe

        self.train_data, self.val_data = self.train_data[:self.train_val_sep],  self.train_data[self.train_val_sep:]
        self.train_labels, self.val_labels = self.train_labels[:self.train_val_sep], self.train_labels[self.train_val_sep:]
        self.train_mask, self.val_mask = self.train_mask[:self.train_val_sep], self.train_mask[self.train_val_sep:]

    def configure_model(self):
        # Model hyperparameters
        n_feat = 36
        depth = 3
        ks = 3
        keep_train = 0.9

        self.model = build_davood_net(
            ks, depth, n_feat, self.n_sig, self.n_tar, keep_train, lr=self.learning_rate)
        self.model.compile(tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, weight_decay=0.001
        ), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)

        custom_lr_callback = ReduceLROnPlateau(
            monitor='val_loss', factor=0.9, patience=3, verbose=1,)
        model_checkpoint = WandbModelCheckpoint(
            filepath=os.path.join(self.ckpt_path, 'weights.best.hdf5'),
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            verbose=1)
        tensorboard = TensorBoard(
            log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        wandb_metrics_logger = WandbMetricsLogger(log_freq="batch")

        self.callbacks = [
            # custom_lr_callback,
            model_checkpoint,
            # tensorboard,
            wandb_metrics_logger
        ]

    def train(self):
        self.configure_model()
        self.model.fit(train_generator(self.train_data, self.train_labels, self.train_mask, self.batch_size),
                       epochs=self.epochs,
                       callbacks=self.callbacks,
                       validation_data=val_generator_w(self.val_data, self.val_labels, self.val_mask, self.batch_size),
                       validation_steps=self.validation_steps,
                       workers=-1,
                       use_multiprocessing=True,
                       verbose=2,
                       shuffle=True,
                       steps_per_epoch=self.train_data.shape[0] // self.batch_size * self.test_interval,
                       )


class Test:
    def __init__(self, model_dir, data_dir, data_list, data_file_name, bvec_file_name, output_dir=None, mul_coe=10, n_sig=6, n_tar=45, output_file_name='dl_mix.nii.gz'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.bvec_file_name = bvec_file_name
        self.output_dir = output_dir if output_dir else data_dir
        self.data_list = data_list
        self.mul_coe = mul_coe
        self.n_sig = n_sig
        self.n_tar = n_tar
        self.output_file_name = output_file_name

        self.model = None

    def load_model(self):
        # Model hyperparameters
        n_feat = 36
        depth = 3
        ks = 3
        keep_train = 0.9

        self.model = build_davood_net(
            ks, depth, n_feat, self.n_sig, self.n_tar, keep_train)
        self.model.load_weights(os.path.join(self.model_dir))

    def test(self):
        self.load_model()

        for subject in tqdm(self.data_list):
            subject_dir = os.path.join(self.data_dir, subject)

            dmri_vol = nib.load(os.path.join(
                subject_dir, self.data_file_name))
            affine = dmri_vol.affine
            dmri = dmri_vol.get_fdata()
            dmri = np.clip(dmri, 0, 1)

            bvecs = np.loadtxt(os.path.join(
                subject_dir, self.bvec_file_name)).T
            sphere_bvecs = Sphere(xyz=np.asarray(bvecs))
            dmri = sf_to_sh(dmri, sphere=sphere_bvecs,
                            sh_order=2, basis_type='tournier07')

            SX, SY, SZ = dmri.shape[:3]
            LX, LY, LZ = 16, 16, 16
            test_shift = LX // 3
            lx_list = np.squeeze(np.concatenate((np.arange(0, SX - LX, test_shift)
                                                [:, np.newaxis], np.array([SX - LX])[:, np.newaxis])).astype(int))
            ly_list = np.squeeze(np.concatenate((np.arange(0, SY - LY, test_shift)
                                                [:, np.newaxis], np.array([SY - LY])[:, np.newaxis])).astype(int))
            lz_list = np.squeeze(np.concatenate((np.arange(0, SZ - LZ, test_shift)
                                                [:, np.newaxis], np.array([SZ - LZ])[:, np.newaxis])).astype(int))
            LXc, LYc, LZc = 6, 6, 6

            y_s = np.zeros((SX, SY, SZ, self.n_tar))
            y_c = np.zeros((SX, SY, SZ, ))

            for lx, ly, lz in product(lx_list, ly_list, lz_list):
                if np.any(dmri[lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc, lz + LZc:lz + LZ - LZc, 0]):
                    batch_x = dmri[lx:lx + LX, ly:ly +
                                   LY, lz:lz + LZ, :].copy()[np.newaxis, ...]
                    batch_y = self.model.predict(batch_x, verbose=0)

                    y_s[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += batch_y[0]
                    y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1

            y_s = np.divide(y_s, y_c[..., np.newaxis],
                            out=np.zeros_like(y_s), where=y_c[..., np.newaxis] != 0)
            y_s /= self.mul_coe

            y_s_nii = nib.Nifti1Image(y_s, affine)
            subject_output_dir = os.path.join(self.output_dir, subject)
            nib.save(y_s_nii, os.path.join(
                subject_output_dir, self.output_file_name))


if __name__ == '__main__':
    model_trained_on = '{YOUR_TASK_NAME}'
    train = Train(batch_size=35, epochs=100000, test_interval=100, learning_rate=5e-5,
                  train_val_sep=70, train_id='00', task_name=model_trained_on, mul_coe=10)
    train.train()
