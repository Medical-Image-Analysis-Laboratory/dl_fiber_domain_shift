import tensorflow as tf
from keras.distribute import distributed_file_utils
import tensorflow.keras.callbacks as cb
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from itertools import product
import numpy as np
import random
import os


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train_generator(X, Y, Mask, batch_size, ):
    SX, SY, SZ = X.shape[1:4]
    n_sig, n_tar = X.shape[-1], Y.shape[-1]
    LX, LY, LZ = 16, 16, 16
    LXc, LYc, LZc = 6, 6, 6
    M = X[..., 0] != 0

    batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
    batch_y = np.zeros((batch_size, LX, LY, LZ, n_tar))
    n_batches = 0

    while True:
        shuffle_idx = np.random.permutation(X.shape[0])
        # shuffle_idx = np.arange(X.shape[0])
        for i in shuffle_idx:
            while True:
                x_i = np.random.randint(SX - LX - 1)
                y_i = np.random.randint(SY - LY - 1)
                z_i = np.random.randint(SZ - LZ - 1)

                batch_m = M[i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ].copy()
                cond_0 = np.all(
                    batch_m[LXc:LX - LXc, LYc:LY - LYc, LZc:LZ - LZc])
                cond_1 = Mask[i, x_i + LX // 2,
                              y_i + LY // 2, z_i + LZ // 2] > 0.5

                if cond_0 and cond_1:
                    batch_x[n_batches] = X[i, x_i:x_i + LX,
                                           y_i:y_i + LY, z_i:z_i + LZ].copy()
                    batch_y[n_batches] = Y[i, x_i:x_i + LX,
                                           y_i:y_i + LY, z_i:z_i + LZ].copy()
                    n_batches += 1
                    break

            if n_batches == batch_size:
                yield batch_x, batch_y
                n_batches = 0
                batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
                batch_y = np.zeros((batch_size, LX, LY, LZ, n_tar))

        if n_batches > 0:
            yield batch_x[:n_batches], batch_y[:n_batches]


def train_generator_adv(X, Y, Mask, batch_size, ):
    SX, SY, SZ = X.shape[1:4]
    n_sig, n_tar = X.shape[-1], Y.shape[-1]
    LX, LY, LZ = 16, 16, 16
    LXc, LYc, LZc = 6, 6, 6
    M = X[..., 0] != 0

    batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
    batch_y = np.zeros((batch_size, LX, LY, LZ, n_tar))
    batch_adv = np.zeros((batch_size, 1), dtype=np.uint8)
    n_batches = 0

    while True:
        shuffle_idx = np.random.permutation(X.shape[0])
        for i in shuffle_idx:
            while True:
                x_i = np.random.randint(SX - LX - 1)
                y_i = np.random.randint(SY - LY - 1)
                z_i = np.random.randint(SZ - LZ - 1)

                batch_m = M[i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ].copy()
                cond_0 = np.all(
                    batch_m[LXc:LX - LXc, LYc:LY - LYc, LZc:LZ - LZc])
                cond_1 = Mask[i, x_i + LX // 2,
                              y_i + LY // 2, z_i + LZ // 2] > 0.5

                if cond_0 and cond_1:
                    batch_x[n_batches] = X[i, x_i:x_i + LX,
                                           y_i:y_i + LY, z_i:z_i + LZ].copy()
                    batch_y[n_batches] = Y[i, x_i:x_i + LX,
                                           y_i:y_i + LY, z_i:z_i + LZ].copy()
                    batch_adv[n_batches] = i % 2
                    n_batches += 1
                    break

            if n_batches == batch_size:
                yield batch_x, [batch_y, batch_adv]
                n_batches = 0
                batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
                batch_y = np.zeros((batch_size, LX, LY, LZ, n_tar))
                batch_adv = np.zeros((batch_size, 1))

        if n_batches > 0:
            yield batch_x[:n_batches], [batch_y[:n_batches], batch_adv[:n_batches]]


def val_generator(X, Y, Mask, batch_size=1):
    SX, SY, SZ = X.shape[1:4]
    n_sig, n_tar = X.shape[-1], Y.shape[-1]
    LX, LY, LZ = 16, 16, 16
    test_shift = LX // 3
    lx_list = np.squeeze(np.concatenate((np.arange(0, SX - LX, test_shift)
                         [:, np.newaxis], np.array([SX - LX])[:, np.newaxis])).astype(int))
    ly_list = np.squeeze(np.concatenate((np.arange(0, SY - LY, test_shift)
                         [:, np.newaxis], np.array([SY - LY])[:, np.newaxis])).astype(int))
    lz_list = np.squeeze(np.concatenate((np.arange(0, SZ - LZ, test_shift)
                         [:, np.newaxis], np.array([SZ - LZ])[:, np.newaxis])).astype(int))
    LXc, LYc, LZc = 6, 6, 6

    while True:

        for i in range(X.shape[0]):
            batch_x = np.empty((batch_size, LX, LY, LZ, n_sig))
            batch_y = np.empty((batch_size, LX, LY, LZ, n_tar))

            n_batch = 0

            for lx, ly, lz in product(lx_list, ly_list, lz_list):
                cond_1 = np.all(
                    Mask[i:i+1, lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc])

                if cond_1:
                    batch_x[n_batch] = X[i:i+1, lx:lx+LX, ly:ly+LY, lz:lz+LZ]
                    batch_y[n_batch] = Y[i:i+1, lx:lx+LX, ly:ly+LY, lz:lz+LZ]
                    n_batch += 1

                if n_batch == batch_size:
                    yield batch_x, batch_y
                    n_batch = 0
                    batch_x = np.empty((batch_size, LX, LY, LZ, n_sig))
                    batch_y = np.empty((batch_size, LX, LY, LZ, n_tar))

            if n_batch > 0:
                yield batch_x[:n_batch], batch_y[:n_batch]


def val_generator_w(X, Y, Mask, batch_size=1):
    while True:
        for i in range(X.shape[0]):
            yield X[i:i+1], Y[i:i+1]


def val_generator_w_adv(X, Y, Mask, batch_size=1):
    while True:
        for i in range(X.shape[0]):
            yield X[i:i+1], [Y[i:i+1], np.array([[i % 2]])]


if __name__ == '__main__':

    import numpy as np

    X = np.random.rand(100, 100, 100, 100, 1)
    Y = np.random.rand(100, 100, 100, 100, 1)
    Mask = np.random.randint(0, 2, (100, 100, 100, 100))

    gen = val_generator_w_adv(X, Y, Mask, batch_size=1)
    for _ in range(10):
        data, labels = next(gen)
        print(data.shape, labels[0].shape, labels[1].shape)
