import tensorflow as tf
import numpy as np
import random
import os


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train_generator(
    X,
    Y,
    batch_size,
):

    # X: (n_voxels, n_sig)
    # Y: (n_voxels, n_tar)
    # batch_X: (batch_size, n_sig)
    # batch_Y: (batch_size, n_tar)

    while True:
        # Shuffle the data
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        Y = Y[indices]

        # Generate batches
        for i in range(0, X.shape[0], batch_size):

            batch_X = (
                X[i : i + batch_size].copy()
                if i + batch_size < X.shape[0]
                else X[i:].copy()
            )
            batch_Y = (
                Y[i : i + batch_size].copy()
                if i + batch_size < Y.shape[0]
                else Y[i:].copy()
            )

            # yield batch_X, batch_Y
            with tf.device("/cpu:0"):
                batch_X = tf.convert_to_tensor(batch_X, dtype=tf.float32)
                batch_Y = tf.convert_to_tensor(batch_Y, dtype=tf.float32)

            yield batch_X, batch_Y

            del batch_X, batch_Y


def main():
    # Assume n_voxels = 10000, n_sig = 45, n_tar = 45
    n_voxels, n_sig, n_tar = 10000, 45, 45

    # Generate some random data
    X = np.random.rand(n_voxels, n_sig)
    Y = np.random.rand(n_voxels, n_tar)

    # Define batch size
    batch_size = 2000

    # Create generator
    gen = train_generator(X, Y, batch_size)

    # Get a batch
    batch_X, batch_Y = next(gen)

    # Print shapes of the batch to verify
    print(batch_X.shape)  # Expected output: (32, 20)
    print(batch_Y.shape)  # Expected output: (32, 10)


if __name__ == "__main__":
    main()
