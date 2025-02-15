import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
import numpy as np
from itertools import product


class DavoodModel(Model):
    """
    This is a DavoodNet model class, for the purpose of the different validation method
    below -- instead of computing the loss and metrics on 16x16x16 blocks, do it on the
    whole image, to make sure the model performs well cross-blocks.
    """

    def test_step(self, data):
        X, Y = data

        M = X[..., 0] != 0
        SX, SY, SZ = X.shape[1:4]
        n_sig, n_tar = X.shape[-1], Y.shape[-1]
        Y_pred = np.zeros_like(Y)
        Y_pred_c = np.zeros(Y.shape[:-1])

        LX, LY, LZ = 16, 16, 16
        test_shift = LX // 3
        lx_list = np.squeeze(
            np.concatenate(
                (
                    np.arange(0, SX - LX, test_shift)[:, np.newaxis],
                    np.array([SX - LX])[:, np.newaxis],
                )
            ).astype(int)
        )
        ly_list = np.squeeze(
            np.concatenate(
                (
                    np.arange(0, SY - LY, test_shift)[:, np.newaxis],
                    np.array([SY - LY])[:, np.newaxis],
                )
            ).astype(int)
        )
        lz_list = np.squeeze(
            np.concatenate(
                (
                    np.arange(0, SZ - LZ, test_shift)[:, np.newaxis],
                    np.array([SZ - LZ])[:, np.newaxis],
                )
            ).astype(int)
        )
        LXc, LYc, LZc = 6, 6, 6

        batch_size = 64
        batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
        batch_y_pred = np.zeros((batch_size, LX, LY, LZ, n_tar))
        n_batch = 0
        batches = []

        for lx, ly, lz in product(lx_list, ly_list, lz_list):
            if np.all(
                M[
                    0:1,
                    lx + LXc : lx + LX - LXc,
                    ly + LYc : ly + LY - LYc,
                    lz + LZc : lz + LZ - LZc,
                ]
            ):
                batch_x[n_batch, ...] = X[
                    0:1, lx : lx + LX, ly : ly + LY, lz : lz + LZ, :
                ]
                n_batch += 1
                batches.append((lx, ly, lz))

            if n_batch == batch_size:
                batch_y_pred = self(batch_x, training=False)
                for i, (lxx, lyy, lzz) in enumerate(batches):
                    Y_pred[
                        0:1, lxx : lxx + LX, lyy : lyy + LY, lzz : lzz + LZ, :
                    ] += batch_y_pred[i, ...]
                    Y_pred_c[0:1, lxx : lxx + LX, lyy : lyy + LY, lzz : lzz + LZ] += 1
                n_batch = 0
                batches = []
                batch_x = np.zeros((batch_size, LX, LY, LZ, n_sig))
                batch_y_pred = np.zeros((batch_size, LX, LY, LZ, n_tar))

        if n_batch > 0:
            batch_y_pred = self(batch_x[:n_batch], training=False)
            for i, (lxx, lyy, lzz) in enumerate(batches):
                Y_pred[
                    0:1, lxx : lxx + LX, lyy : lyy + LY, lzz : lzz + LZ, :
                ] += batch_y_pred[i, ...]
                Y_pred_c[0:1, lxx : lxx + LX, lyy : lyy + LY, lzz : lzz + LZ] += 1

        # Y_pred /= Y_pred_c[..., np.newaxis]
        # change to nonzero division
        Y_pred = np.divide(
            Y_pred,
            Y_pred_c[..., np.newaxis],
            out=np.zeros_like(Y_pred),
            where=Y_pred_c[..., np.newaxis] != 0,
        )
        Y_pred = tf.convert_to_tensor(Y_pred, dtype=tf.float32)

        Y = Y[M]
        Y_pred = Y_pred[M]
        self.compute_loss(y=Y, y_pred=Y_pred)

        return self.compute_metrics(x=None, y=Y, y_pred=Y_pred, sample_weight=None)


def conv_block(
    x,
    n_feat_0,
    n_feat,
    ks,
    strd,
    bias_init=0.001,
    name=None,
    p_keep_conv=1.0,
    activation="relu",
):
    """
    This is a wrapper function for the convolutional block in DavoodNet.
    """
    n_l = n_feat_0 * ks**3
    s_dev = tf.math.sqrt(2.0 / n_l)

    x = layers.Conv3D(
        n_feat,
        ks,
        strides=strd,
        padding="SAME",
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=s_dev),
        bias_initializer=tf.keras.initializers.Constant(value=bias_init),
        name="conv_" + name,
        activation=activation,
    )(x)
    if p_keep_conv <= 1.0 and p_keep_conv > 0.0:
        x = layers.Dropout(1 - p_keep_conv, name="dropout_" + name)(x)

    return x


def conv_trans_block(
    x,
    n_feat_0,
    n_feat,
    ks,
    strd,
    bias_init=0.001,
    name=None,
    p_keep_conv=1.0,
    activation="relu",
):
    """
    This is a wrapper function for the transposed convolutional block in DavoodNet.
    """
    n_l = n_feat_0 * ks**3
    s_dev = tf.math.sqrt(2.0 / n_l)

    x = layers.Conv3DTranspose(
        n_feat,
        ks,
        strides=strd,
        padding="SAME",
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=s_dev),
        bias_initializer=tf.keras.initializers.Constant(value=bias_init),
        name="conv_" + name,
        activation=activation,
    )(x)
    x = layers.Dropout(1 - p_keep_conv, name="dropout_" + name)(x)

    return x


def build_davood_net(
    ks_0,
    depth,
    n_feat_0,
    n_channel,
    n_class,
    p_keep_conv,
    bias_init=0.001,
    sx=16,
    sy=16,
    sz=16,
    lr=0.001,
):

    feat_fine = [None] * (depth - 1)
    x0 = layers.Input(shape=(sx, sy, sz, n_channel))

    # Encoding path
    for level in range(depth):
        ks = ks_0
        strd = 1 if level == 0 else 2

        x = conv_block(
            x0,
            n_channel,
            n_feat_0,
            ks,
            strd,
            bias_init=bias_init,
            name=str(level) + "_init",
            p_keep_conv=p_keep_conv,
        )

        if level != 0:
            for i in range(1, level):
                x = conv_block(
                    x,
                    n_feat_0,
                    n_feat_0,
                    ks,
                    strd,
                    bias_init=bias_init,
                    name=str(level) + "_" + str(i) + "_init",
                    p_keep_conv=p_keep_conv,
                )

            for level_reg in range(level):
                x_0 = feat_fine[level_reg]
                level_diff = level - level_reg
                n_feat = n_feat_0 * 2**level_reg

                for j in range(level_diff):
                    x_0 = conv_block(
                        x_0,
                        n_feat,
                        n_feat,
                        ks,
                        strd,
                        bias_init=bias_init,
                        name=str(level) + "_" + str(level_reg) + "_" + str(j) + "_reg",
                        p_keep_conv=p_keep_conv,
                    )

                x = layers.Concatenate(
                    name="concat_" + str(level) + "_" + str(level_reg) + "_reg"
                )([x, x_0])

        ks = ks_0
        n_feat = n_feat_0 * 2**level

        x_0 = x
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_2_down",
            p_keep_conv=p_keep_conv,
        )
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_3_down",
            p_keep_conv=p_keep_conv,
        )
        x = layers.Add(name="conv_add_" + str(level) + "_1")([x, x_0])

        x_1 = x
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_4_down",
            p_keep_conv=p_keep_conv,
        )
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_5_down",
            p_keep_conv=p_keep_conv,
        )
        x = layers.Add(name="conv_add_" + str(level) + "_2")([x, x_1, x_0])

        if level < depth - 1:
            feat_fine[level] = x

    # Decoding path
    for level in range(depth - 2, -1, -1):
        ks = ks_0

        x = conv_trans_block(
            x,
            n_feat,
            n_feat // 2,
            ks,
            2,
            bias_init=bias_init,
            name=str(level) + "_up",
            p_keep_conv=p_keep_conv,
        )
        x = layers.Concatenate(name="concat_" + str(level) + "_up")(
            [feat_fine[level], x]
        )

        n_concat = n_feat if level == depth - 2 else n_feat * 3 // 4
        n_feat = n_feat // 2 if level < depth - 2 else n_feat

        x = conv_block(
            x,
            n_concat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_1_up",
            p_keep_conv=p_keep_conv,
        )

        x_0 = x
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_2_up",
            p_keep_conv=p_keep_conv,
        )
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_3_up",
            p_keep_conv=p_keep_conv,
        )
        x = layers.Add(name="deconv_add_" + str(level) + "_1")([x, x_0])

        x_1 = x
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_4_up",
            p_keep_conv=p_keep_conv,
        )
        x = conv_block(
            x,
            n_feat,
            n_feat,
            ks,
            1,
            bias_init=bias_init,
            name=str(level) + "_5_up",
            p_keep_conv=p_keep_conv,
        )
        x = layers.Add(name="deconv_add_" + str(level) + "_2")([x, x_1, x_0])

    # Output
    output = conv_block(
        x,
        n_feat,
        n_class,
        ks,
        1,
        bias_init=bias_init,
        name="out",
        p_keep_conv=1,
        activation=None,
    )
    output_s = conv_block(
        x,
        n_feat,
        1,
        ks,
        1,
        bias_init=bias_init,
        name="out_s",
        p_keep_conv=1,
        activation=None,
    )

    # model = Model(inputs=x0, outputs=[output, output_s], name='DavoodNet')
    # model = Model(inputs=x0, outputs=[output], name='DavoodNet')
    model = DavoodModel(inputs=x0, outputs=[output], name="DavoodNet")

    # model.compile(optimizer=keras.optimizers.Adam(
    #     learning_rate=tf.Variable(lr)), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)

    return model
