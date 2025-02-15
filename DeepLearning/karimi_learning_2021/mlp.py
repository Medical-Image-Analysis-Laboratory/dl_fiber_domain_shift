import tensorflow as tf
import tensorflow.keras as keras


def build_mlp_SH(n_feat_vec, p_keep_hidden, bias_init=0.001):
    model = keras.models.Sequential()
    for i in range(len(n_feat_vec) - 1):
        n_feat_in, n_feat_out = n_feat_vec[i], n_feat_vec[i + 1]
        model.add(
            keras.layers.Dense(
                n_feat_out,
                activation="relu" if i < len(n_feat_vec) - 2 else None,
                kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                bias_initializer=keras.initializers.Constant(value=bias_init),
                input_shape=(n_feat_in,),
                name=f"dense_layer_{i+1}",
            )
        )
        model.add(
            keras.layers.Dropout(rate=1 - p_keep_hidden, name=f"dropout_layer_{i+1}")
        )
    return model


if __name__ == "__main__":
    n_sig = 15
    n_tar = 45
    n_feat_vec = [n_sig, 300, 300, 300, 400, 500, 600, n_tar]
    p_keep_hidden = 1.0
    model = build_mlp_SH(n_feat_vec, p_keep_hidden)
    model.summary()

    x = tf.random.uniform((2000, n_sig))
    y = model(x)
    print(y.shape)

    # save the model
    model.save("mlp_SH_.ckpt")
