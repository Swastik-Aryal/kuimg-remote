import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from keras import layers

# setting gpu device
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initializes weights according to the WGAN paper
INITIAL_WEIGHTS_MEAN = 0.0
INITIAL_WEIGHTS_STDDEV = 0.02
# Initializer for trainable layers
INITIALIZER = lambda seed : keras.initializers.RandomNormal(
    mean=INITIAL_WEIGHTS_MEAN, stddev=INITIAL_WEIGHTS_STDDEV, seed=seed
)

last_seed = -1
def get_seed():
    """
    Return unique seed each time

    todo: Determine if random seeds are better than fixed ones and use the best one
    """

    global last_seed
    last_seed += 1
    return last_seed


def _critic_block(
    out_channels: int, kernel_size: int, stride: int, padding: str, alpha: int
) -> keras.models.Model:
    """
    Parameters
    ----------
    out_channels: number of filters to apply to the input \n
    alpha: alpha value for LeakyReLU
    """

    return keras.Sequential(
        [
            layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                use_bias=False,
                kernel_initializer=INITIALIZER(seed=get_seed()),
            ),
            # 2d instance norm layer
            layers.BatchNormalization(
                trainable=True,  # normalize each filters independently (instance norm)
            ),  # the output channel is same as input channel
            layers.LeakyReLU(alpha=alpha),
            layers.Dropout(rate=0.3)
        ]
    )


def critic(
    features_d: int, input_shape: tuple[int, int, int] = (64, 64, 3)
) -> keras.models.Model:
    """
    Also known as discriminator (different activation function than simple discriminator)

    Parameters
    ----------

    features_d: number of features for the first critic convolution layer \n
    input_shape: (height, width, channels) of image

    """
    alpha: float = 0.2

    # since all layers use same kernel_size, stride and padding
    kernel_size = 4
    stride = 2
    padding = "same"

    return keras.Sequential(
        [
            layers.Conv2D(
                filters=features_d,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=INITIALIZER(seed=get_seed()),
                input_shape=input_shape
            ),
            layers.LeakyReLU(alpha),
            # _block(out_channels, kernel_size, stride, padding)
            _critic_block(features_d * 2, kernel_size, stride, padding, alpha=alpha),
            _critic_block(features_d * 4, kernel_size, stride, padding, alpha=alpha),
            _critic_block(features_d * 8, kernel_size, stride, padding, alpha=alpha),
            layers.Flatten(),
            layers.Dense(1, activation="linear"),
        ],
        name="critic",
    )


def _generator_block(
    out_channels: int, kernel_size: int, stride: int, padding: str
) -> keras.models.Model:
    """

    Parameters
    ------

    out_channels: the output of inverse convolution (The output channel is less than the input one)\n

    """

    return keras.Sequential(
        [
            layers.Conv2DTranspose(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                use_bias=False,
                kernel_initializer=INITIALIZER(seed=get_seed()),
            ),
            # 2d batch normalization layer
            layers.BatchNormalization(
                trainable=True,
            ),  # the output channel is same as input channel
            layers.Activation("relu"),
        ]
    )


def generator(
    channels_noise: int,
    channels_img: int,
    features_g: int,
    img_height: int = 64,
    img_width: int = 64,
) -> keras.models.Model:
    """
    Parameters
    ----------
    channels_noise: latent dimension (the generator maps this dimensions into image)\n
    channels_img: number of channesl in image \n
    features_g: In wgan paper this is same as number of features for the first critic convolution layer

    """

    # since all layers use same kernel_size, stride and padding
    kernel_size = 4
    stride = 2
    padding = "same"

    return keras.Sequential(
        [
            layers.Dense(
                2 * 2 * features_g * 16, use_bias=False, input_shape=(channels_noise,)
            ),
            layers.Reshape((2, 2, features_g * 16)),
            _generator_block(
                features_g * 16, kernel_size, stride, padding
            ),  # from 4*4 to features_g * 16
            _generator_block(features_g * 8, kernel_size, stride, padding),  # img: 8x8
            _generator_block(
                features_g * 4, kernel_size, stride, padding
            ),  # img: 16x16
            _generator_block(
                features_g * 2, kernel_size, stride, padding
            ),  # img: 32x32
            layers.Conv2DTranspose(
                filters=channels_img,
                kernel_size=5,
                strides=stride,
                padding=padding,
                kernel_initializer=INITIALIZER(seed=get_seed()),
            ),
            keras.layers.Resizing(img_height, img_width, interpolation="bilinear"),
            # Output: N x height x width * n_channels
            layers.Activation("tanh"),
        ],
        name="generator",
    )


def gradient_penalty(critic: keras.models.Model, real: tf.Tensor, fake: tf.Tensor):
    BATCH_SIZE = tf.shape(real)[0]

    # Generate random values for epsilon
    epsilon = tf.random.uniform(
        shape=[BATCH_SIZE, 1, 1, 1], minval=0, maxval=1, dtype=dtypes.float32
    )

    # 1. Interpolate between real and fake samples
    interpolated_samples = epsilon * real + ((1 - epsilon) * fake)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        # 2. Get the Critic's output for the interpolated image
        logits = critic(interpolated_samples)

    # 3. Calculate the gradients w.r.t to the interpolated image
    gradients = tape.gradient(logits, interpolated_samples)

    # 4. Calculate the L2 norm of the gradients.
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))

    # 5. Calculate gradient penalty
    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)

    return gradient_penalty


def test():
    BATCH_SIZE, image_height, image_width, image_channels = 8, 28, 28, 1
    latent_dim = 100  # the noise channel
    features_d = 16
    features_g = 16

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

    x = initializer(shape=(BATCH_SIZE, image_height, image_width, image_channels))
    _critic = critic(
        features_d, input_shape=(image_height, image_width, image_channels)
    )
    _critic.summary()
    critic_shape = _critic(x).shape
    print("Critic Shape: ", critic_shape)
    assert critic_shape == (BATCH_SIZE, 1), "Critic test failed"

    z = initializer(shape=(BATCH_SIZE, latent_dim))
    gen = generator(latent_dim, image_channels, features_g, image_height, image_width)
    gen.summary()
    generator_shape = gen(z).shape
    print("Generator shape: ", generator_shape)
    assert generator_shape == (
        BATCH_SIZE,
        image_height,
        image_width,
        image_channels,
    ), "Generator test failed"


if __name__ == "__main__":
    test()
