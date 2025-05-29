

import os
import pathlib
import sys

from PIL import Image

import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from autotag.ml.GAN.wgan import critic, generator, gradient_penalty

#resolving path problem
# FILE= pathlib.Path(__file__).resolve()
# ROOT=FILE.parents[1]
# print(ROOT)
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

from utils.utils import save_with_other_dataset
from autotag.ml.generators.standard_pretrained_CNN_EarlyStop import parse_dataset, encode
# The general name of model
MODEL_NAME = "WGAN-GP"

from typing import Callable, Union

# either or not to log on tensorboard
LOG_TO_TENSORBOARD: bool = True
LOG_PATH = pathlib.Path(__file__).parent.joinpath("logs")

# for saving checkpoints and model itself
CHECKPOINTS_PATH = pathlib.Path(__file__).parents[3].joinpath("models").joinpath(MODEL_NAME)

# for saving images
IMAGES_PATH = pathlib.Path(__file__).parents[3].joinpath("temp").joinpath("auto", "images")

# for tensorboard logging
if LOG_TO_TENSORBOARD:
    # for logs
    log_writer = tf.summary.create_file_writer(str(LOG_PATH))

    # for images
    n_images_to_show = 10
    writer_real = tf.summary.create_file_writer(str(LOG_PATH.joinpath(MODEL_NAME).joinpath("real")))
    writer_fake = tf.summary.create_file_writer(str(LOG_PATH.joinpath(MODEL_NAME).joinpath("fake")))

# WGAN Hyperparameters
WGAN_LEARNING_RATE = 1e-4
WGAN_BATCH_SIZE = 128
WGAN_IMAGE_HEIGHT = 64
WGAN_IMAGE_WIDTH = 64
WGAN_Z_DIM = 100
WGAN_FEATURES_CRITIC = 16
WGAN_FEATURES_GEN = 16
WGAN_CRITIC_ITERATIONS = 5
WGAN_LAMBDA_GP = 10


def train_wgan(
    dataset: tf.data.Dataset,
    n_epoch: int,
    save_every_n_epoch: int = 5,
    model_identifier: str = "Generic",
    restore_older_checkpoint: bool = False,
    img_shape: tuple[int, int, int] = (WGAN_IMAGE_HEIGHT, WGAN_IMAGE_WIDTH, 3),
    channel_noise: int = WGAN_Z_DIM,
    batch_size: int = WGAN_BATCH_SIZE,
    learning_rate: int = WGAN_LEARNING_RATE,
    n_feature_critic: int = WGAN_FEATURES_CRITIC,
    n_feature_generator: int = WGAN_FEATURES_GEN,
    n_critic_iteration: int = WGAN_CRITIC_ITERATIONS,
    lambda_gp=WGAN_LAMBDA_GP,
) -> tuple[keras.models.Model, keras.models.Model]:
    """
    Train a wgan model with given dataset.
    The images must already be resized and normalized.

    Parameters
    ----------
    dataset: (batch_size, img_height, img_width, img_channel) \n
    img_shape: (img_height, img_width, img_channel) \n

    Returns
    -------
    A tuple of trained models (generator, critic)


    """

    # wgan model
    generator_model = generator(channel_noise, img_shape[2], n_feature_generator, img_height=img_shape[0], img_width=img_shape[1])
    critic_model = critic(n_feature_critic, img_shape)

    # initializer optimiser
    betas = (0.0, 0.9)

    opt_generator = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=betas[0], beta_2=betas[1]
    )
    opt_critic = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=betas[0], beta_2=betas[1]
    )

    # checkpoint related stuffs
    SPECIFIC_CHECKPOINT_PATH = CHECKPOINTS_PATH.joinpath(model_identifier)
    ckpt = tf.train.Checkpoint(generator=generator_model, critic=critic_model, G_optimizer=opt_generator, D_optimizer=opt_critic)
    ckpt_manager = tf.train.CheckpointManager(ckpt, SPECIFIC_CHECKPOINT_PATH, max_to_keep=3)

    previous_epochs = 1 # start from epoch 1
    if restore_older_checkpoint:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) # assuming there is not other - in the filename
            previous_epochs = latest_epoch*save_every_n_epoch
            print (f'Latest checkpoint of epoch {previous_epochs} restored!!')
    

    # Wasserstein loss for the critic
    def critic_w_loss(pred_real, pred_fake):
        real_loss = tf.reduce_mean(pred_real)
        fake_loss = tf.reduce_mean(pred_fake)
        return fake_loss - real_loss

    # Wasserstein loss for the generator
    def generator_w_loss(pred_fake):
        return -tf.reduce_mean(pred_fake)

    if LOG_TO_TENSORBOARD:
        step = 0 # for tensorboard logging
    
    epoch = previous_epochs
    for epoch in range(previous_epochs, n_epoch + 1):
        print(f"Epoch [{epoch}/{n_epoch}]")
        for idx, real_samples in enumerate(tqdm(dataset)):
            cur_batch_size = tf.shape(real_samples)[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(n_critic_iteration):
                random_latent_vectors = tf.random.normal(
                    shape=(cur_batch_size, channel_noise)
                )

                # Step 1. Train the critic with both real images and fake images
                with tf.GradientTape() as crit_tape:
                    fake_samples = generator_model(random_latent_vectors, training=True)
                    pred_real = critic_model(real_samples, training=True)
                    pred_fake = critic_model(fake_samples, training=True)

                    # Calculate the WGAN-GP gradient penalty
                    gp = gradient_penalty(critic_model, real_samples, fake_samples)

                    # Add gradient penalty to the original critic loss
                    critic_loss = critic_w_loss(pred_real, pred_fake) + gp * lambda_gp

                # Compute critic gradients
                grads = crit_tape.gradient(critic_loss, critic_model.trainable_weights)

            # Update critic weights
            opt_critic.apply_gradients(grads_and_vars=zip(grads, critic_model.trainable_weights))

            # Train the generator
            with tf.GradientTape() as gen_tape:
                # the reason for duplicating these from critic is because
                # use of gradient tape caused the output of gradient to be array of None if 
                # some dependent variables aren't watched
                fake_samples = generator_model(random_latent_vectors, training=True)
                pred_fake = critic_model(fake_samples, training=True)
                gen_loss = generator_w_loss(pred_fake)

            # Compute generator gradients
            grads = gen_tape.gradient(gen_loss, generator_model.trainable_weights)

            # Update generator wieghts
            opt_generator.apply_gradients(grads_and_vars=zip(grads, generator_model.trainable_weights))

            if idx % 100 == 0:
                if LOG_TO_TENSORBOARD:

                    with log_writer.as_default():
                        tf.summary.scalar('gen_loss', gen_loss, step=step)
                        tf.summary.scalar('desc_loss', critic_loss, step=step)

                    # (img + 1 )/127 is done to revert normalization from [-1,1] to [0,1]
                    with writer_real.as_default():
                        real_images = np.reshape(real_samples[:n_images_to_show], (-1,*img_shape))
                        tf.summary.image("Real", (real_images + 1)/2, step=step, max_outputs=batch_size)

                    with writer_fake.as_default():
                        fake_images = np.reshape(fake_samples[:n_images_to_show], (-1, *img_shape))
                        tf.summary.image("Fake", (fake_images + 1)/2, step=step, max_outputs=batch_size)
                    
                    step += 1

                # img = keras.preprocessing.image.array_to_img(fake_samples)
                # img.save(f"generated_images/generated_img{epoch}_{idx}_.png")
                print(f" Critic Loss: {critic_loss}, Generator Loss: {gen_loss}")

        if epoch % save_every_n_epoch == 0:
                ckpt_save_path = ckpt_manager.save()
                print (f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')
            

    # save at the last
    ckpt_save_path = ckpt_manager.save()
    print (f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')

    return (generator_model, critic_model)

# def generate_and_save_images(model: keras.models.Model, noise_input, image_path: pathlib.Path | None = None, identifier="Generated-Img", start_index: int =0) -> np.ndarray:
def generate_and_save_images(model: keras.models.Model, noise_input, image_path: Union[pathlib.Path, None] = None, identifier="Generated-Img", start_index: int =0) -> np.ndarray:
    '''
        Generate images and save it. Here n_images should be equal to the batch size specified during training.\n
        If more images than batch_size is required, the function must be called mutliple times with different noise_input

        Parameters
        ----------
        model: The generator model which accepts latent vector as input \n
        image_path: The path to save generated images, if None, the generated images won't be saved

        Returns 
        -------
        The generated images having shape: (n_images, height, width, channels)
    '''
    predictions = model.predict(noise_input)
    print(image_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for i in range(start_index, predictions.shape[0]):
        img_array = np.uint8((predictions[i] * 127)+127) # from [-1,1] to [0,255]
        img = Image.fromarray(img_array)
        curr_img_path= os.path.join(image_path,f"{identifier}-{i}.jpeg")
        # image_path.joinpath(f"{identifier}-{i}.jpeg")
        img.save(curr_img_path)
        img.close()
      
    return predictions


def train_wgan_on_mnist(n_epoch: int = 10):
    image_height: int = 64
    image_width: int = 64
    batch_size: int = 128
    latent_dim: int = 100

    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
    # this doesn't work
    # train_images = tf.image.resize(train_images, [image_height, image_width])
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1] 

    dataset_shape = tf.shape(train_images)
    total_images = dataset_shape[0]

    try:
        channels = dataset_shape[3] if dataset_shape[3] is not None else 1
    except tf.errors.InvalidArgumentError:
        channels = 1

    # Preprocess dataset
    def preprocess_image(image):
        # Resize image to target dimensions
        image = tf.image.resize(image, [image_height, image_width])
        # Normalize image to [-1, 1]
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        return image
    
    # adding channel dimension if not already present
    train_images = np.reshape(train_images, (total_images, dataset_shape[1], dataset_shape[2], channels))

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    # Apply preprocessing to each image and batch the dataset
    dataset = dataset.map(preprocess_image).batch(batch_size)

    generator, critic = train_wgan(
        dataset,
        n_epoch,
        save_every_n_epoch=1,
        model_identifier="mnist",
        img_shape=(image_height, image_width, channels),
        channel_noise=latent_dim,
        batch_size=batch_size,
    )

    return generator, critic

def train_on_augmented_data(zipped_dataset_path, latent_dim: int = 100, n_epoch = 10, batch_size = 128, image_dim = (WGAN_IMAGE_HEIGHT, WGAN_IMAGE_WIDTH)):
    """
    Train wgan on provided dataset.
    todo: implement pipeline to generate and save images and also incorporate it into the dataset

    Parameters
    ----------
    zipped_dataste_path: The path containing a zip file of the dataset


    Returns
    --------
    The trained model of generator and critic: (generator_model, critic_model)

    """

    (train_images, _,  tags_as_int, _), tags = parse_dataset(zipped_dataset_path, img_dim=(image_dim[1], image_dim[0]), test_size=0.0)
    encoded_tags = encode(tags)
    int_tag_for_other=""
    if 'other' in encoded_tags:
     int_tag_for_other = encoded_tags['other']

    # removing other from dataset, as it is not required
    indices_to_remove = []
    for i, val in enumerate(tags_as_int):
        if val == int_tag_for_other:
            indices_to_remove.append(i)
    train_images = np.delete(train_images, indices_to_remove, 0)


    dataset_shape = tf.shape(train_images)
    total_images = dataset_shape[0]

    try:
        channels = dataset_shape[3] if dataset_shape[3] is not None else 1
    except tf.errors.InvalidArgumentError:
        channels = 1

    # Preprocess dataset
    def preprocess_image(image):
        # Resize image to target dimensions
        image = tf.image.resize(image, [*image_dim])
        image = tf.cast(image, tf.float32)
        # the image obtained from parse_dataset will be normalized to [0,1]
        # so, changing this to [-1, 1]
        image = image*2 - 1
        return image
    
    # adding channel dimension if not already present
    train_images = np.reshape(train_images, (total_images, dataset_shape[1], dataset_shape[2], channels))

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    # Apply preprocessing to each image and batch the dataset
    dataset = dataset.map(preprocess_image).batch(batch_size)

    generator, critic = train_wgan(
        dataset,
        n_epoch,
        save_every_n_epoch=1,
        model_identifier="mnist",
        img_shape=(*image_dim, channels),
        channel_noise=latent_dim,
        batch_size=batch_size,
    )

    return generator, critic


if __name__ == "__main__":
  
    dataset_name = "chair_dataset"
    # train_wgan_on_mnist()
    images_identifier = "augmented-generic"
    # hyperparameters
    
    zipped_dataset_path = IMAGES_PATH.joinpath(f"{dataset_name}.zip")
    latent_vec_dim = 100
    n_epoch = 5
    batch_size = 32 # reducing batch size becuase of lesser number of data
    image_dim = (WGAN_IMAGE_HEIGHT, WGAN_IMAGE_WIDTH)
    n_images_to_generate = 100


    gen, _ = train_on_augmented_data(zipped_dataset_path=zipped_dataset_path, latent_dim=latent_vec_dim, n_epoch=n_epoch, batch_size=batch_size, image_dim=image_dim)
    
    
    image_save_path = IMAGES_PATH.joinpath(f"{dataset_name}.data")
    new_images_path = image_save_path.joinpath("GAN")

    start_index = 0
    for i in range(0, n_images_to_generate + 1, batch_size):
        random_latent_vectors = tf.random.normal(
                shape=(batch_size, latent_vec_dim), seed=i
            )
        images = generate_and_save_images(gen, random_latent_vectors, new_images_path, images_identifier, start_index=start_index)
        start_index = len(images)

    # archieve back
    save_with_other_dataset(zipped_dataset_path, new_images_path, dataset_name=dataset_name[:dataset_name.rfind("_dataset")])
