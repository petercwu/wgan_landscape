from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


def gen_w_loss(pred_fake):
    return -tf.reduce_mean(pred_fake)


def generate_noise(num_images=25):
    noise = np.random.normal(0, 1, (num_images, 128)) # 128 is the noise dimension
    return noise


def generate_images(noise):
    generated_images = gen_model.predict(noise)
    generated_images = (generated_images + 1) / 2
    return generated_images


def create_subplots(rows, cols, generated_images, title):
    fig, axes = plt.subplots(rows, cols, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i])
        ax.axis('off')
    plt.suptitle(title, fontsize=30)
    plt.tight_layout()
    plt.savefig("INSERT SAVE PATH HERE" + title)
    plt.close()


# Generates noisy images and feeds the images into the generator model
# The generator model that is used is from epochs 1-500
# The output would be a newly generated landscape image for each generator
noise = generate_noise()
for iter in range(500, 0, -1):
    epoch = str(iter).zfill(3)
    with tf.keras.utils.custom_object_scope({"gen_w_loss": gen_w_loss}):
        gen_model_path = "INSERT GENERATOR MODEL PATH HERE" + str(iter)
        gen_model = keras.models.load_model(gen_model_path)
    images = generate_images(noise)
    create_subplots(5, 5, images, f"wgan-gp_epoch_{epoch}")

