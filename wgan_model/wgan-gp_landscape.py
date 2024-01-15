import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# Hyperparameters
tf.random.set_seed(42)
batch_size = 128
input_shape = (64, 64, 3)
noise_dim = 128
image_path = "INSERT IMAGE PATH HERE"
epochs = 500
lr = 0.0001

# Referenced Rokas Liuberskis's and keras's WGAN tutorial for WGAN_GP class
class WGAN_GP(tf.keras.models.Model):
    def __init__(
        self,
        discriminator,
        generator,
        noise_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0
    ):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, disc_optimizer, gen_optimizer, disc_w_loss, gen_w_loss_):
        super().compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_w_loss = disc_w_loss
        self.gen_w_loss_ = gen_w_loss_

    def add_noise(self, x, stddev=0.1):
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=stddev, dtype=x.dtype)
        return x + noise

    def gradient_penalty(self, real_samples, fake_samples, discriminator):
        # Calculates gradient penalty on interpolated data and added to discriminator loss
        batch_size = tf.shape(real_samples)[0]
        # Random values of epsilon
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        # Interpolated between real and fake samples
        interpolated_samples = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            logits = discriminator(interpolated_samples, training=True)

        # Calculate gradients wrt to interpolated samples
        gradients = tape.gradient(logits, interpolated_samples)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        noise = tf.random.normal([batch_size, noise_dim])
        gps = []

        # Train the discriminator with real and fake samples
        # Train the discriminator more than the generator
        for _ in range(5):
            with tf.GradientTape() as tape:
                fake_samples = self.generator(noise, training=True)
                pred_real = self.discriminator(real_samples)
                pred_fake = self.discriminator(fake_samples)
                # Add noise
                real_samples = self.add_noise(real_samples)
                fake_samples = self.add_noise(fake_samples)
                # Calculate gradient penalty
                gp = self.gradient_penalty(real_samples, fake_samples, self.discriminator)
                gps.append(gp)
                # Add gradient loss to discriminator loss
                disc_loss = self.disc_w_loss(pred_real, pred_fake) + gp * self.gp_weight
            grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_samples = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_samples, training=True)
            gen_loss = self.gen_w_loss_(pred_fake)
        # Compute generator gradients
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.compiled_metrics.update_state(real_samples, fake_samples)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"d_loss": disc_loss, "g_loss": gen_loss, "gp": tf.reduce_mean(gps)})

        return results

class SaveModelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, discriminator):
        super(SaveModelsCallback, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.best_val_loss = float('inf')

    # Saves the generator and discriminator after each epoch
    def on_epoch_end(self, epoch, logs=None):
        self.generator.save(f"INSERT SAVE PATH HERE{epoch + 1 }")
        self.discriminator.save(f"INSERT SAVE PATH HERE{epoch + 1 }")


class PlotLossesCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PlotLossesCallback, self).__init__()
        self.tot_logs = {}

    def on_epoch_end(self, epoch, logs=None):
        plt.figure(figsize=(8, 6))
        # Appends the current values in the log dict to tot_logs
        for loss, value in logs.items():
            if loss in self.tot_logs:
                self.tot_logs[loss].append(value)
            else:
                self.tot_logs[loss] = [value]
        # Plots the losses as the model trains
        for loss in self.tot_logs:
            plt.plot(self.tot_logs[loss], label=loss, marker='o')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"WGAN-GP {loss + 1}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"wgan-gp_training_{loss + 1}.png")
            plt.close()


# Gradually decrease the learning rate as training goes on to make the optimization converge easier
class LRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, decay_epochs=epochs, min_lr=0.00001):
        super(LRSchedule, self).__init__()
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
        self.gen_lr = [lr]
        self.disc_lr = [lr]
        self.compiled = False

    # After each epoch, plot the learning rate for the generator and discriminator
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(figsize=(10, 6))
        if not self.compiled:
            self.generator_lr = self.model.gen_optimizer.lr.numpy()
            self.discriminator_lr = self.model.disc_optimizer.lr.numpy()
            self.compiled = True
        if epoch < self.decay_epochs:
            new_gen_lr = max(self.generator_lr * (1 - (epoch / self.decay_epochs)), self.min_lr)
            self.model.gen_optimizer.lr.assign(new_gen_lr)
            self.gen_lr.append(new_gen_lr)
            for LR in self.gen_lr:
                plt.plot(self.gen_lr, label=LR, marker='o')
                plt.xlabel("Epochs")
                plt.ylabel("Learning Rate")
                plt.title("Generator LR")
                plt.legend()
                plt.grid(True)
                plt.subplots_adjust(left=0.2)
                plt.savefig("generator_lr.png")
                plt.close()
            new_disc_lr = max(self.discriminator_lr * (1 - (epoch / self.decay_epochs)), self.min_lr)
            self.model.disc_optimizer.lr.assign(new_disc_lr)
            self.disc_lr.append(new_disc_lr)
            for LR in self.disc_lr:
                plt.plot(self.disc_lr, label=LR, marker='o')
                plt.xlabel("Epochs")
                plt.ylabel("Learning Rate")
                plt.title("Discriminator LR")
                plt.legend()
                plt.grid(True)
                plt.subplots_adjust(left=0.20)
                plt.savefig("discriminator_lr.png")
                plt.close()


# Wasserstein loss for the discriminator
def disc_w_loss(pred_real, pred_fake):
    real_loss = tf.reduce_mean(pred_real)
    fake_loss = tf.reduce_mean(pred_fake)
    return fake_loss - real_loss

# Wasserstein loss for the generator
def gen_w_loss(pred_fake):
    return -tf.reduce_mean(pred_fake)

datagen = ImageDataGenerator(
    preprocessing_function=lambda x: (x / 127.5) - 1, # pixel values are [-1, 1]
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    directory=image_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=True,
    class_mode=None
)

# Generator model
gen_input = tf.keras.Input(shape=noise_dim, name="gen_input")
x = tf.keras.layers.Dense(4*4*512, use_bias=False)(gen_input)
x = tf.keras.layers.Reshape((4, 4, 512))(x)
x = tf.keras.layers.Conv2DTranspose(filters=512, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2DTranspose(filters=256, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2DTranspose(filters=128, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2DTranspose(filters=64, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2DTranspose(filters=32, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)
gen_output = tf.keras.layers.Conv2D(filters=3, strides=2, kernel_size=5, padding="same", use_bias=False,
                           activation="tanh", dtype="float32")(x)
gen_model = tf.keras.Model(inputs=[gen_input], outputs=[gen_output])
print(gen_model.summary())

# Discriminator model
disc_input = tf.keras.Input(shape=input_shape, name="disc_input")
x = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=5, padding="same", use_bias=False)(disc_input)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2D(filters=256, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Conv2D(filters=512, strides=2, kernel_size=5, padding="same", use_bias=False)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
disc_output = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
disc_model = tf.keras.Model(inputs=[disc_input], outputs=[disc_output], name="disc_model")
print(disc_model.summary())

# Configure models' optimizers
gen_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
disc_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)

# Compile both models and fit
gen_model.compile(loss=gen_w_loss, optimizer=gen_opt)
disc_model.compile(loss=disc_w_loss, optimizer=disc_opt)

wgan = WGAN_GP(disc_model, gen_model, noise_dim, discriminator_extra_steps=5)
wgan.compile(disc_opt, gen_opt, disc_w_loss, gen_w_loss)

callback_save_model = SaveModelsCallback(gen_model, disc_model)
plot_losses = PlotLossesCallback()
plot_lr = LRSchedule()
history = wgan.fit(train_generator, epochs=epochs, callbacks=[callback_save_model, plot_losses, plot_lr])


