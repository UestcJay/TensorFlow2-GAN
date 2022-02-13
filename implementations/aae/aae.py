import argparse
import os
import numpy as np
import math

import PIL
import time
import tensorflow.keras.layers as layers
import tensorflow as tf
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--freq", type=int, default=1, help="number of epochs of saving")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--buffer_size", type=int, default=60000, help="size of the buffers")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.img_size, opt.img_size,opt.channels)



# data load & preprocessing
(train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
BUFFER_SIZE=train_x.shape[0]
train_x = (train_x - 127.5) / 127.5
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
train_ds = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER_SIZE).batch(opt.batch_size, drop_remainder=True)
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, opt.latent_dim])
# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
# define discriminator
def make_discriminaor():
    return tf.keras.Sequential([
        layers.Dense(512,  activation=None, input_shape=(opt.latent_dim,)),
        layers.LeakyReLU(0.2),
        layers.Dense(256,  activation=None),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

class AAE(tf.keras.Model):
  def __init__(self):
    super(AAE, self).__init__()
    self.encoder= tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(opt.latent_dim + opt.latent_dim),
      ]
    )

    self.decoder = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(opt.latent_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='tanh'),
        layers.Reshape(img_shape)
    ])

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    img = self.decoder(z)
    return img



def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


# Initialize generator and discriminator
generator = AAE()
discriminator = make_discriminaor()


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(opt.lr)
discriminator_optimizer = tf.keras.optimizers.Adam(opt.lr)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = get_random_z(opt.latent_dim, images.shape[0])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        mean, logvar = generator.encode(images)
        encoded_imgs = generator.reparameterize(mean, logvar)
        decoded_imgs = generator.decode(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * generator_loss(discriminator(encoded_imgs)) + 0.999 * pixelwise_loss(images,decoded_imgs)
        real_output = discriminator(encoded_imgs, training=True)
        fake_output = discriminator(noise, training=True)

        d_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return g_loss, d_loss

# ----------
#  Training
# ----------
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()


        for batch_idx, image_batch in enumerate(dataset):
            g_loss, d_loss = train_step(image_batch)
            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            total_loss_metrics(g_loss + d_loss)

            template = '[Epoch{}/{}], Batch[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(epoch, epochs, batch_idx, len(dataset), d_loss_metrics.result(),
                                  g_loss_metrics.result(), total_loss_metrics.result()))
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()


        # Save the model every 15 epochs
        if (epoch + 1) % opt.freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

if __name__ == "__main__":
    train(train_ds,opt.n_epochs)

