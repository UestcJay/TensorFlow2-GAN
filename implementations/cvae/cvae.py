import argparse
import os
import numpy as np
import math

import PIL
import time
import tensorflow.keras as keras
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
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
BUFFER_SIZE=train_images.shape[0]
train_images= train_images/ 255.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_labels=tf.one_hot(train_labels,depth=10)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(opt.batch_size,drop_remainder=True)
num_examples_to_generate = 16

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    y = tf.reshape(y, [-1, 1, 1, y_shapes[1]])
    y_shapes = tf.shape(y)
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def make_encoder():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 4,strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, 4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(2*opt.latent_dim))
    return model


def make_decoder():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(128*7*7))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(64, 4, 2,padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(1, 4, 2,padding='same',activation='tanh'))
    return model



encoder = make_encoder()
decoder = make_decoder()

optimizer = keras.optimizers.Adam(lr=opt.lr, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              optimizer=optimizer,
                                              encoder=encoder,
                                              decoder=decoder)
# metrics setting
nll_loss_metric = tf.keras.metrics.Mean('nll_loss', dtype=tf.float32)
kl_loss_metric = tf.keras.metrics.Mean('kl_loss', dtype=tf.float32)
total_loss_metric = tf.keras.metrics.Mean('totol_loss', dtype=tf.float32)
@tf.function
def train_step(batch_images, batch_labels):
    with tf.GradientTape() as gradient_tape:
      batch_images_y = conv_cond_concat(batch_images, batch_labels)
      gaussian_params = encoder(batch_images_y, training=True)
      mu = gaussian_params[:, :opt.latent_dim]
      sigma = 1e-6 + tf.keras.activations.softplus(gaussian_params[:, opt.latent_dim:])
      z = mu + sigma * tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
      z_y = tf.concat([z, batch_labels], 1)

      out = decoder(z_y, training=True)
      out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)
      marginal_likelihood = tf.reduce_sum(batch_images * tf.math.log(out) + (1. - batch_images) * tf.math.log(1. - out),
                                          [1, 2])
      KL_divergence = 0.5 * tf.reduce_sum(
          tf.math.square(mu) + tf.math.square(sigma) - tf.math.log(1e-8 + tf.math.square(sigma)) - 1, [1])
      neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
      KL_divergence = tf.reduce_mean(KL_divergence)
      ELBO = neg_loglikelihood - KL_divergence
      loss = -ELBO

    trainable_variables = decoder.trainable_variables + encoder.trainable_variables
    gradients = gradient_tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return neg_loglikelihood, KL_divergence, loss

# ----------
#  Training
# ----------
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()


        for batch_idx, (batch_images, batch_labels)  in enumerate(dataset):
            neg_loglikelihood, KL_divergence, loss = train_step(batch_images, batch_labels)
            nll_loss_metric(neg_loglikelihood)
            kl_loss_metric(KL_divergence)
            total_loss_metric(loss)

            template = '[Epoch{}/{}], Batch[{}/{}] nll_loss={:.5f} kl_loss={:.5f} Total_loss={:.5f}'
            print(template.format(epoch, epochs, batch_idx, len(dataset), nll_loss_metric.result(),
                                  kl_loss_metric.result(), total_loss_metric.result()))
            nll_loss_metric.reset_states()
            kl_loss_metric.reset_states()
            total_loss_metric.reset_states()


        # Save the model every 15 epochs
        if (epoch + 1) % opt.freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))




if __name__ == "__main__":
    train(train_dataset,opt.n_epochs)

