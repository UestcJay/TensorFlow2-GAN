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
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
BUFFER_SIZE=train_images.shape[0]
train_images= (train_images - 127.5) / 127.5
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_labels=tf.one_hot(train_labels,depth=10)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(opt.batch_size,drop_remainder=True)
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, opt.latent_dim])
# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# define discriminator
def make_discriminaor(input_shape):
    return tf.keras.Sequential([
        # layers.Input(),
        layers.Flatten(),
        layers.Dense(256,  activation=None, input_shape=input_shape),
        layers.LeakyReLU(0.2),
        layers.Dense(256,  activation=None),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')
    ])


# define generator
def make_generator(input_shape):
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='tanh'),
        layers.Reshape(img_shape)
    ])



def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


# Initialize generator and discriminator
generator = make_generator((opt.latent_dim + 10,))
discriminator = make_discriminaor((28*28*11,))


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

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    y = tf.reshape(y, [-1, 1, 1, y_shapes[1]])
    y_shapes = tf.shape(y)
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

@tf.function
def train_step(batch_images,batch_labels):
    z = get_random_z(opt.latent_dim, batch_images.shape[0])
    z_y = tf.concat([z, batch_labels], 1)
    real_images = conv_cond_concat(batch_images, batch_labels)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      fake_imgs = generator(z_y, training=True)
      fake_imgs = conv_cond_concat(fake_imgs, batch_labels)

      real_output = discriminator(real_images, training=True)
      fake_output = discriminator(fake_imgs, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# ----------
#  Training
# ----------
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()


        for batch_idx, (batch_images, batch_labels) in enumerate(dataset):
            g_loss, d_loss = train_step(batch_images, batch_labels)
            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            total_loss_metrics(g_loss + d_loss)

            template = '[Epoch{}/{}], Batch[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(epoch, epochs, batch_idx, len(dataset), d_loss_metrics.result(),
                                  g_loss_metrics.result(), total_loss_metrics.result()))
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()

        # Produce images for the GIF as we go
        generate_and_save_images(generator,epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % opt.freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # # Generate after the final epoch
    generate_and_save_images(generator,epochs,seed)
# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
if __name__ == "__main__":
    train(train_dataset,opt.n_epochs)
    # x = tf.random.uniform([2, 28,28,1], minval=-1, maxval=1)
    # y = tf.random.uniform([2, 10], minval=-1, maxval=1)
    # c = conv_cond_concat(x, y)
    # print(c.shape)

