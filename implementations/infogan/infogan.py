import argparse
import os
import numpy as np
import math

import PIL
import time
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
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

img_shape = (opt.img_size, opt.img_size, opt.channels)
len_discrete_code = 10  # categorical distribution (i.e. label)
len_continuous_code = 2 # gaussian distribution (e.g. rotation, thickness)


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

# define discriminator
class Discriminator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Discriminator, self).__init__(name='discriminator')
        self.is_training = is_training
        self.conv_1 = layers.Conv2D(64, 4,strides=2, padding='same')
        self.conv_2 = layers.Conv2D(128, 4,strides=2, padding='same')
        self.bn_1 = layers.BatchNormalization(trainable=self.is_training)
        self.bn_2 = layers.BatchNormalization(trainable=self.is_training)
        self.fc_1 = layers.Dense(1024)
        self.fc_2 = layers.Dense(1)

    def call(self, inputs, training):
        x = self.conv_1(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = self.conv_2(x)
        x = self.bn_1(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        x = self.fc_1(x)
        x = self.bn_2(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        out_logits = self.fc_2(x)
        out = keras.activations.sigmoid(out_logits)
        return out, out_logits, x


class Generator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Generator, self).__init__(name='generator')
        self.is_training = is_training
        self.fc_1 = layers.Dense(1024)
        self.fc_2 = layers.Dense(128*7*7)
        self.bn_1 = layers.BatchNormalization(trainable=self.is_training)
        self.bn_2 = layers.BatchNormalization(trainable=self.is_training)
        self.bn_3 = layers.BatchNormalization(trainable=self.is_training)
        self.up_conv_1 = layers.Conv2DTranspose(64, 4, 2,padding='same')
        self.up_conv_2 = layers.Conv2DTranspose(1, 4, 2,padding='same')

    def call(self, inputs, training):
        x = self.fc_1(inputs)
        x = self.bn_1(x, training)
        x = layers.ReLU()(x)
        x = self.fc_2(x)
        x = self.bn_2(x, training)
        x = layers.ReLU()(x)
        x = layers.Reshape((7, 7, 128))(x)
        x = self.up_conv_1(x)
        x = self.bn_3(x, training)
        x = layers.ReLU()(x)
        x = self.up_conv_2(x)
        x = keras.activations.sigmoid(x)
        return x


class Classifier(tf.keras.Model):
    def __init__(self, y_dim, is_training=True):
        super(Classifier, self).__init__(name='classifier')
        self.is_training = is_training
        self.y_dim = y_dim
        self.fc_1 = layers.Dense(64)
        self.fc_2 = layers.Dense(self.y_dim)
        self.bn_1 = layers.BatchNormalization(trainable=self.is_training)

    def call(self, inputs, training):
        x = self.fc_1(inputs)
        x = self.bn_1(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        out_logits = self.fc_2(x)
        out=keras.layers.Softmax()(out_logits)
        return out, out_logits



def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


# Initialize generator and discriminator and classifier
g = Generator()
d = Discriminator()
c = Classifier(12)

def generator_loss(fake_output):
    return  tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output))

def discriminator_loss(real_output, fake_output):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
    total_loss = d_loss_fake + d_loss_real
    return total_loss


def q_loss_fun(disc_code_est, disc_code_tg, cont_code_est, cont_code_tg):
    q_disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=disc_code_tg, logits=disc_code_est))
    q_cont_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(cont_code_tg - cont_code_est), axis=1))
    q_loss = q_disc_loss + q_cont_loss
    return q_loss


g_optimizer = keras.optimizers.Adam(lr=5 * opt.lr, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(lr=opt.lr, beta_1=0.5)
q_optimizer = keras.optimizers.Adam(lr=5 * opt.lr, beta_1=0.5)
# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
q_loss_metrics = tf.keras.metrics.Mean('q_loss', dtype=tf.float32)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=g_optimizer,
                                              discriminator_optimizer=d_optimizer,
                                              classifier_optimizer=q_optimizer,
                                              generator=g,
                                              discriminator=d,
                                              classifier=c)
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
    code = get_random_z(len_continuous_code, batch_images.shape[0])
    batch_codes = tf.concat((batch_labels, code), axis=1)
    batch_z = tf.concat([z, batch_codes], 1)
    real_images = conv_cond_concat(batch_images, batch_codes)

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as q_tape:
      fake_imgs = g(batch_z, training=True)
      fake_imgs = conv_cond_concat(fake_imgs, batch_codes)

      d_fake, d_fake_logits, input4classifier_fake = d(fake_imgs, training=True)
      d_real, d_real_logits, input4classifier_real = d(real_images, training=True)

      g_loss = generator_loss(d_fake_logits)
      d_loss = discriminator_loss(d_real_logits, d_fake_logits)

      code_fake, code_logit_fake = c(input4classifier_fake, training=True)

      disc_code_est = code_logit_fake[:, :len_discrete_code]
      disc_code_tg = batch_codes[:, :len_discrete_code]
      cont_code_est = code_logit_fake[:, len_discrete_code:]
      cont_code_tg = batch_codes[:, len_discrete_code:]
      q_loss = q_loss_fun(disc_code_est, disc_code_tg, cont_code_est, cont_code_tg)

    gradients_of_d = d_tape.gradient(d_loss, d.trainable_variables)
    gradients_of_g = g_tape.gradient(g_loss, g.trainable_variables)
    gradients_of_q = q_tape.gradient(q_loss, c.trainable_variables)



    d_optimizer.apply_gradients(zip(gradients_of_d, d.trainable_variables))
    g_optimizer.apply_gradients(zip(gradients_of_g, g.trainable_variables))
    q_optimizer.apply_gradients(zip(gradients_of_q, c.trainable_variables))
    return g_loss, d_loss, q_loss

# ----------
#  Training
# ----------
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()


        for batch_idx, (batch_images, batch_labels) in enumerate(dataset):
            g_loss, d_loss, q_loss = train_step(batch_images, batch_labels)
            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            q_loss_metrics(q_loss)

            template = '[Epoch{}/{}], Batch[{}/{}] D_loss={:.5f} G_loss={:.5f} q_loss={:.5f}'
            print(template.format(epoch, epochs, batch_idx, len(dataset), d_loss_metrics.result(),
                                  g_loss_metrics.result(), q_loss_metrics.result()))
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            q_loss_metrics.reset_states()

        # Produce images for the GIF as we go
        generate_and_save_images(g,epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % opt.freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # # Generate after the final epoch
    generate_and_save_images(g,epochs,seed)


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

