import argparse
import tensorflow as tf
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow.keras.layers as layers
import time



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--freq", type=int, default=1, help="number of epochs of saving")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
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
# BEGAN Parameter

variable_k = tf.Variable(0.0,trainable=False)
gamma = 0.75
lamda = 0.001


# data load & preprocessing
(train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_x= train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
BUFFER_SIZE=train_x.shape[0]
train_x = (train_x - 127.5) / 127.5
train_ds = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER_SIZE).batch(opt.batch_size)
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, opt.latent_dim])



# define discriminator
class Discriminator(tf.keras.Model):
    def __init__(self, batch_size=64, is_training=True):
        super(Discriminator, self).__init__(name='discriminator')
        self.batch_size = batch_size
        self.is_training = is_training
        self.bn_1 = layers.BatchNormalization(trainable=self.is_training)
        self.bn_2 = layers.BatchNormalization(trainable=self.is_training)
        self.fc_1 = layers.Dense(32)
        self.fc_2 = layers.Dense(64*14*14)
        self.conv_1 = layers.Conv2D(64, 4,strides=2, padding='same')
        self.up_conv_1 = layers.Conv2DTranspose(1, 4, 2,padding='same')

    def call(self, inputs, training):
        x = self.conv_1(inputs)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        x = self.fc_1(x)
        x = self.bn_1(x, training)
        code = layers.ReLU()(x)
        x = self.fc_2(code)
        x = self.bn_2(x, training)
        x = layers.ReLU()(x)
        x = layers.Reshape((14, 14, 64))(x)
        x = self.up_conv_1(x)
        out = tf.keras.activations.sigmoid(x)
        recon_error = tf.math.sqrt(2 * tf.nn.l2_loss(out - inputs)) / self.batch_size
        return out, recon_error, code


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
        x = tf.keras.activations.sigmoid(x)
        return x



def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()



generator_optimizer = tf.keras.optimizers.Adam(opt.lr, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(opt.lr, beta_1=0.5, beta_2=0.999)
# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images, variable_k):

    noise = get_random_z(opt.latent_dim, images.shape[0])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      D_real_img, D_real_err, D_real_code = discriminator(images, training=True)
      fake_imgs = generator(noise, training=True)
      D_fake_img, D_fake_err, D_fake_code = discriminator(fake_imgs, training=True)
      d_loss = D_real_err - variable_k * D_fake_err

      # get loss for generator
      g_loss = D_fake_err

      # convergence metric
      M = D_real_err + tf.math.abs(gamma * D_real_err - D_fake_err)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    variable_k  = tf.clip_by_value(variable_k + lamda * (gamma * D_real_err - D_fake_err), 0, 1)
    return g_loss, d_loss, variable_k

# ----------
#  Training
# ----------
def train(dataset, epochs):
    global variable_k
    for epoch in range(epochs):
        start = time.time()


        for batch_idx, image_batch in enumerate(dataset):
            g_loss, d_loss, variable_k = train_step(image_batch, variable_k)
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
    train(train_ds,opt.n_epochs)

