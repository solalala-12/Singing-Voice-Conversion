import os
import tensorflow as tf
from datetime import datetime

from Utils.networks import discriminator_cbgan, generator
from Utils.losses import *

class CycleBeGAN(object):
    def __init__(self, num_features, discriminator=discriminator_cbgan, generator=generator, mode='train',
                 log_dir='./log'):

        self.num_features = num_features
        self.input_shape = [None, num_features, None]  # [batch_size, num_features, num_frames]

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver() #save checkpoint
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_real')

        self.input_A_fake = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_fake')

        self.input_A_test = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_test')
        self.input_B_test = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_test')

        self.generation_B = self.generator(inputs=self.input_A_real, reuse=False, name='generator_A2B')
        self.cycle_A = self.generator(inputs=self.generation_B, reuse=False, name='generator_B2A')

        self.generation_A = self.generator(inputs=self.input_B_real, reuse=True, name='generator_B2A')
        self.cycle_B = self.generator(inputs=self.generation_A, reuse=True, name='generator_A2B')

        self.generation_A_identity = self.generator(inputs=self.input_A_real, reuse=True, name='generator_B2A')
        self.generation_B_identity = self.generator(inputs=self.input_B_real, reuse=True, name='generator_A2B')

        self.discrimination_A_fake = self.discriminator(inputs=self.generation_A, reuse=False,
                                                        name='discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs=self.generation_B, reuse=False,
                                                        name='discriminator_B')

        self.cycle_loss = l1_loss(y=self.input_A_real, y_hat=self.cycle_A) + l1_loss(y=self.input_B_real,
                                                                                     y_hat=self.cycle_B)

        self.identity_loss = l1_loss(y=self.input_A_real, y_hat=self.generation_A_identity) + l1_loss(
            y=self.input_B_real, y_hat=self.generation_B_identity)

        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')

        self.generator_loss_B2A = l1_loss(y=self.discrimination_A_fake, y_hat=self.generation_A)
        self.generator_loss_A2B = l1_loss(y=self.discrimination_B_fake, y_hat=self.generation_B)

        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss

        self.discrimination_input_A_real = self.discriminator(inputs=self.input_A_real, reuse=True,
                                                              name='discriminator_A')
        self.discrimination_input_B_real = self.discriminator(inputs=self.input_B_real, reuse=True,
                                                              name='discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(inputs=self.generation_A, reuse=True,
                                                              name='discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(inputs=self.generation_B, reuse=True,
                                                              name='discriminator_B')

        self.k_t_A = tf.placeholder(tf.float32, None, name='k_t_A')
        self.k_t_B = tf.placeholder(tf.float32, None, name='k_t_B')
        self.gamma_A = tf.placeholder(tf.float32, None, name='gamma_A')
        self.gamma_B = tf.placeholder(tf.float32, None, name='gamma_B')
        self.lambda_k_A = tf.placeholder(tf.float32, None, name='lambda_k_A')
        self.lambda_k_B = tf.placeholder(tf.float32, None, name='lambda_k_B')


        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = l1_loss(y=self.discrimination_input_A_real,
                                                       y_hat=self.input_A_real)
        self.discriminator_loss_input_A_fake = l1_loss(y=self.discrimination_input_A_fake,
                                                       y_hat=self.generation_A)

        self.discriminator_loss_A = self.discriminator_loss_input_A_real - (self.k_t_A * self.discriminator_loss_input_A_fake)


        self.discriminator_loss_input_B_real = l1_loss(y=self.discrimination_input_B_real,
                                                       y_hat=self.input_B_real)
        self.discriminator_loss_input_B_fake = l1_loss(y=self.discrimination_input_B_fake,
                                                       y_hat=self.generation_B)

        self.discriminator_loss_B = self.discriminator_loss_input_B_real - (self.k_t_B * self.discriminator_loss_input_B_fake)


        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B


        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]

        # Reserved for test
        self.generation_B_test = self.generator(inputs=self.input_A_test, reuse=True, name='generator_A2B')
        self.generation_A_test = self.generator(inputs=self.input_B_test, reuse=True, name='generator_B2A')

    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')


        self.balance_A = self.gamma_A * self.discriminator_loss_A - self.generator_loss_B2A
        self.balance_B = self.gamma_B * self.discriminator_loss_B - self.generator_loss_A2B


        self.measure_A = self.discriminator_loss_A + tf.abs(self.balance_A)
        self.measure_B = self.discriminator_loss_B + tf.abs(self.balance_B)

        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate,
                                                          beta1=0.5).minimize(self.generator_loss,
                                                                              var_list=self.generator_vars)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate,
                                                              beta1=0.5).minimize(self.discriminator_loss,
                                                                                  var_list=self.discriminator_vars)



    def train(self, input_A, input_B, lambda_cycle, lambda_identity, gamma_A, gamma_B, lambda_k_A, lambda_k_B,
              generator_learning_rate, discriminator_learning_rate, k_t_A, k_t_B):

        generation_A, generation_B, = self.sess.run([self.generation_A, self.generation_B], \
                                                    feed_dict={self.input_A_real: input_A, self.input_B_real: input_B})
        #
        fetch_dict = {self.gamma_A: gamma_A, self.gamma_B: gamma_B, self.lambda_k_A: lambda_k_A,
                     self.lambda_k_B: lambda_k_B, self.lambda_cycle: lambda_cycle,
                     self.lambda_identity: lambda_identity, self.input_A_real: input_A, self.input_B_real: input_B,
                     self.input_A_fake: generation_A,
                     self.input_B_fake: generation_B,
                     self.generator_learning_rate: generator_learning_rate,
                     self.discriminator_learning_rate: discriminator_learning_rate,
                     self.k_t_A: k_t_A, self.k_t_B: k_t_B}

        generator_loss, _1, generator_summaries, discriminator_loss, _2, discriminator_summaries, measure_A, measure_B, k_t_A, k_t_B, balance_A, balance_B = self.sess.run(
            [self.generator_loss, self.generator_optimizer,
             self.generator_summaries, self.discriminator_loss, self.discriminator_optimizer,
             self.discriminator_summaries, self.measure_A, self.measure_B, self.k_t_A, self.k_t_B, self.balance_A, self.balance_B], \
            feed_dict = fetch_dict)

        self.writer.add_summary(generator_summaries, self.train_step)
        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss, measure_A, measure_B, k_t_A, k_t_B, balance_A, balance_B

    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict={self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict={self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation

#     def save(self, directory, filename):

#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         self.saver.save(self.sess, os.path.join(directory, filename))

#         return os.path.join(directory, filename)

#     def load(self, filepath):

#         self.saver.restore(self.sess, filepath)

    def save(self, directory, filename, epoch) :
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename), global_step=epoch)

        return os.path.join(directory, filename)
    
    def load(self, filepath):
            ckpt = tf.train.get_checkpoint_state(filepath) # checkpoint load
            if ckpt:
                print('Loading Previous Checkpoint', ckpt.model_checkpoint_path)
                global_step = int(ckpt.model_checkpoint_path
                                  .split('-')[1]
                                  .split('.')[0])
                print("  Global step was: {}".format(global_step))
                print("  Restoring...", end="")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print(" Done.")
                return global_step
            else:
                print('No CheckPoint, Train을 처음부터 하겠습니다!')
                return None

    def summary(self):
        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge(
                [cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary,
                 generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            k_t_A = tf.summary.scalar('k_t_A', self.k_t_A)
            k_t_B = tf.summary.scalar('k_t_B', self.k_t_B)
            balance_A = tf.summary.scalar('balance_A', self.balance_A)
            balance_B = tf.summary.scalar('balance_B', self.balance_B)
            measure_A = tf.summary.scalar('measure_A', self.measure_A)
            measure_B = tf.summary.scalar('measure_B', self.measure_B)

            discriminator_summaries = tf.summary.merge(
                [discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary
                    , k_t_A, k_t_B, balance_A, balance_B, measure_A, measure_B])

        return generator_summaries, discriminator_summaries
