# CycleGan-voice convert module
import tensorflow as tf
import numpy as np
import os, datetime
import random 

from Utils.networks import discriminator, generator, generator_unet
from Utils.losses import *

class CycleGAN(object) :
    def __init__(self, num_features, g_type = "gated_cnn",discriminator = discriminator ,generator = generator, generator_unet = generator_unet ,mode = 'train', log_dir = './') :
        self.num_features = num_features
        self.input_shape = [None,num_features,None] # batch_size, num_features, num_frames
        self.mode = mode 
        if g_type == "gated_cnn" :
            self.generator = generator # gatedCNN
        else :
            self.generator = generator_unet
            assert g_type == "u_net"
            
        self.discriminator = discriminator
        self.build_model()
        self.optimizer_initializer()
        
        self.saver = tf.train.Saver()  # save checkpoint
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))      
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph()) # Tensorboard
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self) : 

        tf.reset_default_graph()
        

        self.input_A_real = tf.placeholder(tf.float32, shape = self.input_shape)
        self.input_B_real = tf.placeholder(tf.float32, shape = self.input_shape)
    
        self.input_A_fake = tf.placeholder(tf.float32, shape = self.input_shape)  
        self.input_B_fake = tf.placeholder(tf.float32, shape = self.input_shape)
        
        self.input_A_test = tf.placeholder(tf.float32, shape = self.input_shape)
        self.input_B_test = tf.placeholder(tf.float32, shape = self.input_shape)
    
        self.generation_A = self.generator(self.input_B_real, name = "g_B2A") # G : B = > A
        self.generation_B = self.generator(self.input_A_real, name = "g_A2B") # F(inverse of G) : A => B

        self.cycle_A = self.generator(self.generation_B, reuse = True, name = "g_B2A") # F(g_A)
        self.cycle_B = self.generator(self.generation_A, reuse = True, name = "g_A2B") # G(g_B)

        # for identity loss 
        self.identity_A = self.generator(self.input_A_real, reuse = True, name = "g_B2A")
        self.identity_B = self.generator(self.input_B_real, reuse = True, name = "g_A2B")

        # generator loss
        # adversarial loss  
        self.discrimination_A_fake = self.discriminator(self.generation_A, name = "d_A") # discriminator for A 
        self.discrimination_B_fake = self.discriminator(self.generation_B, name = "d_B") # discriminator for B

        self.generator_loss_B2A = l2_loss(tf.ones_like(self.discrimination_A_fake), self.discrimination_A_fake) 
        self.generator_loss_A2B = l2_loss(tf.ones_like(self.discrimination_B_fake), self.discrimination_B_fake)

        # Cycle loss
        self.cycle_loss = l1_loss(self.cycle_A,self.input_A_real) + l1_loss(self.cycle_B, self.input_B_real)
        self.cycle_loss_lambda = tf.placeholder(tf.float32, shape = None, name = "cycle_loss_lambda")

        # Identity loss 
        self.identity_loss = l1_loss(self.identity_A, self.input_A_real) + l1_loss(self.identity_B, self.input_B_real)
        self.identity_loss_lambda = tf.placeholder(tf.float32, None, name = "identity_loss_lambda")

        # Full generator loss 
        self.generator_loss = (self.generator_loss_B2A + 
                               self.generator_loss_A2B + 
                               self.cycle_loss_lambda*self.cycle_loss + 
                               self.identity_loss_lambda * self.identity_loss)

        # Discriminator loss 
        self.discrimination_input_A_real = self.discriminator(self.input_A_real, reuse = True, name = "d_A")
        self.discrimination_input_B_real = self.discriminator(self.input_B_real, reuse = True, name = "d_B")
        self.discrimination_input_A_fake = self.discriminator(self.input_A_fake, reuse = True, name = "d_A")
        self.discrimination_input_B_fake = self.discriminator(self.input_B_fake, reuse = True, name = "d_B")

        self.discriminator_loss_A_real = l2_loss(tf.ones_like(self.discrimination_input_A_real),self.discrimination_input_A_real)
        self.discriminator_loss_A_fake = l2_loss(tf.zeros_like(self.discrimination_input_A_fake),self.discrimination_input_A_fake)
        self.discriminator_loss_B_real = l2_loss(tf.ones_like(self.discrimination_input_B_real),self.discrimination_input_B_real)
        self.discriminator_loss_B_fake = l2_loss(tf.zeros_like(self.discrimination_input_B_fake),self.discrimination_input_B_fake)

        self.discriminator_loss_A = self.discriminator_loss_A_real + self.discriminator_loss_A_fake
        self.discriminator_loss_B = self.discriminator_loss_B_real + self.discriminator_loss_B_fake
        
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B 

        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'd' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'g' in var.name]

    def optimizer_initializer(self) :
        '''linearly decay over next 200000 iter '''
        self.generator_lr = tf.placeholder(tf.float32, shape = None) # default : 0.0002
        self.discriminator_lr = tf.placeholder(tf.float32, shape = None) # default : 0.0001
        self.generator_optimizer = tf.train.AdamOptimizer(self.generator_lr, beta1 = 0.5).minimize(self.generator_loss, var_list = self.generator_vars)
        self.discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_lr, beta1 = 0.5).minimize(self.discriminator_loss, var_list = self.discriminator_vars)
        
    def train(self,input_A,input_B,cycle_lambda,identity_lambda,generator_lr,discriminator_lr) :
        # generator training
        generation_A, generation_B, generator_loss, generator_summaries, _ = self.sess.run(
            [self.generation_A,self.generation_B,self.generator_loss,self.generator_summaries,self.generator_optimizer],
        feed_dict = {self.input_A_real : input_A, self.input_B_real : input_B, 
                     self.cycle_loss_lambda :  cycle_lambda, self.identity_loss_lambda :  identity_lambda,
                    self.generator_lr :  generator_lr})
        self.writer.add_summary(generator_summaries, self.train_step)
        
        # discriminator training
        discriminator_loss, _, discriminator_summaries = self.sess.run(
            [self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries],
        feed_dict = {self.input_A_real : input_A, self.input_B_real : input_B, 
                    self.input_A_fake : generation_A, self.input_B_fake : generation_B, 
                    self.discriminator_lr : discriminator_lr})
        self.writer.add_summary(discriminator_summaries,self.train_step)
        
        self.train_step += 1 
        return generator_loss, discriminator_loss
        
    def test(self, inputs, direction) :
        # Test A2B 
        if direction == "A2B" :
            generation = self.sess.run(self.generation_B,feed_dict={self.input_A_real:inputs}) # generate B
        elif direction == "B2A" :
            generation = self.sess.run(self.generation_A,feed_dict={self.input_B_real:inputs}) # generate A 
        else : 
            assert (direction in ["A2B","B2A"])
        
        return generation
    
    def save(self, directory, filename, epoch) :
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename), global_step=epoch)

        return os.path.join(directory, filename)

    def load(self, filepath):
        print('='*30, self.mode, '='*30)
        ckpt = tf.train.get_checkpoint_state(filepath) # checkpoint load
        print('ckpt!!!', ckpt)
        if ckpt:
            print('Loading Previous Checkpoint', ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path
                              .split('-')[1]
                              .split('.')[0])
            print("  Global step was: {}".format(global_step))
            print("  Restoring...", end="")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" Model Load Done.")
            return global_step
        else:
            print('No CheckPoint, Train을 처음부터 하겠습니다!')
            return None
        
    def summary(self) :
        with tf.name_scope("generator_summaries") :
            generator_loss_B2A_summary = tf.summary.scalar("generator_loss_B2A",self.generator_loss_B2A)
            generator_loss_A2B_summary = tf.summary.scalar("generator_loss_A2B",self.generator_loss_A2B)
            generator_loss_summary = tf.summary.scalar("generator_loss" , self.generator_loss)
            generator_summaries = tf.summary.merge([generator_loss_B2A_summary,generator_loss_A2B_summary,generator_loss_summary])
            
        with tf.name_scope("discriminator_summaries") :
            discriminator_loss_A_summary = tf.summary.scalar("discriminator_loss_A", self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar("discriminator_loss_B", self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar("discriminator_loss", self.discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_A_summary,discriminator_loss_B_summary,discriminator_loss_summary])
            
        return generator_summaries, discriminator_summaries
    
if __name__ == "__main__" :
    # the speech data were 24 mel - cepstral coefficients ( 24 MCEPS)
    model = CycleGAN(num_features=24) 
    
