import tensorflow as tf
from Utils.layers import *

def generator(inputs, reuse=False, name= "generator") :
    
    inputs = tf.transpose(inputs, perm= [0,2,1])
    
    with tf.variable_scope(name) as scope :
        
        if reuse : 
            scope.reuse_variables()
        else : 
            assert scope.reuse is False
            
        conv1 = conv1d_layer(inputs, 128, 15, 1)
        conv1_gates = conv1d_layer(inputs, 128, 15, 1)
        conv1_glu = gated_linear_unit(conv1, conv1_gates)
    
        down1 = downsample_1d(conv1_glu, 256, 5, 2)
        down2 = downsample_1d(down1, 512, 5, 2)
    
        res_block1 = residual_block(down2, 1024, 3, 1)
        res_block2 = residual_block(res_block1, 1024, 3, 1)
        res_block3 = residual_block(res_block2, 1024, 3, 1)
        res_block4 = residual_block(res_block3, 1024, 3, 1)
        res_block5 = residual_block(res_block4, 1024, 3, 1)
        res_block6 = residual_block(res_block5, 1024, 3, 1)
    
        up1 = upsample_1d(res_block6, 1024, 5, 1)
        up2 = upsample_1d(up1, 512, 5, 1)
    
        conv2 = conv1d_layer(up2, 24, 15, 1)
        outputs = tf.transpose(conv2, perm = [0,2,1])
    
    return outputs

def discriminator(inputs, reuse=False, name="discriminator") :
    ''' 
    inputs shape : [batch_size, num_features, time] => [batch_size, num_features, time, 1]
    '''
    inputs = tf.expand_dims(inputs, -1)
    
    with tf.variable_scope(name) as scope :
        
        if reuse :
            scope.reuse_variables()
        else : 
            assert scope.reuse is False 
        
        conv1 = conv2d_layer(inputs, 128, [3,3],[1,2])
        conv1_gates = conv2d_layer(inputs, 128, [3,3], [1,2])
        conv1_glu = gated_linear_unit(conv1, conv1_gates)
    
        down1 = downsample_2d(conv1_glu, filters=256, kernel_size=[3,3],strides=[2,2])
        down2 = downsample_2d(down1, filters=512, kernel_size=[3,3], strides=[2,2])
        down3 = downsample_2d(down2, filters=1024, kernel_size=[6,3], strides=[1,2])
    
    #fully connected layer
        outputs = tf.layers.dense(down3, 1, activation=tf.nn.sigmoid)
    
    return outputs


def generator_unet(inputs, reuse=False, name = "generator") :

    inputs = tf.transpose(inputs, perm= [0,2,1])
    
    with tf.variable_scope(name) as scope :
        
        if reuse : 
            scope.reuse_variables()
        else : 
            assert scope.reuse is False  
        
        conv1 = conv1d_layer(inputs, 128, 15, 1)

        #downsampling
        down1 = conv_res_conv(conv1, 256, 5, 1)
        down2 = conv_res_conv(down1, 512, 5, 1)
        down3 = conv_res_conv(down2, 1024, 3, 1)
        
        bridge = conv1d_layer(down3, 1024, 1, 1)
        
        #upsampling
        up1 = conv_res_conv(bridge, 1024, 3, 1)
        up2 = conv_res_conv(skip_connection(up1, down3), 512, 3, 1)
        up3 = conv_res_conv(skip_connection(up2, down2), 256, 5, 1)
        
        conv2 = conv1d_layer(skip_connection(up3, down1), 128, 5, 1)
        conv3 = conv1d_layer(skip_connection(up4,conv1), 24, 15, 1)
        
        outputs = tf.transpose(conv2, perm = [0,2,1])
        
    return outputs
