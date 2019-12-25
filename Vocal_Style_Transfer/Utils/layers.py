import tensorflow as tf

def conv1d_layer(inputs, filters, kernel_size, strides=1, padding="same", activation=None) :
    conv1d =  tf.layers.Conv1D(filters=filters, 
                            kernel_size=kernel_size, 
                            strides=strides, 
                            padding=padding, 
                            activation=activation)(inputs)
    return conv1d

def instance_norm(inputs, activation=None) :
    instance_norm = tf.contrib.layers.instance_norm(
                                    inputs=inputs,
                                    activation_fn= activation)
    return instance_norm

def gated_linear_unit(inputs,gates) :
    glu = tf.multiply(inputs,tf.sigmoid(gates))
    return glu

def downsample_1d(inputs, filters, kernel_size, strides) :
    conv = conv1d_layer(inputs,filters,kernel_size,strides=strides)
    conv_norm = instance_norm(conv)
    
    gates = conv1d_layer(inputs,filters,kernel_size,strides=strides)
    gates_norm = instance_norm(gates)
    
    glu = gated_linear_unit(conv_norm,gates_norm)
    return glu

def residual_block(inputs,filters,kernel_size,strides) :
    
    conv1_glu = downsample_1d(inputs, filters, kernel_size, strides)
    
    conv2 = conv1d_layer(conv1_glu, filters // 2, kernel_size, strides)
    conv2_norm = instance_norm(conv2)
    
    conv_sum = tf.add(inputs,conv2_norm)
    return conv_sum

# effective for high-resolution image generation
def pixel_shuffler(inputs,shuffle_size=2) :
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc])

    return outputs

def upsample_1d(inputs, filters, kernel_size, strides) :
    conv1 = conv1d_layer(inputs, filters, kernel_size, strides)
    conv1_pixel_shuffle = pixel_shuffler(conv1)
    conv1_norm = instance_norm(conv1_pixel_shuffle)
    
    gates = conv1d_layer(inputs, filters, kernel_size, strides)
    gates_pixel_shuffle = pixel_shuffler(conv1)
    gates_norm = instance_norm(gates_pixel_shuffle)
    
    glu = gated_linear_unit(conv1_norm,gates_norm)
    return glu

def downsample_2d(inputs, filters, kernel_size, strides ) : 
    conv = conv2d_layer(inputs, filters, kernel_size, strides)
    conv_norm = instance_norm(conv)
    
    conv_gates = conv2d_layer(inputs, filters, kernel_size, strides)
    conv_gates_norm = instance_norm(conv_gates)
    
    glu = gated_linear_unit(conv_norm, conv_gates_norm)
    
    return glu

def conv2d_layer(inputs, filters, kernel_size, strides, activation =None) :
    conv = tf.layers.Conv2D(filters = filters, 
                            kernel_size=kernel_size, 
                            strides=strides,
                            padding = "same",
                           activation = activation)(inputs)
    return conv

# U-net 

def skip_connection(input_1,input_2) :
    return tf.add(input_1,input_2)

def conv1d_with_norm_act(inputs,filters, kernel_size, strides, padding="same") :
    conv1d = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    conv1_norm_act = instance_norm(inputs=conv1d, activation=tf.nn.leaky_relu)
    return conv1_norm_act

def conv1d_with_3_layers(inputs, filters, kernel_size, strides, repeat = 3) :
    for i in range(repeat) :
        conv = conv1d_with_norm_act(inputs=inputs,filters=filters,kernel_size=kernel_size,strides=strides)
        inputs= conv
    return conv

def res_block(inputs, filters, kernel_size, strides) :
    conv = conv1d_with_3_layers(inputs=inputs,filters=filters,kernel_size=kernel_size,strides=strides)
    return skip_connection(inputs,conv)

def conv_res_conv(inputs, filters, kernel_size, strides) :
    conv1 = conv1d_with_norm_act(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    res = res_block(inputs=conv1, filters=filters, kernel_size=kernel_size, strides=strides)
    conv2 = conv1d_with_norm_act(inputs=res, filters=filters, kernel_size=kernel_size, strides=strides)
    return conv2
