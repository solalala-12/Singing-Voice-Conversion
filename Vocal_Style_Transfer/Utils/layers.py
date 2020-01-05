import tensorflow as tf

def conv1d_layer(inputs, filters, kernel_size, strides=1, padding="same", activation=None) :
    conv1d =  tf.layers.Conv1D(filters=filters, 
                            kernel_size=kernel_size, 
                            strides=strides, 
                            padding=padding, 
                            activation=activation)(inputs)
    return conv1d

# https://blog.lunit.io/2018/04/12/group-normalization/  IN 설명
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


# Cycle BeGAN

def gated_linear_layer(inputs, gates, name = None):
    
    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)
    
    return activation

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 2 ,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size = 2, name = None):
    
    if shuffle_size == 2:
        n = tf.shape(inputs)[0]
        w = tf.shape(inputs)[1]
        c = inputs.get_shape().as_list()[2]

        oc = c // shuffle_size
        ow = w * shuffle_size
        outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)
    else:
        outputs = inputs

    return outputs

def generator_gatedcnn(inputs, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')
    
    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        #[batch_size, num_features, time] 1, 24, 128
        
        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # Upsample
        u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2
    

def discriminator(inputs, reuse = False, scope_name = 'discriminator'):

    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose_dic')

    with tf.variable_scope(scope_name) as scope:

        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')


        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')


        u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

        return o2
    
# U-net 

def skip_connection(input_1,input_2) :
    return tf.add(input_1,input_2)

def conv1d_with_norm_act(inputs,filters, kernel_size, strides, padding="same") :
    conv1d = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    conv1_norm_act = instance_norm(inputs=conv1d, activation=tf.nn.leaky_relu())
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
