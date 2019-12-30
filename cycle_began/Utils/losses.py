import tensorflow as tf

def l1_loss (y, y_hat) : 
    # cycle_consistency_loss, identity_loss 
    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss (y, y_hat) :
    # adversarial loss (LSGAN)
    return tf.reduce_mean(tf.square(y - y_hat))
