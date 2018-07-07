import tensorflow as tf
import ops


def c5s1k64(ipt, name='c5s1k64', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 64, 5, 2, norm='batch', activation=ops.leaky_relu,
                          name=name, reuse=reuse, is_training=is_training)


def c3s1k64(ipt, name='c3s1k64', reuse=False, is_training=True):
    with tf.variable_scope(name):
        return ops.conv2d(ipt, 64, 3, 1, norm='batch', activation=ops.leaky_relu,
                          name=name, reuse=reuse, is_training=is_training)


def residual_block(ipt, i, name='resblock', reuse=False, is_training=True):
    with tf.variable_scope(name + str(i)):
        c3s1k64 = ops.conv2d(ipt, 64, 3, 1, 1, norm='batch', activation=ops.leaky_relu,
                             reuse=reuse, is_training=is_training, name='c3s1k64_1')
        c3s1k64 = ops.conv2d(c3s1k64, 64, 3, 1, 1, norm='batch', activation=ops.leaky_relu,
                             reuse=reuse, is_training=is_training, name='c3s1k64_2')
        return tf.add(ipt, c3s1k64)


def up_sample4(ipt, name='up_sample4', reuse=False, is_training=True):
    with tf.variable_scope(name):
        c3s1k256 = ops.conv2d(ipt, 256, 3, 1, norm=None, activation=None,
                              reuse=reuse, is_training=is_training, name='c3s1k256_1')
        d_to_s_1 = tf.depth_to_space(c3s1k256, 2)
        relu1 = ops.leaky_relu(d_to_s_1)
        c3s1k256 = ops.conv2d(relu1, 256, 3, 1, norm=None, activation=None,
                              reuse=reuse, is_training=is_training, name='c3s1k256_2')
        d_to_s_2 = tf.depth_to_space(c3s1k256, 2)
        relu2 = ops.leaky_relu(d_to_s_2)
        return ops.conv2d(relu2, 3, 3, 1, 1, norm=None, activation=None,
                          name='c3s1k3', reuse=reuse, is_training=is_training)
