import tensorflow as tf
import layer
import numpy as np
import utils


class RESESPCN:
    def __init__(self, name, image_size=256, norm='batch', num_residual=16,
                 learning_rate=0.001):
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.name = name
        self.norm = norm
        self.num_residual = num_residual
        self.reuse = len([var for var in tf.global_variables() if
                          var.name.startswith(self.name)]) > 0
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.y = tf.placeholder(tf.float32, shape=[None, self.image_size,
                                                   self.image_size, 3],
                                name='y')
        self.x = tf.image.resize_images(self.y, (np.int32(self.image_size/4),
                                                 np.int32(self.image_size/4)))
        with tf.variable_scope(self.name):
            c5s1k64 = layer.c5s1k64(self.x, reuse=self.reuse, is_training=self.is_training)
            resblock = []
            for i in range(self.num_residual):
                resblock.append(layer.residual_block(resblock[-1] if i else c5s1k64, i,
                                                     reuse=self.reuse,
                                                     is_training=self.is_training))
            c3s1k64 = layer.c3s1k64(resblock[-1], reuse=self.reuse, is_training=self.is_training)
            sum1 = tf.add(c3s1k64, c5s1k64)

            upsample4 = layer.up_sample4(sum1, reuse=self.reuse, is_training=self.is_training)
            self.output = upsample4
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.loss = tf.reduce_mean(tf.square(self.output - self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).\
                minimize(self.loss, var_list=self.var_list)

    def model(self):
        x_nearest = tf.image.resize_images(self.x, (self.image_size, self.image_size),
                                           tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x_bilins = tf.image.resize_images(self.x, (self.image_size, self.image_size),
                                          tf.image.ResizeMethod.BILINEAR)
        x_bicubic = tf.image.resize_images(self.x, (self.image_size, self.image_size),
                                           tf.image.ResizeMethod.BICUBIC)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.image('Origin image', utils.batch_convert2int(self.y))
        tf.summary.image('Near', utils.batch_convert2int(x_nearest))
        tf.summary.image('Bilinears', utils.batch_convert2int(x_bilins))
        tf.summary.image('Bicubic', utils.batch_convert2int(x_bicubic))
        tf.summary.image('Reconstruct', utils.batch_convert2int(self.output))
