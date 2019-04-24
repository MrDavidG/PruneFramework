# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: ib_layer
@time: 2019-04-02 10:41

Description. 
"""

from layers.base_layer import BaseLayer

import tensorflow as tf
import numpy as np


class InformationBottleneckLayer(BaseLayer):
    def __init__(self, x, layer_type, weight_dict=None, is_training=False, kl_mult=1, mask_threshold=0):
        super(InformationBottleneckLayer, self).__init__()
        self.kl_mult = kl_mult
        self.layer_type = layer_type
        self.create(x, weight_dict, is_training, mask_threshold)

    def create(self, x, weight_dict=None, is_training=False, mask_threshold=0):
        self.layer_input = x
        # x of conv:    [batch_size, height, width, channel_size]
        # x of fc:      [batch_size, dim]

        shape_x = tf.shape(x)
        batch_size = shape_x[0]
        # get params
        mu, logD = self.get_ib_param(weight_dict=weight_dict)
        self.weight_tensors = [mu, logD]

        # [batch_size, dim]
        z_scale = self.reparameterize(mu, logD, batch_size)

        # if not training, prune the weights under threshold
        z_scale *= tf.cond(is_training, lambda: 1., lambda: self.get_mask(mask_threshold))

        # obtain the kl diver
        kld = self.get_kld(x)

        # name      conv                                        fc
        # x         [batch_size, height, width, channel_size]   [batch_size, dim]
        # new_shape [batch_size, channel_size, 1, 1]            [batch_size, dim]
        if self.layer_type == 'C_ib':
            new_shape = self.adapt_shape(tf.shape(z_scale), tf.shape(x))
            # convert x from [batch_size, h, w, channel_size]
            # to [batch_size, channel_size, h, w]
            x = tf.transpose(x, perm=(0, 3, 1, 2))
            ib = x * tf.reshape(z_scale, shape=new_shape)
            # convert ib back to [batch_size, h, w, channel_size]
            ib = tf.transpose(ib, perm=(0, 2, 3, 1))
        elif self.layer_type == 'F_ib':
            ib = x * z_scale

        self.layer_output = (ib, kld)

    def adapt_shape(self, src_shape, x_shape):
        """
        if dimension of src_shape = 2:
            new_shape = src_shape
        else：
            new_shape = [1, src_shape[0]]
        if dimension of x_shape > 2:
            new_shape += [1, 1]
        :param src_shape:
        :param x_shape:
        :return:
        """
        # fc: src_shape
        # conv: [1,src_shape[0]]
        new_shape = tf.cond(tf.equal(tf.shape(src_shape)[0], 2), lambda: src_shape,
                            lambda: tf.convert_to_tensor([1, src_shape[0]]))

        # fc: new_shape不变
        # conv: [1, src_shape[0], 1, 1]
        new_shape = tf.cond(tf.greater(tf.shape(x_shape)[0], 2),
                            lambda: tf.concat([new_shape, tf.constant([1, 1])], axis=0),
                            lambda: new_shape)

        return new_shape

    def reparameterize(self, mu, logD, batch_size):
        # std dev
        std = tf.exp(logD * 0.5)

        # num of in_channels for conv
        # num of dim of fc
        dim = tf.shape(logD)[0]

        # the random epsilon
        eps = tf.random.normal(shape=[batch_size, dim])

        # [batch_size, dim]
        return tf.reshape(mu, shape=[1, -1]) + eps * tf.reshape(std, shape=[1, -1])

    def get_mask(self, threshold=0):
        # logalpha: [dim]
        logalpha = self.logD - tf.log(tf.pow(self.mu, 2) + 1e-8)
        mask = tf.cast(logalpha < threshold, dtype=tf.float32)
        return mask

    def get_kld(self, x):
        """
        return kl divergence of this layer with respect to x
        :param x: [batch_size, h, w, channel_size]
        :return: kl divergence
        """
        x_shape = tf.shape(x)

        mu, logD = self.weight_tensors

        # mu: [dim]
        # return: conv  [1, dim, 1, 1] / fc [1, dim]
        new_shape = self.adapt_shape(tf.shape(mu), x_shape)

        h_D = tf.exp(tf.reshape(logD, shape=new_shape))
        h_mu = tf.reshape(mu, shape=new_shape)

        if self.layer_type == 'C_ib':
            KLD = tf.reduce_sum(tf.log(1 + tf.pow(h_mu, 2) / (h_D + 1e-8))) * tf.cast(x_shape[3],
                                                                                      dtype=tf.float32) / tf.cast(
                tf.shape(h_D)[1], dtype=tf.float32)
            KLD *= tf.cast(tf.reduce_prod(x_shape[1:3]), dtype=tf.float32)
        elif self.layer_type == 'F_ib':
            KLD = tf.reduce_sum(tf.log(1 + tf.pow(h_mu, 2) / (h_D + 1e-8))) * tf.cast(x_shape[1],
                                                                                      dtype=tf.float32) / tf.cast(
                tf.shape(h_D)[1], dtype=tf.float32)

        return KLD * 0.5 * self.kl_mult

    def get_ib_param(self, weight_dict):
        mu = tf.get_variable(name='mu', initializer=weight_dict[self.layer_name + '/info_bottle/mu'],
                             trainable=True)
        logD = tf.get_variable(name='logD', initializer=weight_dict[self.layer_name + '/info_bottle/logD'],
                               trainable=True)

        return mu, logD
