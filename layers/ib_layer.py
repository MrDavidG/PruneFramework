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
    def __init__(self, x, weight_dict=None, is_training=False, kl_mult=1, mask_threshold=0):
        """

        :param x:
        :param weight_dict:
        :param is_training:
        :param kl_mult:
        """
        super(InformationBottleneckLayer, self).__init__()
        self.kl_mult = kl_mult

        self.create(x, weight_dict, is_training, mask_threshold)

    def create(self, x, weight_dict=None, is_training=False, mask_threshold=0):
        self.layer_input = x
        # x of conv:    [batch_size, height, width, channel_size]
        # x of fc:      [batch_size, dim]
        shape_x = tf.shape(x)
        # get params
        mu, delta = self.get_ib_param(weight_dict=weight_dict)

        epsilon = tf.random.normal(shape=shape_x)
        self.weight_tensors = [mu, delta]
        # if it isn't training, prune the weights under threshold
        ib = (mu + epsilon * delta) * x
        if is_training:
            self.layer_output = (ib, self.get_kld(x))
        else:
            # prune weights
            mask = self.get_mask(mu, delta, mask_threshold)
            self.layer_output = (ib * mask, self.get_kld(x))

    def get_mask(self, mu, delta, threshold=0):
        alpha = tf.pow(mu, 2) / tf.poe(delta + 1e-8)
        return tf.cast(alpha < threshold, dtype=tf.float32)

    def get_kld(self, x):
        mu, delta = self.weight_tensors
        # 对应一个neuron的kl散度
        KLD = tf.reduce_sum(tf.log(1 + tf.pow(mu, 2) / (delta + 1e-8)))
        # 乘以一个feature map的大小
        KLD *= tf.cond(tf.greater(tf.size(tf.shape(x)), 2), lambda: tf.reduce_prod(tf.to_float(tf.shape(x)[1:3])),
                       lambda: 1.)
        return KLD * 0.5 * self.kl_mult

    def get_ib_param(self, weight_dict):
        mu = tf.get_variable(name='mu', initializer=weight_dict[self.layer_name + '/info_bottle/mu'], trainable=True)
        delta = tf.get_variable(name='delta', initializer=weight_dict[self.layer_name + '/info_bottle/delta'],
                                trainable=True)

        return mu, delta
