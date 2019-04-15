# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: bn_layer.py
@time: 2019-03-27 15:21

Description. 
"""

from layers.base_layer import BaseLayer

import tensorflow as tf


class BatchNormalizeLayer(BaseLayer):
    def __init__(self, x, weight_dict=None, regularizer_conv=None, is_training=False):
        super(BatchNormalizeLayer, self).__init__()
        self.create(x, weight_dict, regularizer_conv, is_training)

    def create(self, x, weight_dict=None, regularizer_conv=None, is_training=False):
        self.layer_input = x

        beta, mean, variance = self.get_bn_param(weight_dict)

        bn = tf.layers.batch_normalization(x, momentum=0.1, epsilon=1e-05, training=is_training,
                                           beta_initializer=beta, scale=False, moving_mean_initializer=mean,
                                           moving_variance_initializer=variance,
                                           beta_regularizer=regularizer_conv)
        self.layer_output = bn
        return self.layer_output

    def get_bn_param(self, weight_dict):
        beta = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/beta'])
        mean = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/moving_mean'])
        variance = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/moving_variance'])

        return beta, mean, variance
