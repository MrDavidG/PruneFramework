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
    def __init__(self, x, name=None, weight_dict=None, is_training=False):
        super(BatchNormalizeLayer, self).__init__()
        self.create(x, name, weight_dict, is_training)

    def create(self, x, name=None, weight_dict=None, is_training=False):
        self.layer_input = x

        beta, gamma = self.get_bn_param(name, weight_dict)

        bn = tf.layers.batch_normalization(
            x, momentum=0.1, epsilon=1e-05, training=is_training,
            beta_initializer=beta,
            name=name,
            gamma_initializer=gamma
            # moving_mean_initializer=mean,
            # moving_variance_initializer=variance
        )
        # bn = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
        #                                variance_epsilon=1e-05)
        # self.weight_tensors = [beta, gamma], mean, variance]

        self.layer_output = bn

    def get_bn_param(self, name, weight_dict):
        # beta = tf.get_variable(name='beta', initializer=weight_dict[self.layer_name + '/beta'], trainable=False)
        # gamma = tf.get_variable(name='gamma', initializer=weight_dict[self.layer_name + '/gamma'], trainable=False)
        # mean = tf.get_variable(name='mean', initializer=weight_dict[self.layer_name + '/mean'], trainable=False)
        # variance = tf.get_variable(name='var', initializer=weight_dict[self.layer_name + '/var'], trainable=False)
        if self.layer_name == '':
            beta = tf.constant_initializer(weight_dict['%s/beta' % name])
            gamma = tf.constant_initializer(weight_dict['%s/gamma' % name])
        else:
            beta = tf.constant_initializer(weight_dict[self.layer_name + '/%s/beta' % name])
            gamma = tf.constant_initializer(weight_dict[self.layer_name + '/%s/gamma' % name])
        # mean = tf.constant_initializer(weight_dict[self.layer_name + '/mean'])
        # variance = tf.constant_initializer(weight_dict[self.layer_name + '/var'])

        return beta, gamma  # , mean, variance
