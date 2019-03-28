# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: conv_layer.py
@time: 2019-03-27 11:44

Description. 
"""

from layers.base_layer import BaseLayer

import tensorflow as tf


class ConvLayer(BaseLayer):
    def __init__(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1):
        super(ConvLayer, self).__init__()
        self.layer_input = x
        self.weight_dict = weight_dict
        self.is_dropout = is_dropout
        self.is_training = is_training
        self.regularizer_conv = regularizer_conv
        self.stride = 1

    def create(self, x):
        self.layer_input = x

        filt, beta, mean, variance = self.get_conv_fileter_bn()

        conv = tf.nn.conv2d(x, filt, [1, self.stride, self.stride, 1], padding='SAME')

        if self.is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=self.is_training)

        bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training=self.is_training,
                                           beta_initializer=beta, scale=False, moving_mean_initializer=mean,
                                           moving_variance_initializer=variance,
                                           beta_regularizer=self.regularizer_conv)
        self.weight_tensors = [filt, tf.get_variable('batch_normalization/beta'),
                               tf.get_variable('batch_normalization/moving_mean'),
                               tf.get_variable('batch_normalization/moving_variance')]
        self.layer_output = bn

    @staticmethod
    def get_conv_filter_bn(weight_dict, regularizer_conv):
        """
        :param weight_dict:
        :param regularizer_conv:
        :return:
        """
        scope_name = tf.get_variable_scope().name

        filt = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                               regularizer=regularizer_conv)
        beta = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/beta'])
        mean = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_mean'])
        variance = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_variance'])

        return filt, beta, mean, variance
