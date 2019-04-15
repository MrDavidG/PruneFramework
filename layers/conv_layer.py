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
    def __init__(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1,
                 is_shared=False, share_scope=None, is_merge_bn=False):
        super(ConvLayer, self).__init__()


        if is_merge_bn:
            # with biases
            self.layer_type = 'C'
            self.create_merge(x, weight_dict, is_dropout, is_training, regularizer_conv, stride)
        else:
            # without biases, but with params of batch normalization
            self.layer_type = 'R'
            self.create(x, weight_dict, is_dropout, is_training, regularizer_conv, stride)

    def create(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1):
        self.layer_input = x

        filt, beta, mean, variance = self.get_conv_filter_bn(weight_dict, regularizer_conv)

        conv = tf.nn.conv2d(x, filt, [1, stride, stride, 1], padding='SAME')

        if is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=is_training)

        bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training=is_training,
                                           beta_initializer=beta, scale=False, moving_mean_initializer=mean,
                                           moving_variance_initializer=variance,
                                           beta_regularizer=regularizer_conv)
        # TODO: 后三个变量哪里来的
        self.weight_tensors = [filt, tf.get_variable('batch_normalization/beta'),
                               tf.get_variable('batch_normalization/moving_mean'),
                               tf.get_variable('batch_normalization/moving_variance')]
        self.layer_output = bn

    def create_merge(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1):
        """
        create a conv layer with biases and without bn
        :param x:
        :param weight_dict:
        :param is_dropout:
        :param is_training:
        :param regularizer_conv:
        :param stride:
        :return:
        """
        self.layer_input = x
        filt, biases = self.get_conv_filter_bn_merge(weight_dict, regularizer_conv)
        conv = tf.nn.conv2d(x, filt, [1, stride, stride, 1], padding='SAME')
        conv = conv + biases

        if is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=is_training)
        self.weight_tensors = [filt, biases]
        self.layer_output = conv

    def get_conv_filter_bn(self, weight_dict, regularizer_conv):
        filt = tf.get_variable(name="weights", initializer=weight_dict[self.layer_name + '/weights'],
                               regularizer=regularizer_conv)
        beta = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/beta'])
        mean = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/moving_mean'])
        variance = tf.constant_initializer(weight_dict[self.layer_name + '/batch_normalization/moving_variance'])

        return filt, beta, mean, variance

    def get_conv_filter_bn_merge(self, weight_dict, regularizer_conv, trainable=True):
        filt = tf.get_variable(name="weights", initializer=weight_dict[self.layer_name + '/weights'],
                               regularizer=regularizer_conv, trainable=trainable)
        biases = tf.get_variable(name="biases", initializer=weight_dict[self.layer_name + '/biases'],
                                 regularizer=regularizer_conv, trainable=trainable)

        return filt, biases
