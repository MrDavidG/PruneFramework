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
    def __init__(self, x, weight_dict=None, is_training=False, regularizer_conv=None, is_dropout=False, stride=1):
        super(ConvLayer, self).__init__()

        self.gate = None
        # With biases
        self.layer_type = 'C'
        # with biases
        self.create_merge(x, weight_dict, is_dropout, is_training, regularizer_conv=regularizer_conv, stride=stride)

    def create_merge(self, x, weight_dict=None, is_dropout=False, is_training=False,
                     regularizer_conv=None, stride=1, trainable=True):
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
        filt, biases, gate = self.get_conv_filter_bn_merge(weight_dict, regularizer_conv, trainable=trainable)

        conv = tf.nn.conv2d(x, filt, [1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv = conv * gate

        if is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=is_training)
        self.weight_tensors = [filt, biases]
        self.layer_output = conv
        self.gate = gate

    def get_conv_filter_bn_merge(self, weight_dict, regularizer_conv, trainable):
        filt = tf.get_variable(name="w", initializer=weight_dict[self.layer_name + '/w'],
                               regularizer=regularizer_conv, trainable=trainable)
        biases = tf.get_variable(name="b", initializer=weight_dict[self.layer_name + '/b'],
                                 regularizer=regularizer_conv, trainable=trainable)

        gate = tf.ones_like(biases, dtype=tf.float32, name="g")
        return filt, biases, gate
