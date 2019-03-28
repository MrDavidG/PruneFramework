# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: res_block
@time: 2019-03-28 11:52

Description. 
"""


from layers.conv_layer import ConvLayer
import tensorflow as tf


class ResBlock:
    def __init__(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, scale_down=False, is_shared=[False, False], share_scope=None, is_merge_bn=False):
        self.layer_input = x
        self.layer_output, conv_1, conv_2 = self.create(x, weight_dict, is_dropout, is_training, regularizer_conv, scale_down, is_shared, share_scope, is_merge_bn)
        self.layers = [conv_1, conv_2]

    @staticmethod
    def create(x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, scale_down=False, is_shared=[False, False], share_scope=None, is_merge_bn=False):

        if scale_down:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope("conv_1"):
            conv_1_layer = ConvLayer(x, weight_dict, is_dropout, is_training, regularizer_conv, stride, is_shared=is_shared[0], share_scope=share_scope, is_merge_bn=is_merge_bn)
            conv_1_relu = tf.nn.relu(conv_1_layer.layer_output)
        with tf.variable_scope("conv_2"):
            conv_2_layer = ConvLayer(conv_1_relu, weight_dict, is_dropout, is_training, regularizer_conv, is_shared=is_shared[1], share_scope=share_scope, is_merge_bn=is_merge_bn)
        residual = x
        if scale_down:
            residual = tf.nn.avg_pool(residual, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.concat([residual, residual * 0.], axis=-1)

        end_relu = tf.nn.relu(conv_2_layer.layer_output + residual)

        return end_relu, conv_1_layer, conv_2_layer

    def get_params(self, sess):
        return {**self.layers[0].get_params(sess), **self.layers[1].get_params(sess)}