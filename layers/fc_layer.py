# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: fc_layer
@time: 2019-03-27 14:54

Description. 
"""

from layers.base_layer import BaseLayer
from abc import abstractmethod

import tensorflow as tf


class FullConnectedLayer(BaseLayer):
    def __init__(self, x, weight_dict=None, regularizer_fc=None, is_musked=False):
        super(FullConnectedLayer, self).__init__()
        self.layer_type = 'F'
        self.create(x, weight_dict, regularizer_fc, is_musked)

    def create(self, x, weight_dict=None, regularizer_fc=None, is_musked=False):
        self.layer_input = x

        weights, biases = self.get_fc_param(weight_dict, regularizer_fc)
        self.weight_tensors = [weights, biases]
        if is_musked:
            musk = tf.get_variable(name="musk", initializer=weight_dict[self.layer_name + '/musk'], trainable=False)
            fc = tf.matmul(x, weights * musk)
        else:
            fc = tf.matmul(x, weights)

        fc = tf.nn.bias_add(fc, biases)

        self.layer_output = fc
        return self.layer_output

    def get_fc_param(self, weight_dict, regularizer_fc):
        weights = tf.get_variable(name="weights", initializer=weight_dict[self.layer_name + '/weights'],
                                  regularizer=regularizer_fc, trainable=True)
        biases = tf.get_variable(name="biases", initializer=weight_dict[self.layer_name + '/biases'],
                                 regularizer=regularizer_fc, trainable=True)

        return weights, biases
