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
    def __init__(self, x, weight_dict=None, regularizer_fc=None):
        super(FullConnectedLayer, self).__init__()
        self.create(x, weight_dict, regularizer_fc)

    def create(self, x, weight_dict=None, regularizer_fc=None):
        self.layer_input = x

        weights, biases = self.get_fc_param(weight_dict, regularizer_fc)
        self.weight_tensors = [weights, biases]
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        self.layer_output = fc
        return self.layer_output

    # 参数只实例化一次，之后就存在内存中
    @staticmethod
    def get_fc_param(weight_dict, regularizer_fc):
        scope_name = tf.get_variable_scope().name

        weights = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                                  regularizer=regularizer_fc)
        biases = tf.get_variable(name="biases", initializer=weight_dict[scope_name + '/biases'],
                                 regularizer=regularizer_fc)

        return weights, biases
