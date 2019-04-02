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
    def __init__(self, x, weight_dict=None):
        """

        :param x: 相当于论文中的f_i(h_{i-1})
        :param weight_dict:
        """
        super(InformationBottleneckLayer, self).__init__()
        self.create(x, weight_dict)

    def create(self, x, weight_dict=None):
        self.layer_input = x

        # TODO: 0 or 1?[0]: batch_size, [1]:dim
        dim = tf.shape(x)[1]
        # get parameters
        mu, delta = self.get_ib_param(weight_dict)
        # 每次都重新初始化
        epsilon = tf.random.normal(shape=[dim, 1])
        self.weight_tensors = [mu, delta]
        ib = (mu + epsilon * delta) * x

        self.layer_output = ib
        return self.layer_output, self.get_kld(x)

    def get_kld(self, x):
        mu, delta = self.weight_tensors
        # 对应一个neuron的kl散度
        KLD = tf.reduce_sum(tf.log(1 + tf.pow(mu, 2) / (delta + 1e-8)))
        # 乘以一个feature map的大小
        if x.dim() > 2:
            KLD *= np.pord(tf.shape(x)[2:])

        return KLD * 0.5

    @staticmethod
    def get_ib_param(self, weight_dict):
        scope_name = tf.get_variable_scope().name

        mu = tf.get_variable(name='mu', initializer=weight_dict[scope_name + '/mu'], trainable=True)
        delta = tf.get_variable(name='delta', initializer=weight_dict[scope_name + '/delta'], trainable=True)

        return mu, delta
