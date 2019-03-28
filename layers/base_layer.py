# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: base_layer.py
@time: 2019-03-27 11:24

Description. 
"""

import tensorflow as tf
from abc import abstractmethod


class BaseLayer:
    def __init__(self):
        # TODO: do we need share?
        # self.is_share = False
        self.layer_name = '/'.join(tf.get_variable_scope.name.split('/')[1:])
        self.weight_tensors = list()
        self.layer_input = None
        self.layer_output = None

    @abstractmethod
    def create(self, x, **params):
        pass

    def get_params(self, sess):
        weight_vals = sess.run(self.weight_tensors)
        weight_dict = dict()
        for i in range(len(self.weight_tensors)):
            # TODO: why a list has the attribute name?
            weight_dict[self.weight_tensors.name] = weight_vals[i]
        return weight_dict
