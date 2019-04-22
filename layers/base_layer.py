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
        self.layer_name = '/'.join(tf.get_variable_scope().name.split('/')[1:])
        self.weight_tensors = list()
        self.layer_input = None
        self.layer_output = None
        self.layer_type = None


    @abstractmethod
    def create(self, x, **params):
        pass

    def get_params(self, sess):
        weight_vals = sess.run(self.weight_tensors)
        weight_dict = dict()
        for i in range(len(self.weight_tensors)):
            key = '/'.join(self.weight_tensors[i].name.split('/')[1:])
            weight_dict[key] = weight_vals[i]
        return weight_dict
