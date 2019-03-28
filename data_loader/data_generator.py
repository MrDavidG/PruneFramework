# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: data_loader.py
@time: 2019-03-27 17:40

Description. 
"""

import tensorflow as tf


class DataGenerator:
    @staticmethod
    def dataset_iterator(dataset_train, dataset_val, dataset_hessian):
        vgg_iter = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        x, y = vgg_iter.get_next()

        # initializer for train_data
        train_init = vgg_iter.make_initializer(dataset_train)
        test_init = vgg_iter.make_initializer(dataset_val)
        hessian_init = vgg_iter.make_initializer(dataset_hessian)

        # return dataset_train, dataset_val, nb_exp_train, np_exp_val
        return train_init, test_init, hessian_init, x, y
