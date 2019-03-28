# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: test
@time: 2019-03-27 21:43

Description. 
"""

import numpy as np
import tensorflow as tf
import pickle

def run():
    dataset_name='ucf101'
    handler = open('../datasets/decathlon_mean_std.pickle', 'rb')
    dict_mean_std = pickle.load(handler, encoding='bytes')
    print(dict_mean_std)
    means = np.array(dict_mean_std[bytes(dataset_name + 'mean', encoding='utf8')], dtype=np.float32)
    means_tensor = tf.constant(np.expand_dims(np.expand_dims(means, axis=0), axis=0))
    print(np.expand_dims(np.expand_dims(means, axis=0), axis=0))


if __name__ == '__main__':
    run()