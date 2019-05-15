# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: baseline_pruning
@time: 2019-05-09 14:40

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/")

from utils.config import process_config
from models.vgg_celeba import VGGNet
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def baseline2():
    # Retrain with VIB
    config = process_config("../configs/ib_vgg.json")
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['celeba1', 'celeba2']:
        print('[%s] Training on task {:s}' % (datetime.now(), task_name))
        tf.reset_default_graph()

        # Init training params
        session = tf.Session(config=gpu_config)
        training = tf.placeholder(dtype=tf.bool, name='training')
        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.001)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.001)

        # Obtain model and weights
        model = VGGNet(config, task_name, musk=False)

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        model.eval_once(session, model.test_init, -1)
        model.train(sess=session, n_epochs=10, lr=0.1)
        model.get_CR()
        model.train(sess=session, n_epochs=10, lr=0.01)
        model.get_CR()
        model.train(sess=session, n_epochs=10, lr=0.001)
        model.get_CR()


def get_tensors(shape, type, scale=None):
    if type == 'NORMAL':
        return np.random.normal(loc=0, scale=scale, size=shape)
    elif type == 'ZERO':
        return np.zeros(shape=shape)


def get_weight_combine(weight_dict_1, weight_dict_2, type_init):
    # New weights dict
    weight_dict_combine = dict()

    # Reorganize weights
    for i, key in enumerate(weight_dict_1.keys()):
        if key == 'is_merge_bn':
            continue

        # Layers weights of two models
        weight_layer_1 = weight_dict_1[key]
        weight_layer_2 = weight_dict_2[key]

        if len(weight_layer_1.shape) == 1:
            # Biases
            weight_combine = np.concatenate((weight_layer_1, weight_layer_2), axis=0)
        else:
            # Weights
            if len(weight_layer_1.shape) == 4:
                index_depth, index_channel = 2, 3
            elif len(weight_layer_1.shape) == 2:
                index_depth, index_channel = 0, 1

            # TODO: Input layer
            if i == 0:
                # Just channel
                weight_combine = np.concatenate((weight_layer_1, weight_layer_2), axis=index_channel)
            else:
                # Depth first
                weight_layer_1_expand = np.concatenate(
                    (weight_layer_1, get_tensors(weight_layer_1.shape, type_init, weight_layer_1.shape[index_depth])),
                    axis=index_depth)
                weight_layer_2_expand = np.concatenate(
                    (get_tensors(weight_layer_2.shape, type_init, weight_layer_2.shape[index_depth]), weight_layer_2),
                    axis=index_depth)

                # Then combine channel
                weight_combine = np.concatenate((weight_layer_1_expand, weight_layer_2_expand), axis=index_channel)

        weight_dict_combine[key] = weight_combine.astype(np.float32)

    return weight_dict_combine


def baseline1(model_path_1, model_path_2, type_init):
    # Load model weight
    weight_dict_1 = pickle.load(open(model_path_1, 'rb'))
    weight_dict_2 = pickle.load(open(model_path_2, 'rb'))

    # Obtain combined weights
    print('[%s] Combine two models' % (datetime.now()))
    weight_dict_combine = get_weight_combine(weight_dict_1, weight_dict_2, type_init=type_init)

    # Retrain with VIB
    print('[%s] Retrain the combined model' % (datetime.now()))
    config = process_config("../configs/vgg_net.json")
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['celeba']:
        print('[%s] Training on task %s' % (datetime.now(), task_name))
        tf.reset_default_graph()

        # Init training params
        session = tf.Session(config=gpu_config)
        training = tf.placeholder(dtype=tf.bool, name='training')
        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.01)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.01)

        # Obtain model and weights
        model = VGGNet(config, task_name, musk=False)
        model.weight_dict = dict(model.weight_dict, **weight_dict_combine)

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        # model.eval_once(session, model.test_init, -1)
        model.get_CR(session)
        model.train(sess=session, n_epochs=10, lr=0.1)
        model.get_CR(session)
        model.train(sess=session, n_epochs=10, lr=0.01)
        model.get_CR(session)
        model.train(sess=session, n_epochs=10, lr=0.001)
        model.get_CR(session)


if __name__ == '__main__':
    baseline1(model_path_1='/local/home/david/Remote/models/model_weights/vgg_celeba1_fix_conv_0.889316',
              model_path_2='/local/home/david/Remote/models/model_weights/vgg_celeba2_fix_conv_0.873415',
              type_init='ZERO')
