# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: baseline_pruning
@time: 2019-05-17 14:07

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/")

from models.vgg_combine import VGG_Combined
from utils.config import process_config
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def rebuild_model(weight_a, weight_b, cluster_res_list, signal_list, gamma, regu_decay, ib_threshold):
    """
    Rebuild the combined model and train.
    :param weight_a: weight dictionary of model a
    :param weight_b: weight dictionary of model b
    :param cluster_res_list:
    :param signal_list:
    :param gamma: param for kl loss of fc layers
    :param regu_decay: regularizer for A->AB and B->AB
    :return:
    """
    config = process_config("../configs/ib_vgg.json")
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    task_name = 'celeba'
    print('[%s] Rebuild VGG model on task %s' % (datetime.now(), task_name))

    tf.reset_default_graph()
    # session = tf.Session(config=gpu_config)
    session = tf.InteractiveSession(config=gpu_config)

    # Set training params
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_zero = tf.contrib.layers.l2_regularizer(scale=0.)
    regularizer_decay = tf.contrib.layers.l2_regularizer(scale=regu_decay * 1.)

    # Rebuild model
    model = VGG_Combined(config, task_name, weight_a, weight_b, cluster_res_list, signal_list, musk=False, gamma=gamma,
                         ib_threshold=ib_threshold)
    model.set_global_tensor(training, regularizer_zero, regularizer_decay, regularizer_zero)
    model.build()

    # Train
    session.run(tf.global_variables_initializer())

    model.eval_once(session, model.test_init, -1)

    model.train(sess=session, n_epochs=20, task_name='AB', lr=0.01)
    model.train(sess=session, n_epochs=30, task_name='AB', lr=0.001)
    model.train(sess=session, n_epochs=30, task_name='AB', lr=0.0001)


def get_connection_signal(cluster_res_list, dim_list):
    """
    Get the connection signal for all layers (including output layer)
    :param cluster_res_dict:
    :param dim_list:
    :return:
    """
    signal_list = list()

    # cluster_res_list只有15层，不包括输出层
    for layer_index, layer_clusters in enumerate(cluster_res_list):

        signal_layer_dict = dict()

        num_A = len(layer_clusters['A'])
        num_AB = len(layer_clusters['AB'])
        num_AB_from_a = (np.array(layer_clusters['AB']) < dim_list[layer_index]).sum()
        num_AB_from_b = num_AB - num_AB_from_a
        num_B = len(layer_clusters['B'])

        if layer_index == 0:
            signal_layer_dict['A'] = not num_A == 0
            signal_layer_dict['AB'] = not num_AB == 0
            signal_layer_dict['B'] = not num_B == 0
        else:
            num_A_last = len(cluster_res_list[layer_index - 1]['A'])
            num_AB_last = len(cluster_res_list[layer_index - 1]['AB'])
            num_AB_from_a_last = (np.array(cluster_res_list[layer_index - 1]['AB']) < dim_list[layer_index - 1]).sum()
            num_AB_from_b_last = num_AB_last - num_AB_from_a_last
            num_B_last = len(cluster_res_list[layer_index - 1]['B'])

            # 自己的neuron数不为0 且 输入不为0
            signal_layer_dict['A'] = not num_A == 0 and not (num_A_last + num_AB_last) == 0
            signal_layer_dict['AB'] = not num_AB == 0 and not (num_A_last + num_AB_last + num_B_last) == 0
            signal_layer_dict['B'] = not num_B == 0 and not (num_AB_last + num_B_last) == 0
            signal_layer_dict['AB_a'] = not num_AB_from_a == 0
            signal_layer_dict['AB_b'] = not num_AB_from_b == 0

            # If there is such input neuron
            signal_layer_dict['fromA'] = not num_A_last == 0
            signal_layer_dict['fromAB'] = not num_AB_last == 0
            signal_layer_dict['fromAB_a'] = not num_AB_from_a_last == 0
            signal_layer_dict['fromAB_b'] = not num_AB_from_b_last == 0
            signal_layer_dict['fromB'] = not num_B_last == 0

        signal_list.append(signal_layer_dict)

    # num_A_last = len(cluster_res_list[-1]['A'])
    # num_AB_last = len(cluster_res_list[-1]['AB'])
    # num_AB_from_a_last = (np.array(cluster_res_list[-1]['AB']) < 4096).sum()
    # num_AB_from_b_last = num_AB_last - num_AB_from_a_last
    # num_B_last = len(cluster_res_list[-1]['B'])
    #
    # # The output layer
    # signal_layer_dict = dict()
    # signal_layer_dict['A'] = True
    # signal_layer_dict['B'] = True
    # signal_layer_dict['AB'] = False
    #
    # # If there is such input neuron
    # signal_layer_dict['fromA'] = not num_A_last == 0
    # signal_layer_dict['fromAB'] = not num_AB_last == 0
    # signal_layer_dict['fromAB_a'] = not num_AB_from_a_last == 0
    # signal_layer_dict['fromAB_b'] = not num_AB_from_b_last == 0
    # signal_layer_dict['fromB'] = not num_B_last == 0
    #
    # signal_list.append(signal_layer_dict)

    return signal_list


def pruning(model_path_1, model_path_2, gamma=10, ib_threshold=0.01, regu_decay=0):
    """

    :param model_path_1:
    :param model_path_2:
    :param alpha_threshold: For MI pruning
    :param method_mi: 'kde' for unique 2^20 labels, 'kde_in' for unique 20 labels, 'kde_cus' for self compute 2^20 labels,
    'bin' is histogram method for 2^20 labels
    :param binsize:
    :param gamma: VIB factor for fc layer
    :param ib_threshold: pruning threshold for VIB method
    :param regu_decay: adopted for now
    :return:
    """
    dim_list = [64, 64,
                128, 128,
                256, 256, 256,
                512, 512, 512,
                512, 512, 512,
                4096, 4096, 20]

    print('[%s] Obtain model weights, layers output and labels' % (datetime.now()))
    weight_dict_a = pickle.load(open(model_path_1, 'rb'))
    weight_dict_b = pickle.load(open(model_path_2, 'rb'))

    # TODO: 以下为test
    cluster_res_list = list()

    for layer_index in range(15):
        # Total number of neurons
        num_neuron_total = dim_list[layer_index] * 2

        # Store clusters for each layer
        cluster_layer_dict = dict()

        # Init T^A, T^B and T^AB to store clusters for each layer
        cluster_layer_dict['A'] = list()
        cluster_layer_dict['B'] = list()
        cluster_layer_dict['AB'] = [x for x in range(dim_list[layer_index] * 2)]
        cluster_res_list += [cluster_layer_dict]
    # Output layer
    cluster_layer_dict = dict()
    cluster_layer_dict['A'] = [x for x in range(20)]
    cluster_layer_dict['B'] = [x + 20 for x in range(20)]
    cluster_layer_dict['AB'] = list()
    cluster_res_list += [cluster_layer_dict]
    # end Test

    print('[%s] Model Summary:')
    for layer_index, layer_clusters in enumerate(cluster_res_list):
        num_A = len(layer_clusters['A'])
        num_AB = len(layer_clusters['AB'])
        num_AB_from_a = (np.array(layer_clusters['AB']) < dim_list[layer_index]).sum()
        num_AB_from_b = num_AB - num_AB_from_a
        num_B = len(layer_clusters['B'])
        num_pruned = dim_list[layer_index] * 2 - num_A - num_B - num_AB

        print('Layer %d: num_A=%d   |   num_AB=%d(%d:%d)   |   num_B=%d |   num_pruned=%d' % (
            layer_index + 1, num_A, num_AB, num_AB_from_a, num_AB_from_b, num_B, num_pruned))

    # 获得连接与否的flag信号
    signal_list = get_connection_signal(cluster_res_list, dim_list)

    print('[%s] Rebuild and train the combined model' % (datetime.now()))
    rebuild_model(weight_dict_a, weight_dict_b, cluster_res_list, signal_list, gamma=gamma, regu_decay=regu_decay,
                  ib_threshold=ib_threshold)


if __name__ == '__main__':
    path = '/local/home/david/Remote/models/model_weights/'

    pruning(model_path_1=path + 'vgg_celeba1_0.908977_best',
            model_path_2=path + 'vgg_celeba2_0.893588_best',
            gamma=10,
            ib_threshold=0.01)
