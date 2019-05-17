# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: mi_pruning
@time: 2019-05-09 10:58

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/")

from utils.mutual_information import kde_mi, bin_mi, kde_mi_independent, kde_mi_cus
from models.vgg_combine import VGG_Combined
from models.vgg_celeba import VGGNet
from utils.config import process_config
from utils.mi_gpu import kde_gpu, get_K_function, kde_in_gpu
from datetime import datetime


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'
# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def argmin_mi(layer_output, labelixs, method_mi, binsize, labelprobs=None, labels=None, labels_unique=None,
              labels_count=None, labels_inverse=None, entropy_func_upper=None):
    """
    Get the neuron in layer_output who has the minimal mi with label
    :param layer_output:
    :param labelixs:
    :param method_mi:
    :param binsize:
    :param labelprobs:
    :return: the index, and mi of the neuron with minimal MI
    """
    mi_min = 0
    min_index_neuron = -1

    # Test:
    mis = list()

    for index_neuron in range(layer_output.shape[-1]):
        shape_neuron_output = layer_output[..., index_neuron].shape
        if len(shape_neuron_output) == 1:
            # [batch_size, ]->[batch_size, 1]
            layer_output_expand = np.expand_dims(layer_output[..., index_neuron], axis=1)
        elif len(shape_neuron_output) == 3:
            # [batch_size, h, w]->[batch_size, h*w]
            layer_output_expand = np.reshape(layer_output[..., index_neuron], newshape=(shape_neuron_output[0], -1))

        if method_mi == 'bin':
            _, mi_neuron = bin_mi(layer_output_expand, labelixs=labelixs, binsize=binsize)
        elif method_mi == 'kde':
            _, mi_neuron = kde_mi(layer_output_expand, labelixs=labelixs, labelprobs=labelprobs)
        elif method_mi == 'kde_gpu':
            _, mi_neuron = kde_gpu(layer_output_expand, labelixs=labelixs, labelprobs=labelprobs,
                                   entropy_func_upper=entropy_func_upper)
        elif method_mi == 'kde_in':
            # mi_neuron = kde_mi_independent(layer_output_expand, labels)
            _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)
        elif method_mi == 'kde_cus':
            _, mi_neuron = kde_mi_cus(layer_output_expand, labels_unique, labels_count, labels_inverse)


        mis += [mi_neuron]

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    plt.scatter(layer_output.shape[-1], mis)
    plt.show()


    return min_index_neuron, mi_min


def argmin_marginal_mi(layer_output, F, neuron_list_previous, labelixs, method_mi, binsize, labelprobs=None,
                       labels=None, labels_unique=None, labels_count=None, labels_inverse=None,
                       entropy_func_upper=None):
    """
    Get the neuron in F who has the minimal marginal MI w.r.t neuron_list_previous
    :param layer_output:
    :param F:
    :param neuron_list_previous:
    :param labelixs:
    :param method_mi:
    :param binsize:
    :param labelprobs:
    :param labels:
    :param labels_unique:
    :param labels_count:
    :param labels_inverse:
    :param entropy_func_upper:
    :return:
    """
    mi_min = 0
    min_index_neuron = -1
    for index_neuron in F:
        neuron_list = neuron_list_previous + [index_neuron]

        shape_neurons_output = layer_output[..., neuron_list].shape
        if len(shape_neurons_output) == 4:
            layer_output_expand = np.reshape(layer_output[..., neuron_list], newshape=(shape_neurons_output[0], -1))
        elif len(shape_neurons_output) == 2:
            layer_output_expand = layer_output[..., neuron_list]

        if method_mi == 'bin':
            # 直方图画格子的方法
            _, mi_neuron = bin_mi(layer_output_expand, labelixs=labelixs, binsize=binsize)
        elif method_mi == 'kde':
            # 2^20个unique label的方法
            _, mi_neuron = kde_mi(layer_output_expand, labelixs=labelixs, labelprobs=labelprobs)
        elif method_mi == 'kde_gpu':
            _, mi_neuron = kde_gpu(layer_output_expand, labelixs=labelixs, labelprobs=labelprobs,
                                   entropy_func_upper=entropy_func_upper)
        elif method_mi == 'kde_in':
            # 当成20个彼此之间独立的label
            # mi_neuron = kde_mi_independent(layer_output_expand, labels)
            _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)
        elif method_mi == 'kde_cus':
            # 自己实现的kde方法，主要针对2^20unique label的场景
            _, mi_neuron = kde_mi_cus(layer_output_expand, labels_unique, labels_count, labels_inverse)

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    return min_index_neuron, mi_min


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
    # config = process_config("../configs/vgg_net.json")
    config = process_config("../configs/vgg_for_layer_output.json")
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
    session.run(model.test_init)

    # Graph
    summary_writer = tf.summary.FileWriter('/local/home/david/log/', session.graph)

    model.eval_once(session, model.test_init, -1)

    model.train(sess=session, n_epochs=20, task_name='A', lr=0.01)
    model.train(sess=session, n_epochs=80, task_name='A', lr=0.001)
    model.train(sess=session, n_epochs=80, task_name='A', lr=0.0001)


def combine_models(y_a, y_b, layer_output_list_1, layer_output_list_2, alpha_threshold, method_mi, dim_list, binsize):
    """
    Get the clusters of all layers (except output layer).
    :param y_a: labels of model a
    :param y_b: labels of model b
    :param layer_output_list_1: outputs of layers in model a except output layer
    :param layer_output_list_2: outputs of layers in model b except output layer
    :param alpha_threshold: threshold to divide neurons into different clusters
    :param method_mi: 'kde' or 'bin'
    :param N_total: number of iteration
    :param binsize: length of the bin
    :return:
    """
    num_layer = 15

    num_label_a = y_a.shape[1]
    num_label_b = y_b.shape[1]

    # 获得Y_A和Y_B统计信息
    labelixs_a, labelprobs_a = None, None
    unique_value_a, unique_counts_a, unique_inverse_a = None, None, None

    labelixs_b, labelprobs_b = None, None
    unique_value_b, unique_counts_b, unique_inverse_b = None, None, None

    entropy_func_upper = None

    # 'kde_in'的方法不需要
    if method_mi in ['kde', 'kde_gpu', 'kde_cus', 'bin', 'kde_in']:
        # # For Y_A
        # uniqueids_a = np.ascontiguousarray(y_a).view(np.dtype((np.void, y_a.dtype.itemsize * y_a.shape[1])))
        # unique_value_a, unique_inverse_a, unique_counts_a = np.unique(uniqueids_a, return_index=False,
        #                                                               return_inverse=True, return_counts=True)
        # # 每一个独特的行（label）在整体中出现的概率，相当于labelprobs
        # labelprobs_a = np.asarray(unique_counts_a / float(sum(unique_counts_a)))
        # # 每一个独特的行（label）在整体中出现的位置
        # labelixs_a = {}
        # for label_index, label_value in enumerate(unique_value_a):
        #     labelixs_a[label_index] = unique_inverse_a == label_index
        #
        # # For Y_B
        # uniqueids_b = np.ascontiguousarray(y_b).view(np.dtype((np.void, y_b.dtype.itemsize * y_b.shape[1])))
        # unique_value_b, unique_inverse_b, unique_counts_b = np.unique(uniqueids_b, return_index=False,
        #                                                               return_inverse=True, return_counts=True)
        #
        # labelprobs_b = np.asarray(unique_counts_b / float(sum(unique_counts_b)))
        #
        # labelixs_b = {}
        # for label_index, label_value in enumerate(unique_value_b):
        #     labelixs_b[label_index] = unique_inverse_b == label_index

        # For gpu

        entropy_func_upper = get_K_function()

        labelixs_a = list()
        labelprobs_a = list()
        for label_index in range(y_a.shape[1]):
            labelixs = {}
            labelixs[0] = y_a[:, label_index] == -1
            labelixs[1] = y_a[:, label_index] == 1
            labelixs_a.append(labelixs)

            prob_label = np.mean((y_a[:, label_index] == 1).astype(np.float32), axis=0)
            labelprobs = np.array([1 - prob_label, prob_label])
            labelprobs_a.append(labelprobs)

        labelixs_b = list()
        labelprobs_b = list()
        for label_index in range(y_b.shape[1]):
            labelixs = {}
            labelixs[0] = y_b[:, label_index] == -1
            labelixs[1] = y_b[:, label_index] == 1
            labelixs_b.append(labelixs)

            prob_label = np.mean((y_b[:, label_index] == 1).astype(np.float32), axis=0)
            labelprobs = np.array([1 - prob_label, prob_label])
            labelprobs_b.append(labelprobs)


    # Record list of clusters for all layers
    cluster_res_list = list()

    # Init dictionary to store clusters
    for layer_index in range(num_layer):
        # Total number of neurons
        num_neuron_total = dim_list[layer_index] * 2

        # Store clusters for each layer
        cluster_layer_dict = dict()

        # Init T^A, T^B and T^AB to store clusters for each layer
        cluster_layer_dict['A'] = list()
        cluster_layer_dict['B'] = list()
        cluster_layer_dict['AB'] = np.arange(num_neuron_total).tolist()
        cluster_res_list += [cluster_layer_dict]

    # Output layer
    cluster_layer_dict = dict()
    cluster_layer_dict['A'] = [x for x in range(num_label_a)]
    cluster_layer_dict['B'] = [x + num_label_a for x in range(num_label_b)]
    cluster_layer_dict['AB'] = list()
    cluster_res_list += [cluster_layer_dict]

    # The main loop
    for layer_index in range(13, num_layer):
        print('[%s] Cluster layer %d' % (datetime.now(), layer_index))
        # Total number of neurons
        num_neuron_total = dim_list[layer_index] * 2

        # Obtain output of this layer
        layer_output_1 = layer_output_list_1[layer_index]
        layer_output_2 = layer_output_list_2[layer_index]

        # 展开的操作
        if len(layer_output_1.shape) == 4 and len(layer_output_2.shape) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_1, layer_output_2 = list(
                map(lambda x: np.max(x, axis=(1, 2)), [layer_output_1, layer_output_2]))

        # Have the same number of neurons
        assert (layer_output_1.shape[1] == layer_output_2.shape[1])

        # Output of all the neurons
        layer_output_all = np.concatenate((layer_output_1, layer_output_2), axis=-1)

        # All neurons (Line2)
        F_A = np.arange(num_neuron_total).tolist()
        F_B = np.arange(num_neuron_total).tolist()

        # Init with the neuron of F_A that has the  minimal MI with Y_B
        min_index_neuron, mi_min = argmin_mi(layer_output_all[..., F_A], labelixs_b, method_mi, binsize, labelprobs_b,
                                             y_b, unique_value_b, unique_counts_b, unique_inverse_b, entropy_func_upper)
        print('[%s] MI with Y_B: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))

        # Lines 3-4
        F_A.remove(min_index_neuron)
        cluster_res_list[layer_index]['A'].append(min_index_neuron)

        while mi_min <= alpha_threshold:
            # Traverse neurons in F_A and find the neuron with the minimal mi with Y_B
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_A, cluster_res_list[layer_index]['A'],
                                                          labelixs_b, method_mi, binsize, labelprobs_b, y_b,
                                                          unique_value_b, unique_counts_b, unique_inverse_b,
                                                          entropy_func_upper)
            print('[%s] MI with Y_B: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))
            # Lines 7-8
            F_A.remove(min_index_neuron)
            cluster_res_list[layer_index]['A'].append(min_index_neuron)

        min_index_neuron, mi_min = argmin_mi(layer_output_all[..., F_B], labelixs_a, method_mi, binsize, labelprobs_a,
                                             y_a, unique_value_a, unique_counts_a, unique_inverse_a, entropy_func_upper)
        print('[%s] MI with Y_A: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))

        # Line 11
        F_B.remove(min_index_neuron)

        # Test if min_index_neuron is also irrelevant to B
        if min_index_neuron in cluster_res_list[layer_index]['A']:
            cluster_res_list[layer_index]['A'].remove(min_index_neuron)
            cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
        else:
            cluster_res_list[layer_index]['B'].append(min_index_neuron)

        while mi_min <= alpha_threshold:
            # Traverse neurons in F_B to find the neuron with the minimal mi with Y_A
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_B, cluster_res_list[layer_index]['B'],
                                                          labelixs_a, method_mi, binsize, labelprobs_a, y_a,
                                                          unique_value_a, unique_counts_a, unique_inverse_a,
                                                          entropy_func_upper)
            print('[%s] MI with Y_A: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))

            if min_index_neuron in cluster_res_list[layer_index]['A']:
                cluster_res_list[layer_index]['A'].remove(min_index_neuron)
                cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
            else:
                cluster_res_list[layer_index]['B'].append(min_index_neuron)

            F_B.remove(min_index_neuron)

        # Sort the list of A and B
        cluster_res_list[layer_index]['A'].sort()
        cluster_res_list[layer_index]['B'].sort()

        cluster_res_list[layer_index]['AB'] = list(
            set(cluster_res_list[layer_index]['AB']) - set(cluster_res_list[layer_index]['A']) - set(
                cluster_res_list[layer_index]['A']))

    return cluster_res_list


def get_layers_output(task_name, model_path, with_relu=True):
    """
    Get the weight dictionary, output of each layer (except output layer)
    and labels.
    :param task_name: point to dataset
    :param model_path:
    :return:
    """
    config = process_config("../configs/vgg_for_layer_output.json")
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    print('[%s] Rebuild VGG model on task %s' % (datetime.now(), task_name))

    tf.reset_default_graph()

    sess = tf.Session(config=gpu_config)

    # Set training params
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_zero = tf.contrib.layers.l2_regularizer(scale=0.)

    # Obtain model
    model = VGGNet(config, task_name, model_path=model_path)
    model.set_global_tensor(training, regularizer_zero, regularizer_zero)
    model.inference()

    sess.run(tf.global_variables_initializer())
    sess.run(model.test_init)

    # layers_output_tf = [layer.layer_output for layer in model.layers[:-1]]
    layers_output_tf = [layer.layer_output for layer in model.layers]
    layers_output, labels = sess.run([layers_output_tf] + [model.Y], feed_dict={model.is_training: False})

    if with_relu:
        layers_output = [x * (x > 0) for x in layers_output]

    return model.weight_dict, layers_output, labels


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


def pruning(model_path_1, model_path_2, alpha_threshold, method_mi, binsize, gamma=10, ib_threshold=0.01, regu_decay=0):
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
    weight_dict_a, layers_output_list_a, y_a, = get_layers_output('celeba1', model_path=model_path_1)
    weight_dict_b, layers_output_list_b, y_b, = get_layers_output('celeba2', model_path=model_path_2)

    print('[%s] Divide neurons in each layer into clusters A, B and AB' % (datetime.now()))
    cluster_res_list = combine_models(y_a, y_b, layers_output_list_a, layers_output_list_b,
                                      alpha_threshold=alpha_threshold, method_mi=method_mi, dim_list=dim_list,
                                      binsize=binsize)

    print('[%s] Save cluster results' % (datetime.now()))
    pickle.dump(open('/local/home/david/Remote/models/model_weights/cluster_res_list_alpha-'+str(alpha_threshold), 'wb'))

    # TODO: 以下为test
    # cluster_res_list = list()
    #
    # for layer_index in range(15):
    #     # Total number of neurons
    #     num_neuron_total = dim_list[layer_index] * 2
    #
    #     # Store clusters for each layer
    #     cluster_layer_dict = dict()
    #
    #     # Init T^A, T^B and T^AB to store clusters for each layer
    #     cluster_layer_dict['A'] = list()
    #     cluster_layer_dict['B'] = list()
    #     cluster_layer_dict['AB'] = [x for x in range(dim_list[layer_index] * 2)]
    #     cluster_res_list += [cluster_layer_dict]
    # # Output layer
    # cluster_layer_dict = dict()
    # cluster_layer_dict['A'] = [x for x in range(20)]
    # cluster_layer_dict['B'] = [x + 20 for x in range(20)]
    # cluster_layer_dict['AB'] = list()
    # cluster_res_list += [cluster_layer_dict]
    # # end Test

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
            alpha_threshold=0.001,
            method_mi='bin',
            binsize=0.5,
            gamma=10,
            ib_threshold=0.01)
