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
from models.vgg_celeba_512 import VGGNet
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
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def draw_mi_hist(layers_output, labels):
    # Labels init
    entropy_func_upper = get_K_function()
    labelixs_ = list()
    labelprobs_ = list()
    for label_index in range(labels.shape[1]):
        labelixs = {}
        labelixs[0] = labels[:, label_index] == -1
        labelixs[1] = labels[:, label_index] == 1
        labelixs_.append(labelixs)

        prob_label = np.mean((labels[:, label_index] == 1).astype(np.float32), axis=0)
        labelprobs = np.array([1 - prob_label, prob_label])
        labelprobs_.append(labelprobs)

    # Draw
    for index_layer, output in enumerate(layers_output):
        # 展开的操作
        if len(output.shape) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            output = np.max(output, axis=(1, 2))

        mi_list = list()
        for index_neuron in range(output.shape[-1]):
            _, mi = kde_in_gpu(np.expand_dims(output[..., index_neuron], axis=1), labelixs_, labelprobs_,
                               entropy_func_upper)
            mi_list += [mi]
        fig = plt.figure()
        plt.hist(mi_list, bins=100, normed=0)
        plt.title('mi distribution with layer_' + str(index_layer))
        plt.xlabel('MI')
        plt.ylabel('Number of neurons')
        plt.savefig('/local/home/david/Remote/models/model_weights/layer_' + str(index_layer) + '.jpg')
        plt.close('all')
        print('Complete drawing layer_' + str(index_layer))


def draw_mi(model_path):
    print('[%s] Obtain model weights, layers output and labels' % (datetime.now()))
    _, layers_output_list_a, y_a, = get_layers_output('celeba1', model_path=model_path)

    draw_mi_hist(layers_output_list_a, y_a)


def argmin_mi(layer_output, labelixs, method_mi, binsize, labelprobs=None, entropy_func_upper=None):
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
    # mi_list = list()
    # index_list_zero = list()

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
            _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)

        # if mi_neuron <= 0.000001:
        #     index_list_zero += [index_neuron]

        # mi_list += [mi_neuron]

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    # _, dd = kde_in_gpu(layer_output[:,index_list_zero[1:2]], labelixs, labelprobs, entropy_func_upper)
    # _, ddd = kde_in_gpu(layer_output[:, index_list_zero], labelixs, labelprobs, entropy_func_upper)

    # plt.hist(mi_list, bins=80, normed=0)
    # # plt.scatter(np.arange(layer_output.shape[-1]), mis, alpha=0.6)
    # plt.savefig('/local/home/david/Remote/models/model_weights/dd.jpg')

    return min_index_neuron, mi_min


def argmin_marginal_mi(layer_output, F, neuron_list_previous, labelixs, method_mi, binsize, labelprobs=None,
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
        elif len(shape_neurons_output) == 1:
            # [batch_size, ]->[batch_size, 1]
            layer_output_expand = np.expand_dims(layer_output[..., index_neuron], axis=1)

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
            _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)

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
    config = process_config("../configs/ib_vgg.json")
    # config = process_config("../configs/vgg_net.json")
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    task_name = 'lfw'
    print('[%s] Rebuild VGG model on task %s' % (datetime.now(), task_name))

    tf.reset_default_graph()
    session = tf.Session(config=gpu_config)
    # session = tf.InteractiveSession(config=gpu_config)

    # Set training params
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_zero = tf.contrib.layers.l2_regularizer(scale=0.)
    regularizer_decay = tf.contrib.layers.l2_regularizer(scale=regu_decay * 1.)

    # Rebuild model
    model = VGG_Combined(config, task_name, weight_a, weight_b, cluster_res_list, signal_list, musk=False, gamma=gamma,
                         ib_threshold=ib_threshold,
                         # model_path=None)
    model_path='/local/home/david/Remote/models/model_weights/vgg512_combine_ib_lfw_0.01_0.839-0.881-0.7969_cr-0.0006-120-epoch-4e-5')
    # model_path='/local/home/david/Remote/models/model_weights/best_vgg512_combine_ib_celeba_0.01_0.8862-0.8939-0.8786_cr-0.0007_rdnet_30-1e-5+10-1e-6-individual')
    model.set_global_tensor(training, regularizer_zero, regularizer_decay, regularizer_zero)
    model.build()

    # Train
    session.run(tf.global_variables_initializer())

    # 建立Graph
    # train_writer = tf.summary.FileWriter('/local/home/david/Remote/log/train/', session.graph)

    model.eval_once(session, model.test_init, -1)
    model.get_CR(session, cluster_res_list, None)
    time_stamp = str(datetime.now())

    # Test
    model.get_CR(session, cluster_res_list, time_stamp)

    print('————————————————————kl_factor=4e-5, 训练120个epoch————————————————————')
    # model.train(sess=session, n_epochs=120, task_name='AB', lr=0.01, time_stamp=time_stamp)
    print('————————————————————改变为1e-6之后, 重新训练0个epoch————————————————————')
    model.kl_factor = 1e-6
    model.loss()
    model.optimize()
    model.evaluate()
    model.eval_once(session, model.test_init, -1, time_stamp=time_stamp)
    # model.train(sess=session, n_epochs=10, task_name='AB', lr=0.01, time_stamp=time_stamp)
    print('————————————————————交替训练任务A和任务B共40次————————————————————')
    model.train_individual(sess=session, n_epochs=40, lr=0.01, time_stamp=time_stamp)


def combine_models(y_a, y_b, layer_output_list_1, layer_output_list_2, alpha_threshold_dict, method_mi, dim_list,
                   binsize,
                   layer_index_range):
    """
    Get the clusters of all layers (except output layer).
    :param y_a: labels of model a
    :param y_b: labels of model b
    :param layer_output_list_1: outputs of layers in model a except output layer
    :param layer_output_list_2: outputs of layers in model b except output layer
    :param alpha_threshold_dict: threshold to divide neurons into different clusters
    :param method_mi: 'kde' or 'bin'
    :param binsize: length of the bin
    :return:
    """

    num_label_a = y_a.shape[1]
    num_label_b = y_b.shape[1]

    # 获得Y_A和Y_B统计信息
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
    for layer_index in range(15):
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

    # 载入之前的结果
    # for layer_index in layer_index_range:
    #     cluster_res_list[layer_index] = pickle.load(open(
    #         '/local/home/david/Remote/models/model_weights/cluster_results/cluster_res_for_layer-' + str(
    #             layer_index) + '_alpha-0.5', 'rb'))

    # The main loop
    for layer_index in layer_index_range:
        # Different alpha_threshold for conv and fc
        if layer_index < 13:
            alpha_threshold = alpha_threshold_dict['conv']
        else:
            alpha_threshold = alpha_threshold_dict['fc']

        print('[%s] Cluster layer %d, alpha threshold = %f' % (datetime.now(), layer_index, alpha_threshold))
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
        # 可能是所有的neurons，或者是从文件中读取出来的结果
        F_A = cluster_res_list[layer_index]['AB']
        F_B = cluster_res_list[layer_index]['AB']

        # Init with the neuron of F_A that has the  minimal MI with Y_B
        min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_A, cluster_res_list[layer_index]['A'],
                                                      labelixs_b, method_mi, binsize, labelprobs_b, entropy_func_upper)

        # min_index_neuron, mi_min = argmin_mi(layer_output_all[..., F_A], labelixs_b, method_mi, binsize, labelprobs_b,
        #                                      entropy_func_upper)
        print('[%s] MI with Y_B, No.1: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))

        # Lines 3-4
        F_A.remove(min_index_neuron)
        cluster_res_list[layer_index]['A'].append(min_index_neuron)
        count_a = 1

        while mi_min <= alpha_threshold:
            # Traverse neurons in F_A and find the neuron with the minimal mi with Y_B
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_A, cluster_res_list[layer_index]['A'],
                                                          labelixs_b, method_mi, binsize, labelprobs_b,
                                                          entropy_func_upper)
            count_a += 1
            print('[%s] MI with Y_B, No.%d: Min_index_neuron=%d,  mi_min=%f' % (
                datetime.now(), count_a, min_index_neuron, mi_min))
            # Lines 7-8
            F_A.remove(min_index_neuron)
            cluster_res_list[layer_index]['A'].append(min_index_neuron)

        min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_B, cluster_res_list[layer_index]['B'],
                                                      labelixs_a, method_mi, binsize, labelprobs_a,
                                                      entropy_func_upper)
        # min_index_neuron, mi_min = argmin_mi(layer_output_all[..., F_B], labelixs_a, method_mi, binsize, labelprobs_a,
        #                                      entropy_func_upper)
        print('[%s] MI with Y_A, No. 1: Min_index_neuron=%d,  mi_min=%f' % (datetime.now(), min_index_neuron, mi_min))
        count_b = 1
        # Line 11
        F_B.remove(min_index_neuron)
        cluster_res_list[layer_index]['B'].append(min_index_neuron)

        while mi_min <= alpha_threshold:
            # Traverse neurons in F_B to find the neuron with the minimal mi with Y_A
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_B, cluster_res_list[layer_index]['B'],
                                                          labelixs_a, method_mi, binsize, labelprobs_a,
                                                          entropy_func_upper)
            count_b += 1
            print('[%s] MI with Y_A, No.%d: Min_index_neuron=%d,  mi_min=%f' % (
                datetime.now(), count_b, min_index_neuron, mi_min))

            # if min_index_neuron in cluster_res_list[layer_index]['A']:
            #     cluster_res_list[layer_index]['A'].remove(min_index_neuron)
            #     cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
            # else:
            #     cluster_res_list[layer_index]['B'].append(min_index_neuron)
            cluster_res_list[layer_index]['B'].append(min_index_neuron)
            F_B.remove(min_index_neuron)

        # Sort the list of A and B
        cluster_res_list[layer_index]['A'].sort()
        cluster_res_list[layer_index]['B'].sort()

        cluster_res_list[layer_index]['AB'] = list(
            set(cluster_res_list[layer_index]['AB']) - set(cluster_res_list[layer_index]['A']) - set(
                cluster_res_list[layer_index]['B']))

        union = set(cluster_res_list[layer_index]['A']) & set(cluster_res_list[layer_index]['B'])
        cluster_res_list[layer_index]['A'] = list(set(cluster_res_list[layer_index]['A']) - union)
        cluster_res_list[layer_index]['B'] = list(set(cluster_res_list[layer_index]['B']) - union)

        # 保存每一层的结果，防止丢失
        path = '/local/home/david/Remote/models/model_weights/cluster_results/'
        pickle.dump(cluster_res_list[layer_index],
                    open(path + 'cluster_res_for_layer-' + str(layer_index) + '_alpha-' + str(alpha_threshold), 'wb'))

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
    sess.run(model.train_init)

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


def pruning(task_name_1, model_path_1, task_name_2, model_path_2, alpha_threshold_dict, method_mi, binsize,
            layer_index_range, gamma=10, ib_threshold=0.01, regu_decay=0, if_rebuild=False, path_cluster_res_list=None):
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
                512, 512, 20]

    if path_cluster_res_list is None:
        print('[%s] Obtain model weights, layers output and labels' % (datetime.now()))
        weight_dict_a, layers_output_list_a, y_a, = get_layers_output(task_name_1, model_path=model_path_1)
        weight_dict_b, layers_output_list_b, y_b, = get_layers_output(task_name_2, model_path=model_path_2)

        print('[%s] Divide neurons in each layer into clusters A, B and AB' % (datetime.now()))
        cluster_res_list = combine_models(y_a, y_b, layers_output_list_a, layers_output_list_b,
                                          alpha_threshold_dict=alpha_threshold_dict, method_mi=method_mi,
                                          dim_list=dim_list, binsize=binsize, layer_index_range=layer_index_range)
        print('[%s] Save cluster results' % (datetime.now()))
        pickle.dump(cluster_res_list, open(
            '/local/home/david/Remote/models/model_weights/cluster_res_list_' + method_mi + '_alpha-' + str(
                alpha_threshold_dict), 'wb'))
    else:
        weight_dict_a = pickle.load(open(model_path_1, 'rb'))
        weight_dict_b = pickle.load(open(model_path_2, 'rb'))
        cluster_res_list = pickle.load(open(path_cluster_res_list, 'rb'))

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

    if if_rebuild:
        # 获得连接与否的flag信号
        signal_list = get_connection_signal(cluster_res_list, dim_list)

        print('[%s] Rebuild and train the combined model' % (datetime.now()))
        rebuild_model(weight_dict_a, weight_dict_b, cluster_res_list, signal_list, gamma=gamma, regu_decay=regu_decay,
                      ib_threshold=ib_threshold)


def combine_cluster_res(alpha_conv, alpha_fc6, alpha_fc7):
    path = '/local/home/david/Remote/models/model_weights/cluster_results/'
    res = list()
    for i in range(15):
        if i <= 12:
            name = path + 'cluster_res_for_layer-' + str(i) + '_alpha-' + str(alpha_conv)
        elif i == 13:
            name = path + 'cluster_res_for_layer-' + str(i) + '_alpha-' + str(alpha_fc6)
        elif i == 14:
            name = path + 'cluster_res_for_layer-' + str(i) + '_alpha-' + str(alpha_fc7)

        w = pickle.load(open(name, 'rb'))
        w['AB'] = list(set(w['AB']) - set(w['A']) - set(w['B']))
        res += [w]

    cluster_layer_dict = dict()
    cluster_layer_dict['A'] = [x for x in range(20)]
    cluster_layer_dict['B'] = [x + 20 for x in range(20)]
    cluster_layer_dict['AB'] = list()
    res += [cluster_layer_dict]

    for i in range(9):
        res[i]['A'] = []
        res[i]['B'] = []


    pickle.dump(res,
                open(path + 'cluster_results_' + str({'conv': alpha_conv, 'fc6': alpha_fc6, 'fc7': alpha_fc7}), 'wb'))
    print(path + 'cluster_results_' + str({'conv': alpha_conv, 'fc6': alpha_fc6, 'fc7': alpha_fc7}))

def get_cluster_res_after_mask(cluster_res_list, ib_threshold, path_rdnet):
    # 根据模型得到相应的结果
    # get_mask的np版本
    def get_mask(mu, logD, threshold=0):
        # logalpha: [dim]
        logalpha = logD - np.log(np.power(mu, 2) + 1e-8)
        mask = logalpha < threshold
        return mask

    weight_rdnet = pickle.load(open(path_rdnet, 'rb'))

    layer_name_lists = ['conv1_1', 'conv1_2',
                        'conv2_1', 'conv2_2',
                        'conv3_1', 'conv3_2', 'conv3_3',
                        'conv4_1', 'conv4_2', 'conv4_3',
                        'conv5_1', 'conv5_2', 'conv5_3',
                        'fc6', 'fc7', 'fc8']

    for layer_index, cluster_res_layer in enumerate(cluster_res_list):
        if layer_index == 15:
            break
        layer_name = layer_name_lists[layer_index]
        for cluster_name in ['A', 'AB', 'B']:
            if len(cluster_res_layer.get(cluster_name, list())) != 0:
                mu = weight_rdnet[layer_name + '/' + cluster_name + '/info_bottle/mu']
                logD = weight_rdnet[layer_name + '/' + cluster_name + '/info_bottle/logD']
                # 0的为屏蔽掉的，1的为保留的
                mask = get_mask(mu, logD, ib_threshold)
                # 直接在cluster_res_list上面做修改
                mask_result = list(np.array(cluster_res_layer[cluster_name])[mask])
                cluster_res_list[layer_index][cluster_name] = mask_result
    return cluster_res_list


def get_inference_time(model_path_1, model_path_2, gamma, ib_threshold, path_cluster_res_list, path_rdnet):
    dim_list = [64, 64,
                128, 128,
                256, 256, 256,
                512, 512, 512,
                512, 512, 512,
                512, 512, 20]

    weight_a = pickle.load(open(model_path_1, 'rb'))
    weight_b = pickle.load(open(model_path_2, 'rb'))
    cluster_res_list = pickle.load(open(path_cluster_res_list, 'rb'))

    # 根据musk来处理cluster_res_list
    cluster_res_list = get_cluster_res_after_mask(cluster_res_list, ib_threshold, path_rdnet)

    # 只有A的
    for i in range(15):
        cluster_res_list[i]['B'] = []
    # 只有B的
    # for i in range(15):
    #     cluster_res_list[i]['A'] = []

    signal_list = get_connection_signal(cluster_res_list, dim_list)

    # build model
    config = process_config("../configs/ib_vgg.json")
    # config = process_config("../configs/vgg_net.json")
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    task_name = 'lfw'
    print('[%s] Rebuild VGG model on task %s' % (datetime.now(), task_name))

    tf.reset_default_graph()
    session = tf.Session(config=gpu_config)
    # session = tf.InteractiveSession(config=gpu_config)

    # Set training params
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_zero = tf.contrib.layers.l2_regularizer(scale=0.)
    regularizer_decay = tf.contrib.layers.l2_regularizer(scale=0.)

    model = VGG_Combined(config, task_name, weight_a, weight_b, cluster_res_list, signal_list, musk=False, gamma=gamma,
                         ib_threshold=ib_threshold,
                         model_path=None)

    model.set_global_tensor(training, regularizer_zero, regularizer_decay, regularizer_zero)
    model.build()
    session.run(tf.global_variables_initializer())

    model.eval_once(session, model.test_init, -1)
    model.get_CR(session, cluster_res_list, None)


if __name__ == '__main__':
    path = '/local/home/david/Remote/models/model_weights/'

    # 画每一层的MI直方图
    # draw_mi(path + 'vgg512_celeba2_0.892631_best')

    # rdnet方法 celeba
    # pruning(task_name_1='celeba1',
    #         model_path_1=path + 'best_vgg512_celeba1_0.907119',
    #         task_name_2='celeba2',
    #         model_path_2=path + 'best_vgg512_celeba2_0.892631',
    #         alpha_threshold_dict={'conv': 0.2, 'fc': 8},
    #         method_mi='kde_in',
    #         binsize=0.05,
    #         layer_index_range=[13, 14],
    #         gamma=15,
    #         ib_threshold=7,
    #         if_rebuild=False,
    #         path_cluster_res_list=None)
    # path_cluster_res_list='/local/home/david/Remote/models/model_weights/cluster_results/cluster_results_{\'conv\': 0.2, \'fc\': 8}')

    # rdnet方法 lfw
    # pruning(task_name_1='lfw1',
    #         model_path_1=path + 'vgg512_lfw1_0.9039',
    #         task_name_2='lfw2',
    #         model_path_2=path + 'vgg512_lfw2_0.833219',
    #         alpha_threshold_dict={'conv': 0.2, 'fc': 5},
    #         method_mi='kde_in',
    #         binsize=0.05,
    #         layer_index_range=[12],
    #         gamma=25,
    #         ib_threshold=0.01,
    #         if_rebuild=True,
    #         path_cluster_res_list='/local/home/david/Remote/models/model_weights/cluster_results/cluster_results_{\'conv\': 0, \'fc6\': 5, \'fc7\': 7}')

    # 合并每一层的cluster结果，并保存在/cluster_results目录下
    # combine_cluster_res(alpha_conv=0.2, alpha_fc6=5, alpha_fc7=7)

    # 测试inference time
    get_inference_time(model_path_1=path + 'vgg512_lfw1_0.9039',
                       model_path_2=path + 'vgg512_lfw2_0.833219',
                       gamma=25,
                       ib_threshold=0.01,
                       path_cluster_res_list='/local/home/david/Remote/models/model_weights/cluster_results/cluster_results_{\'conv\': 0, \'fc6\': 5, \'fc7\': 7}',
                       path_rdnet='/local/home/david/Remote/models/model_weights/best_vgg512_combine_ib_lfw_0.01_0.8483-0.887-0.8096_cr-0.0033')
