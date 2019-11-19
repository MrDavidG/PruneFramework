# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: rdnet_triple
@time: 2019-09-18 10:37

Description. 
"""

import sys
from typing import Optional, Any

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.vgg_combine_triple import VGG_Combined
from models.vgg_celeba_512 import VGGNet
from utils.mi_gpu import get_K_function
from utils.mi_gpu import kde_in_gpu
from utils.configer import load_cfg
from utils.configer import get_cfg
from utils.logger import *
from datetime import datetime

from sklearn import metrics as mr

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import copy
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'
# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

global dim_list
global n_fc


def init_tf():
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    return tf.Session(config=gpu_config)


def get_layers_output(model_path, with_relu=True):
    cfg = load_cfg('/'.join(model_path.split('/')[:-1]) + '/cfg.ini')

    sess = init_tf()

    model = VGGNet(cfg)
    model.inference()

    sess.run(tf.global_variables_initializer())
    sess.run(model.train_init)

    layers_output_tf = [layer.layer_output for layer in model.layers]
    layers_output, labels = sess.run([layers_output_tf] + [model.Y], feed_dict={model.is_training: False})

    if with_relu:
        layers_output = [x * (x > 0) for x in layers_output]
    return layers_output, labels


def argmin_marginal_mi(layer_output, F, neuron_list_previous, labelixs, labelprobs=None, entropy_func_upper=None):
    mi_min, min_index_neuron = 0, -1
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

        _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    return min_index_neuron, mi_min


def draw_mi_distribution(layer_output, F, labelixs, labelprobs, entropy_func_upper):
    # 这里默认F中的index_neuron是按照顺序的
    res_mi = list()

    for index_neuron in F:
        neuron_list = [index_neuron]

        shape_neurons_output = layer_output[..., neuron_list].shape
        if len(shape_neurons_output) == 4:
            layer_output_expand = np.reshape(layer_output[..., neuron_list], newshape=(shape_neurons_output[0], -1))
        elif len(shape_neurons_output) == 2:
            layer_output_expand = layer_output[..., neuron_list]
        elif len(shape_neurons_output) == 1:
            # [batch_size, ]->[batch_size, 1]
            layer_output_expand = np.expand_dims(layer_output[..., index_neuron], axis=1)

        _, mi_neuron = kde_in_gpu(layer_output_expand, labelixs, labelprobs, entropy_func_upper)

        res_mi.append(mi_neuron)
    return res_mi


def get_label_info(y):
    labelixs_x, labelprobs_x = list(), list()
    for label_index in range(y.shape[1]):
        labelixs = {}
        labelixs[0] = y[:, label_index] == -1
        labelixs[1] = y[:, label_index] == 1
        labelixs_x.append(labelixs)

        prob_label = np.mean((y[:, label_index] == 1).astype(np.float32), axis=0)
        labelprobs = np.array([1 - prob_label, prob_label])
        labelprobs_x.append(labelprobs)
    return labelixs_x, labelprobs_x


def draw_mi_each_layer(y_a, y_b, layers_output_list_a, layers_output_list_b, path_save=None, histogram=True):
    def save_and_draw(res_mi_a, res_mi_b, layer_index, sorted=False):
        path_b = path_save + '/mi_distribution_layer-%d_label-b' % layer_index
        pickle.dump(res_mi_b, open(path_b, 'wb'))

        path_a = path_save + '/mi_distribution_layer-%d_label-a' % layer_index
        pickle.dump(res_mi_a, open(path_a, 'wb'))

        assert (len(res_mi_a), len(res_mi_b))
        x = np.arange(len(res_mi_a)).tolist()

        layer_name = \
            ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
             'conv4_3',
             'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7'][layer_index]

        # 画图
        if histogram:
            # label_a 左边
            plt.figure(figsize=(6.4, 3.2))
            grid = plt.GridSpec(4, 1, wspace=0.5, hspace=0.5)
            a1 = plt.subplot(grid[0, 0])
            a3 = plt.subplot(grid[1:4, 0], sharex=a1)

            a1.hist(res_mi_a[:int(len(res_mi_a) / 2)], color='red', bins=130, alpha=0.5)
            a1.hist(res_mi_b[:int(len(res_mi_b) / 2)], color='blue', bins=130, alpha=0.5)

            # plt.title('%s label_ab neuron_a' % layer_name)
            # plt.savefig(path_save + '/plot-%s label_a neuron_a' % layer_name)
            # label_b 左边
            # plt.figure(figsize=(8, 5))

            a3.hist(res_mi_a[:int(len(res_mi_a) / 2)], color='red', bins=130, alpha=0.5)
            a3.hist(res_mi_b[:int(len(res_mi_b) / 2)], color='blue', bins=130, alpha=0.5)
            # plt.title('%s label_b neuron_a' % layer_name)
            a3.set_xlabel('MI (bits)', fontsize=14)
            plt.ylabel('Nr. of neurons', fontsize=14)
            a3.tick_params(labelsize=14)
            plt.legend(labels=['MI with label a', 'MI with label b'], fontsize=15)

            # 截断
            a1.set_ylim(100)  # outliers only
            a3.set_ylim(0, 40)  # most of the data

            a1.spines['bottom'].set_visible(False)
            a3.spines['top'].set_visible(False)
            a1.xaxis.tick_top()
            a1.tick_params(labeltop=False, labelsize=13)  # don't put tick labels at the top
            a3.xaxis.tick_bottom()
            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=a1.transAxes, color='k', clip_on=False)
            a1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
            a1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=a3.transAxes)  # switch to the bottom axes
            a3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            a3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            # plt.legend(labels=['MI with label a', 'MI with label b'], fontsize=15)

            # plt.savefig(path_save + '/plot-%s label_ab neuron_a.eps' % layer_name, format='eps')

            # label_a 右边
            # a2 = plt.subplot(grid[0, 1])
            # a4 = plt.subplot(grid[1:4, 1], sharex=a2)
            #
            # # plt.figure(figsize=(8, 5))
            # a2.hist(res_mi_b[int(len(res_mi_b) / 2):], color='blue', bins=130, alpha=0.5)
            # a2.hist(res_mi_a[int(len(res_mi_a) / 2):], color='red', bins=130, alpha=0.5)
            #
            # # plt.title('%s label_ab neuron_b' % layer_name)
            # # plt.savefig(path_save + '/plot-%s label_a neuron_b' % layer_name)
            # # label_b 右边
            # # plt.figure(figsize=(8, 5))
            #
            # a4.hist(res_mi_b[int(len(res_mi_b) / 2):], color='blue', bins=130, alpha=0.5)
            # a4.hist(res_mi_a[int(len(res_mi_a) / 2):], color='red', bins=130, alpha=0.5)
            #
            # # plt.title('%s label_b neuron_b' % layer_name)
            # plt.xlabel('MI')
            # plt.ylabel('Number of neurons')
            # a2.legend(labels=['MI with label b', 'MI with label a'])
            #
            # # 截断
            # a2.set_ylim(100)  # outliers only
            # a4.set_ylim(0, 40)  # most of the data
            #
            # a2.spines['bottom'].set_visible(False)
            # a4.spines['top'].set_visible(False)
            # a2.xaxis.tick_top()
            # a2.tick_params(labeltop=False)  # don't put tick labels at the top
            # a4.xaxis.tick_bottom()
            # d = .015  # how big to make the diagonal lines in axes coordinates
            # # arguments to pass to plot, just so we don't keep repeating them
            # kwargs = dict(transform=a2.transAxes, color='k', clip_on=False)
            # a2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
            # a2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
            #
            # kwargs.update(transform=a4.transAxes)  # switch to the bottom axes
            # a4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            # a4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            plt.tight_layout()
            plt.savefig(path_save + '/plot-%s label_ab neuron_b.eps' % layer_name, format='eps')
        else:
            plt.figure(figsize=(15, 5))
            if sorted:
                plt.plot(x, res_mi_a.sort(), label='label_a', color='red')
                plt.plot(x, res_mi_b.sort(), label='label_b', color='blue')
            else:
                plt.plot(x, res_mi_a, label='label_a', color='red')
                plt.plot(x, res_mi_b, label='label_b', color='blue')
            plt.legend()
            plt.title('MI distribution of %s' % layer_name)
            plt.savefig(path_save + '/plot-%s' % layer_name)

    global dim_list

    n_label_a = y_a.shape[1]
    n_label_b = y_b.shape[1]

    entropy_func_upper = get_K_function()
    labelixs_a, labelprobs_a = get_label_info(y_a)
    labelixs_b, labelprobs_b = get_label_info(y_b)

    cluster_res_list = list()

    # Init for all layers to store cluster res
    for layer_index in range(15):
        n_neuron_total = dim_list[layer_index] * 2

        cluster_layer_dict = dict()
        cluster_layer_dict['A'], cluster_layer_dict['B'] = list(), list()
        cluster_layer_dict['AB'] = np.arange(n_neuron_total).tolist()
        cluster_res_list += [cluster_layer_dict]
    # Output layer
    cluster_layer_dict = dict()
    cluster_layer_dict['A'] = [x for x in range(n_label_a)]
    cluster_layer_dict['B'] = [x + n_label_a for x in range(n_label_b)]
    cluster_layer_dict['AB'] = list()
    cluster_res_list += [cluster_layer_dict]

    # The main loop
    for layer_index in [14]:  # range(15):
        log_t('Layer %d ...' % layer_index)

        # Obtain output of this layer
        layer_output_a = layers_output_list_a[layer_index]
        layer_output_b = layers_output_list_b[layer_index]

        # 展开
        if len(layer_output_a.shape) == 4 and len(layer_output_b.shape) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_a, layer_output_b = list(
                map(lambda x: np.max(x, axis=(1, 2)), [layer_output_a, layer_output_b]))

        layer_output_all = np.concatenate((layer_output_a, layer_output_b), axis=-1)

        F_A = cluster_res_list[layer_index]['AB']
        F_B = cluster_res_list[layer_index]['AB']

        log_t('Get mi distribution with label_b ...')
        res_mi_b = draw_mi_distribution(layer_output_all, F_A, labelixs_b, labelprobs_b, entropy_func_upper)

        log_t('Get mi distribution with label_a ...')
        res_mi_a = draw_mi_distribution(layer_output_all, F_B, labelixs_a, labelprobs_a, entropy_func_upper)

        save_and_draw(res_mi_a, res_mi_b, layer_index)


def extract_unrelevance_neurons(layer_output_all, n_neurons, labelix_a, labelix_b, labelix_c, labelprobs_a,
                                labelprobs_b, labelprobs_c, entropy_func_upper, alpha_threshold):
    def get_irrelevance_neurons(label_name, labelix, labelprobs):
        neurons_list_selected = list()
        count = 1

        # F保留和task有关系的神经元
        F = np.arange(n_neurons).tolist()
        while True:
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F, neurons_list_selected, labelix,
                                                          labelprobs, entropy_func_upper)

            log_t('MI with label Y_%s, No.%d: Min_index_neuron=%d,  mi_min=%f' % (
                label_name, count, min_index_neuron, mi_min))
            count += 1

            if mi_min > alpha_threshold or min_index_neuron == -1:
                break

            F.remove(min_index_neuron)
            # 记下的是所有的与1无关的神经元
            neurons_list_selected.append(min_index_neuron)
        return neurons_list_selected

    neurons_irrelevance_a = set(get_irrelevance_neurons('A', labelix_a, labelprobs_a))
    neurons_irrelevance_b = set(get_irrelevance_neurons('B', labelix_b, labelprobs_b))
    neurons_irrelevance_c = set(get_irrelevance_neurons('C', labelix_c, labelprobs_c))

    neurons_union = neurons_irrelevance_a & neurons_irrelevance_b & neurons_irrelevance_c

    neurons_irrelevance_a_clean = neurons_irrelevance_a - neurons_union
    neurons_irrelevance_b_clean = neurons_irrelevance_b - neurons_union
    neurons_irrelevance_c_clean = neurons_irrelevance_c - neurons_union

    # 返回的是与bc同时无关的，与ac同时无关的，与ab同时无关的神经元,并且三者之间没有交集
    return np.sort(
        list(set(np.arange(n_neurons)) - neurons_irrelevance_a - neurons_irrelevance_b - neurons_irrelevance_c)), \
           np.sort(list(neurons_irrelevance_b_clean & neurons_irrelevance_c_clean)), \
           np.sort(list(neurons_irrelevance_a_clean & neurons_irrelevance_c_clean)), \
           np.sort(list(neurons_irrelevance_a_clean & neurons_irrelevance_b_clean))


def cluster_neurons(y_a, y_b, y_c, layers_output_list_a, layers_output_list_b, layers_output_list_c,
                    cluster_threshold_dict,
                    cluster_layer_range=np.arange(15), path_save=None):
    global dim_list

    # dimension of labels
    n_label_a = y_a.shape[1]
    n_label_b = y_b.shape[1]
    n_label_c = y_b.shape[1]

    entropy_func_upper = get_K_function()
    labelixs_a, labelprobs_a = get_label_info(y_a)
    labelixs_b, labelprobs_b = get_label_info(y_b)
    labelixs_c, labelprobs_c = get_label_info(y_c)

    cluster_res_list = list()

    # Save cluster results
    if not os.path.exists(path_save + '/cluster_results/'):
        os.mkdir(path_save + '/cluster_results/')

    # Init for all layers to store cluster res
    for layer_index in range(15):
        n_neuron_total = dim_list[layer_index] * 3

        cluster_layer_dict = dict()
        cluster_layer_dict['A'], cluster_layer_dict['B'], cluster_layer_dict['C'] = list(), list(), list()
        cluster_layer_dict['ABC'] = np.arange(n_neuron_total).tolist()
        cluster_res_list += [cluster_layer_dict]

    # Output layer
    cluster_layer_dict = dict()
    cluster_layer_dict['A'] = [x for x in range(n_label_a)]
    cluster_layer_dict['B'] = [x + n_label_a for x in range(n_label_b)]
    cluster_layer_dict['C'] = [x + n_label_a + n_label_b for x in range(n_label_c)]
    cluster_layer_dict['ABC'] = list()
    cluster_res_list += [cluster_layer_dict]

    # The main loop
    for layer_index in cluster_layer_range:
        if layer_index < 13:
            alpha_threshold = cluster_threshold_dict['conv']
        else:
            alpha_threshold = cluster_threshold_dict['fc']

        log_t('Cluster layer %d, alpha threshold = %f' % (layer_index, alpha_threshold))

        # Obtain output of this layer
        layer_output_a = layers_output_list_a[layer_index]
        layer_output_b = layers_output_list_b[layer_index]
        layer_output_c = layers_output_list_c[layer_index]

        # 展开
        if len(layer_output_a.shape) == 4 and len(layer_output_b.shape) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_a, layer_output_b, layer_output_c = list(
                map(lambda x: np.max(x, axis=(1, 2)), [layer_output_a, layer_output_b, layer_output_c]))

        layer_output_all = np.concatenate((layer_output_a, layer_output_b, layer_output_c), axis=-1)

        # Find neurons without relevance with B and C
        cluster_res_list[layer_index]['ABC'], cluster_res_list[layer_index]['A'], cluster_res_list[layer_index]['B'], \
        cluster_res_list[layer_index]['C'] = \
            extract_unrelevance_neurons(layer_output_all,
                                        dim_list[layer_index] * 3,
                                        labelixs_a, labelixs_b, labelixs_c,
                                        labelprobs_a, labelprobs_b, labelprobs_c,
                                        entropy_func_upper,
                                        alpha_threshold)

        # 保存layer_index层之前的所有结果，防止丢失
        path = path_save + '/cluster_results/cluster_layer-%d_threshold-%s' % (layer_index, str(alpha_threshold))
        pickle.dump(cluster_res_list[:layer_index + 1], open(path, 'wb'))

    # Save the final result
    path = path_save + '/cluster_results/cluster_result_threshold-%s' % str(cluster_threshold_dict)
    pickle.dump(cluster_res_list, open(path, 'wb'))
    return cluster_res_list


def model_summary(cluster_res_list):
    global dim_list
    for layer_index, layer_clusters in enumerate(cluster_res_list):
        num_A = len(layer_clusters['A'])
        num_B = len(layer_clusters['B'])
        num_C = len(layer_clusters['C'])

        num_ABC = len(layer_clusters['ABC'])
        num_ABC_from_a = (np.array(layer_clusters['ABC']) < dim_list[layer_index]).sum()
        num_ABC_from_b = (np.array(layer_clusters['ABC']) < dim_list[layer_index] * 2).sum() - num_ABC_from_a
        num_ABC_from_c = num_ABC - num_ABC_from_a - num_ABC_from_b

        num_pruned = dim_list[layer_index] * 2 - num_A - num_B - num_C - num_ABC

        log('Layer %d: num_A=%d\t|\tnum_B=%d\t|\tnum_C=%d\t|\tnum_ABC=%d(%d:%d:%d)\t|\tnum_pruned=%d' % (
            layer_index + 1, num_A, num_B, num_C, num_ABC, num_ABC_from_a, num_ABC_from_b, num_ABC_from_c, num_pruned))


def get_cluster_res(cfg):
    path_cluster_res = cfg['cluster']['path_cluster_res']
    if path_cluster_res == 'None':
        print('[%s] Get the outputs of layers in model a, b and c' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        layers_output_list_a, labels_a = get_layers_output(model_path=cfg['basic']['model_a'])
        layers_output_list_b, labels_b = get_layers_output(model_path=cfg['basic']['model_b'])
        layers_output_list_c, labels_c = get_layers_output(model_path=cfg['basic']['model_c'])

        cluster_res_list = cluster_neurons(labels_a, labels_b, labels_c,
                                           layers_output_list_a, layers_output_list_b, layers_output_list_c,
                                           json.loads(cfg['cluster']['cluster_threshold_dict']),
                                           json.loads(cfg['cluster']['cluster_layer_range']), cfg['path']['path_save'])
    else:
        cluster_res_list = pickle.load(open(path_cluster_res, 'rb'))

    model_summary(cluster_res_list)

    return cluster_res_list


def get_connection_signal(cluster_res_list):
    global dim_list
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
    return signal_list


def retrain_model(cfg, cluster_res_list, plan_retrain):
    log_t('Loading weights of model a and b ...')
    weight_a = pickle.load(open(cfg['basic']['model_a'], 'rb'))
    weight_b = pickle.load(open(cfg['basic']['model_b'], 'rb'))
    weight_c = pickle.load(open(cfg['basic']['model_c'], 'rb'))
    log_t('Done')

    signal_list = get_connection_signal(cluster_res_list)

    sess = init_tf()

    model = VGG_Combined(cfg, weight_a, weight_b, weight_c, cluster_res_list, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, model.test_init, -2)
    model.get_CR(sess, cluster_res_list)
    log_l('')

    for plan in plan_retrain:
        model.set_kl_factor(plan['kl_factor'])

        for set_ in plan['train']:
            if plan['type'] == 'normal':
                model.train(sess=sess, n_epochs=set_['n_epochs'], task_name='ABC', lr=set_['lr'])
            elif plan['type'] == 'individual':
                model.train_individual(sess=sess, n_epochs=set_['n_epochs'], lr=set_['lr'])
        model.save_cfg()


def obtain_cfg(task_name, model_path_a, model_path_b, model_path_c, pruning_set=None, cluster_set=None):
    # cfg
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    cfg = get_cfg(task_name, time_stamp, suffix='rdnet')

    # Create exp dir
    if not os.path.exists(cfg['path']['path_save']):
        os.mkdir(cfg['path']['path_save'])
        log_t('Create directory %s' % cfg['path']['path_save'])

    cfg.set('basic', 'model_a', model_path_a)
    cfg.set('basic', 'model_b', model_path_b)
    cfg.set('basic', 'model_c', model_path_c)

    if pruning_set is not None:
        cfg['basic']['pruning_method'] = 'info_bottle'
        cfg.add_section('pruning')
        for option in pruning_set.keys():
            cfg.set('pruning', option, str(pruning_set[option]))

    if cluster_set is not None:
        cfg.add_section('cluster')
        for option in cluster_set.keys():
            cfg.set('cluster', option, str(cluster_set[option]).replace("\'", "\""))

    return cfg


def pruning(task_name, model_path_a, model_path_b, model_path_c, pruning_set, cluster_set, plan_retrain):
    cfg = obtain_cfg(task_name, model_path_a, model_path_b, model_path_c, pruning_set, cluster_set)
    cfg['path']['path_load'] = str(None)

    global dim_list
    dim_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, n_fc, n_fc,
                cfg['data'].getint('n_classes')]

    # Cluster
    cluster_res_list = get_cluster_res(cfg)

    # Save cfg
    with open(cfg['path']['path_cfg'], 'w') as file:
        cfg.write(file)

    # Test
    # cfg = load_cfg('/local/home/david/Remote/PruneFramework/exp_files/celeba-rdnet-2019-07-15 15:14:51/cfg.ini')

    # global dim_list
    # dim_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512,
    #             cfg['data'].getint('n_classes') / 2]
    #
    # cluster_res_list = pickle.load(open(
    #     '/local/home/david/Remote/PruneFramework/exp_files/celeba-rdnet-2019-07-15 15:14:51/cluster_results/cluster_result_threshold-{\'conv\': 0.2, \'fc\': 8}',
    #     'rb'))

    # Retrain
    retrain_model(cfg, cluster_res_list, plan_retrain)


def draw_mi(task_name, model_path_a, model_path_b):
    cfg = obtain_cfg(task_name, model_path_a, model_path_b)
    global dim_list
    dim_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512,
                cfg['data'].getint('n_classes') / 2]

    layers_output_list_a, labels_a = get_layers_output(model_path=cfg['basic']['model_a'])
    layers_output_list_b, labels_b = get_layers_output(model_path=cfg['basic']['model_b'])

    draw_mi_each_layer(labels_a, labels_b, layers_output_list_a, layers_output_list_b, cfg['path']['path_save'])

    # Save cfg
    with open(cfg['path']['path_cfg'], 'w') as file:
        cfg.write(file)


if __name__ == '__main__':
    plan_retrain = [
        {'kl_factor': 4e-5,
         'type': 'normal',
         'train': [{'n_epochs': 80, 'lr': 0.01}]},
        {'kl_factor': 1e-6,
         'type': 'individual',
         'train': [{'n_epochs': 10, 'lr': 0.01}]}
    ]

    global n_fc
    n_fc = 128

    lfw_0 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__0--2019-07-27 17:53:03/tr00-epo010-acc0.9285'
    lfw_11 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__11--2019-07-27 17:27:25/tr01-epo010-acc0.6594'
    lfw_16 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__16--2019-07-27 17:35:43/tr00-epo010-acc0.9433'
    lfw_17 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__17--2019-07-27 17:12:41/tr01-epo010-acc0.9174'
    lfw_18 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__18--2019-07-27 18:00:41/tr01-epo010-acc0.9083'
    lfw_20 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__20--2019-07-27 17:44:03/tr02-epo020-acc0.8782'
    lfw_27 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__27--2019-07-27 17:11:00/tr01-epo010-acc0.7256'
    lfw_32 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__32--2019-07-27 17:19:13/tr02-epo020-acc0.9338'
    lfw_42 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__42--2019-07-27 18:09:00/tr02-epo010-acc0.8151'
    lfw_46 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__46--2019-07-27 17:52:21/tr02-epo020-acc0.8227'
    lfw_58 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__58--2019-07-27 17:28:50/tr02-epo020-acc0.8664'
    lfw_66 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__66--2019-07-27 17:36:56/tr02-epo020-acc0.9395'
    lfw_68 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__68--2019-07-27 18:01:08/tr02-epo020-acc0.8896'
    lfw_70 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__70--2019-07-27 17:44:59/tr02-epo020-acc0.9486'
    lfw_8 = '/local/home/david/Remote/PruneFramework/exp_files/lfw__8--2019-07-27 17:20:45/tr02-epo020-acc0.8630'

    task_lists = [
        {
            'name': 'lfw_27_32_42',
            'a': lfw_27,
            'b': lfw_32,
            'c': lfw_42
        }
    ]

    for set_ in task_lists:
        task_name = set_['name']
        model_path_a = set_['a']
        model_path_b = set_['b']
        model_path_c = set_['c']
        print(str(set_))

        pruning(
            task_name=task_name,
            model_path_a=model_path_a,
            model_path_b=model_path_b,
            model_path_c=model_path_c,

            cluster_set={
                'path_cluster_res': None,
                'cluster_threshold_dict': {"conv": 0.0001, "fc": 0.02},
                'cluster_layer_range': [13, 14]
            },
            pruning_set={
                'name': 'info_bottle',
                'gamma_conv': 1.,
                'gamma_fc': 20.,
                'ib_threshold_conv': 0.01,
                'ib_threshold_fc': 0.01
            },
            plan_retrain=plan_retrain
        )
