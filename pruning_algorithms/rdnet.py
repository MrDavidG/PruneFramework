# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: rdnet.py
@time: 2019-07-10 16:00

Description. 
"""
import sys
from typing import Optional, Any

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.vgg_combine import VGG_Combined
from models.vgg_celeba_512 import VGGNet
from utils.mi_gpu import get_K_function
from utils.mi_gpu import kde_in_gpu
from utils.configer import load_cfg
from utils.configer import get_cfg
from utils.logger import *
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'
# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

global dim_list


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


def cluster_neurons(y_a, y_b, layers_output_list_a, layers_output_list_b, cluster_threshold_dict,
                    cluster_layer_range=np.arange(15), path_save=None):
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
    for layer_index in cluster_layer_range:
        if layer_index < 13:
            alpha_threshold = cluster_threshold_dict['conv']
        else:
            alpha_threshold = cluster_threshold_dict['fc']

        log_t('Cluster layer %d, alpha threshold = %f' % (layer_index, alpha_threshold))

        # Obtain output of this layer
        layer_output_a = layers_output_list_a[layer_index]
        layer_output_b = layers_output_list_b[layer_index]

        # 展开
        if len(layer_output_a.shape) == 4 and len(layer_output_b.shape) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_a, layer_output_b = list(
                map(lambda x: np.max(x, axis=(1, 2)), [layer_output_a, layer_output_b]))

        layer_output_all = np.concatenate((layer_output_a, layer_output_b), axis=-1)

        #
        F_A = cluster_res_list[layer_index]['AB']
        F_B = cluster_res_list[layer_index]['AB']

        min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_A, cluster_res_list[layer_index]['A'],
                                                      labelixs_b, labelprobs_b, entropy_func_upper)

        log_t('MI with Y_B, No.1: Min_index_neuron=%d,  mi_min=%f' % (min_index_neuron, mi_min))

        F_A.remove(min_index_neuron)
        cluster_res_list[layer_index]['A'].append(min_index_neuron)
        count_a = 1

        while mi_min <= alpha_threshold:
            # Traverse neurons in F_A to find neuron with the minimal mi with Y_B
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_A, cluster_res_list[layer_index]['A'],
                                                          labelixs_b, labelprobs_b, entropy_func_upper)
            count_a += 1
            log_t('MI with Y_B, No.%d: Min_index_neuron=%d,  mi_min=%f' % (count_a, min_index_neuron, mi_min))

            F_A.remove(min_index_neuron)
            cluster_res_list[layer_index]['A'].append(min_index_neuron)

        min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_B, cluster_res_list[layer_index]['B'],
                                                      labelixs_a, labelprobs_a, entropy_func_upper)
        log('MI with Y_A, No. 1: Min_index_neuron=%d,  mi_min=%f' % (min_index_neuron, mi_min))
        count_b = 1

        F_B.remove(min_index_neuron)
        cluster_res_list[layer_index]['B'].append(min_index_neuron)

        while min <= alpha_threshold:
            min_index_neuron, mi_min = argmin_marginal_mi(layer_output_all, F_B, cluster_res_list[layer_index]['B'],
                                                          labelixs_a, labelprobs_a, entropy_func_upper)
            count_b += 1
            log('MI with Y_A, No.%d: Min_index_neuron=%d,  mi_min=%f' % (count_b, min_index_neuron, mi_min))
            cluster_res_list[layer_index]['B'].append(min_index_neuron)
            F_B.remove(min_index_neuron)

        cluster_res_list[layer_index]['A'].sort()
        cluster_res_list[layer_index]['B'].sort()

        # Make sure there is not overlap between a, b and ab
        cluster_res_list[layer_index]['AB'] = list(
            set(cluster_res_list[layer_index]['AB']) - set(cluster_res_list[layer_index]['A']) - set(
                cluster_res_list[layer_index]['B']))

        union = set(cluster_res_list[layer_index]['A']) & set(cluster_res_list[layer_index]['B'])
        cluster_res_list[layer_index]['A'] = list(set(cluster_res_list[layer_index]['A']) - union)
        cluster_res_list[layer_index]['B'] = list(set(cluster_res_list[layer_index]['B']) - union)

        # 保存每一层的结果，防止丢失
        path = path_save + '/cluster_results/cluster_layer-%d_threshold-%f' % (layer_index, alpha_threshold)
        pickle.dump(cluster_res_list[layer_index], open(path, 'wb'))

    # Save the final result
    path = path_save + '/cluster_results/cluster_result_threshold-%s' % str(cluster_threshold_dict)
    pickle.dump(cluster_res_list, open(path, 'wb'))
    return cluster_res_list


def model_summary(cluster_res_list):
    global dim_list
    for layer_index, layer_clusters in enumerate(cluster_res_list):
        num_A = len(layer_clusters['A'])
        num_AB = len(layer_clusters['AB'])
        num_AB_from_a = (np.array(layer_clusters['AB']) < dim_list[layer_index]).sum()
        num_AB_from_b = num_AB - num_AB_from_a
        num_B = len(layer_clusters['B'])
        num_pruned = dim_list[layer_index] * 2 - num_A - num_B - num_AB

        print('Layer %d: num_A=%d   |   num_AB=%d(%d:%d)   |   num_B=%d |   num_pruned=%d' % (
            layer_index + 1, num_A, num_AB, num_AB_from_a, num_AB_from_b, num_B, num_pruned))


def get_cluster_res(cfg):
    path_cluster_res = cfg['cluster']['path_cluster_res']
    if path_cluster_res is None:
        layers_output_list_a, labels_a = get_layers_output(model_path=cfg['basic']['model_a'])
        layers_output_list_b, labels_b = get_layers_output(model_path=cfg['basic']['model_b'])

        cluster_res_list = cluster_neurons(labels_a, labels_b, layers_output_list_a, layers_output_list_b,
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


def retrain_model(cfg, cluster_res_list):
    weight_a = pickle.load(open(cfg['basic']['model_a'], 'rb'))
    weight_b = pickle.load(open(cfg['basic']['model_b'], 'rb'))

    signal_list = get_connection_signal(cluster_res_list)

    sess = init_tf()

    model = VGG_Combined(cfg, weight_a, weight_b, cluster_res_list, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, model.test_init, -2)
    model.get_CR(sess, cluster_res_list)


def obtain_cfg(task_name, model_path_a, model_path_b, pruning_set, cluster_set):
    # cfg
    time_stamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    cfg = get_cfg(task_name, time_stamp)

    cfg.set('basic', 'model_a', model_path_a)
    cfg.set('basic', 'model_b', model_path_b)

    cfg.add_section('pruning')
    for option in pruning_set.keys():
        cfg.set('pruning', option, str(pruning_set[option]))

    cfg.add_section('cluster')
    for option in cluster_set.keys():
        cfg.set('cluster', option, str(cluster_set[option]))

    return cfg


def pruning(task_name, model_path_a, model_path_b, pruning_set, cluster_set):
    cfg = obtain_cfg(task_name, model_path_a, model_path_b, pruning_set, cluster_set)
    global dim_list
    dim_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512,
                cfg['data'].getint('n_classes') / 2]

    # Cluster
    cluster_res_list = get_cluster_res(cfg)

    # Retrain
    retrain_model(cfg, cluster_res_list)


if __name__ == '__main__':
    pruning(task_name='deepfashion',
            model_path_a='',
            model_path_b='',
            cluster_set={
                'path_cluster_res': None,
                'cluster_threshold_dict': {'conv': 0.2, 'fc': 8},
                'cluster_layer_range': [13, 14]
            },
            pruning_set={
                'gamma_conv': -1,
                'gamma_fc': 8,
                'ib_threshold_conv': 0.01,
                'ib_threshold_fc': 0.01
            }
            )
