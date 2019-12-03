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

from models.model_combined import Model_Combined
from models.model import Model
from utils.mi_gpu import get_K_function
from utils.mi_gpu import kde_in_gpu
from utils.configer import load_cfg
from utils.configer import get_cfg_rdnet
from utils.logger import *
from utils.json import read_l
from utils.json import read_i
from utils.json import read_s
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle
import json
import os

global dim_list
global structure
global n_tasks


def init_tf():
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    return tf.Session(config=gpu_config)


def get_layers_output(model_path, batch_size, with_relu=True):
    cfg = load_cfg('/'.join(model_path.split('/')[:-1]) + '/cfg.ini')

    cfg.set('train', 'batch_size', str(batch_size))

    tf.reset_default_graph()
    sess = init_tf()

    model = Model(cfg)
    model.build()

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


def t2c(task_index):
    if task_index == -1:
        return 'CEN'
    else:
        return chr(ord('A') + task_index)


def extract_unrelevance_neurons(layer_output_all, n_neurons, labelixs_list, labelprobs_list, entropy_func_upper,
                                alpha_threshold):
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
        return set(neurons_list_selected)

    global n_tasks

    # 求出来和每一个部分label没有关系的点
    neurons_irrelevance_list = [
        get_irrelevance_neurons(t2c(task_index), labelixs_list[task_index], labelprobs_list[task_index]) for task_index
        in range(n_tasks)]

    # 和所有task都无关的点，可以删除的点
    neurons_union = None
    for neurons_irrelevance in neurons_irrelevance_list:
        if neurons_union is None:
            neurons_union = neurons_irrelevance
        else:
            neurons_union = neurons_union & neurons_irrelevance

    # 找到属于任意一个task的neurons
    neurons_tasks_list = list()
    for task_index in range(n_tasks):
        neurons_exclusive = None
        for index, neurons_irrelevance in enumerate(neurons_irrelevance_list):
            if index != task_index:
                if neurons_exclusive is None:
                    neurons_exclusive = neurons_irrelevance
                else:
                    neurons_exclusive = neurons_exclusive & neurons_irrelevance
        neurons_tasks_list.append(np.sort(list(neurons_exclusive - neurons_union)))

    # 找到中间的所有的神经元
    neurons_centra = set(np.arange(n_neurons)) - neurons_union
    for neurons_list in neurons_tasks_list:
        neurons_centra = neurons_centra - set(neurons_list)

    # set, list(set)
    return neurons_centra, neurons_tasks_list


def cluster_neurons(cfg, y_list, layers_output_list, cluster_threshold_dict, cluster_layer_range, path_save=None):
    global dim_list, n_tasks, structure
    # 记录每一个任务label的维度
    n_label_list = [y.shape[1] for y in y_list]

    entropy_func_upper = get_K_function()
    labelixs_list, labelprobs_list = list(), list()
    for y in y_list:
        labelixs, labelprobs = get_label_info(y)
        labelixs_list.append(labelixs)
        labelprobs_list.append(labelprobs)

    # Save cluster results
    if not os.path.exists(path_save + '/cluster_results/'):
        os.mkdir(path_save + '/cluster_results/')

    # Init for all layers to store cluster res
    cluster_res_dict = dict()
    for ind, dim in enumerate(dim_list):
        n_neurons_total = dim * n_tasks

        clu_layer_dict = dict()
        for task_index in range(n_tasks):
            clu_layer_dict[t2c(task_index)] = list()
        # CEN
        clu_layer_dict['CEN'] = np.arange(n_neurons_total).tolist()
        cluster_res_dict[structure[ind]] = clu_layer_dict

    # Output layer
    labels_str = read_s(cfg, 'task', 'labels_task')[1:-1]

    labels = dict()
    for ind, item in enumerate(labels_str.split(',')):  # [1-10]/[1,2,3]
        if '-' in item:
            s, e = [int(_) for _ in item[1:-1].split('-')]
            labels[t2c(ind)] = np.arange(s, e).tolist()
        else:
            labels[t2c(ind)] = json.loads(item)
    labels['CEN'] = list()
    cluster_res_dict[structure[-1]] = labels

    # Main loop
    for layer_index in cluster_layer_range:
        if structure[layer_index].startswith('c'):
            alpha_threshold = cluster_threshold_dict['conv']
        else:
            alpha_threshold = cluster_threshold_dict['fc']

        log_t('Cluster layer %d, alpha threshold = %f' % (layer_index, alpha_threshold))

        # Obtain output of this layer
        layer_output_list = [layers_output_list[task_index][layer_index] for task_index in range(n_tasks)]

        # 展开
        # 如果是conv层的输出的话
        if len(layer_output_list[0]) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_list = list(map(lambda x: np.max(x, axis=(1, 2)), layer_output_list))

        layer_output_all = np.concatenate(layer_output_list, axis=-1)

        # Find neurons without relevance with B and C
        neurons_cen, neurons_part_list = extract_unrelevance_neurons(layer_output_all,
                                                                     dim_list[layer_index] * n_tasks,
                                                                     labelixs_list, labelprobs_list,
                                                                     entropy_func_upper,
                                                                     alpha_threshold)

        for task_index in range(n_tasks):
            cluster_res_dict[structure[layer_index]][t2c(task_index)] = list(neurons_part_list[task_index])
        cluster_res_dict[structure[layer_index]]['CEN'] = list(neurons_cen)

    # Save the final result
    path = path_save + '/cluster_results/cluster_result_threshold-%s' % str(cluster_threshold_dict)
    pickle.dump(cluster_res_dict, open(path, 'wb'))
    return cluster_res_dict


def model_summary(cluster_res_dict):
    global dim_list, n_tasks, structure

    for ind, [name, layer] in enumerate(cluster_res_dict.items()):
        n_list = [len(layer[t2c(task_index)]) for task_index in range(n_tasks)]
        n_cen = len(layer['CEN'])

        if ind < len(dim_list):
            n_pruned = dim_list[ind] * n_tasks - np.sum(n_list) - n_cen
        else:
            n_pruned = 0

        log('Layer %s: n_tasks: %s, n_CEN: %d, n_pruned: %d' % (name, n_list, n_cen, n_pruned))


def get_cluster_res(cfg):
    path_cluster_res = cfg['cluster']['path_cluster_res']
    if path_cluster_res == 'None':
        print('[%s] Get the layer outputs of the models' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        layers_output_list, labels_list = list(), list()

        model_path_list = read_l(cfg, 'task', 'path_models')
        for model_path in model_path_list:
            layers_output, labels = get_layers_output(model_path, read_i(cfg, 'cluster', 'batch_size'))
            layers_output_list.append(layers_output)
            labels_list.append(labels)

        cluster_res_dict = cluster_neurons(cfg, labels_list, layers_output_list,
                                           read_l(cfg, 'cluster', 'cluster_threshold_dict'),
                                           read_l(cfg, 'cluster', 'cluster_layer_range'),
                                           read_s(cfg, 'path', 'path_save'))

    else:
        cluster_res_dict = pickle.load(open(path_cluster_res.replace('\"', '\''), 'rb'))

    model_summary(cluster_res_dict)

    return cluster_res_dict


def get_connection_signal(cluster_res_dict):
    global dim_list, n_tasks, structure

    signal_dict = dict()
    for name, layer_dict in cluster_res_dict.items():
        signal_layer_dict = dict()

        for task_index in range(-1, n_tasks):
            signal_layer_dict[t2c(task_index)] = not len(layer_dict[t2c(task_index)]) == 0

        signal_dict[name] = signal_layer_dict

    return signal_dict


def retrain_model(cfg, cluster_res_dict, plan_retrain):
    log_t('Loading weights of model a and b ...')
    weight_list = list()
    for model_path in read_l(cfg, 'task', 'path_models'):
        weight_list.append(pickle.load(open(model_path, 'rb')))
    log_t('Done')

    signal_list = get_connection_signal(cluster_res_dict)

    global n_tasks

    sess = init_tf()
    model = Model_Combined(cfg, n_tasks, weight_list, cluster_res_dict, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, model.test_init, -2)
    if model.pruning:
        model.get_CR(sess, cluster_res_dict)
    log_l('')

    for plan in plan_retrain:
        model.set_kl_factor(plan['kl_factor'])

        for set_ in plan['train']:
            if plan['type'] == 'normal':
                model.train(sess=sess, n_epochs=set_['n_epochs'], lr=set_['lr'], save_clean=set_['save_clean'])
            # elif plan['type'] == 'individual':
            #     model.train_individual(sess=sess, n_epochs=set_['n_epochs'], lr=set_['lr'])
        model.save_cfg()


def obtain_cfg(task_name, model_name, data_name, path_model, pruning_set, cluster_set):
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    cfg = get_cfg_rdnet(task_name, model_name, data_name, time_stamp, path_model)

    # Pruning
    cfg.set('basic', 'pruning_method', 'info_bottle')
    cfg.add_section('pruning')
    for option in pruning_set.keys():
        cfg.set('pruning', option, str(pruning_set[option]))

    # Cluster
    cfg.add_section('cluster')
    for option in cluster_set.keys():
        cfg.set('cluster', option, str(cluster_set[option]))

    return cfg


def pruning(task_name, model_name, data_name, path_model, pruning_set, cluster_set, plan_retrain):
    # Config
    cfg = obtain_cfg(task_name, model_name, data_name, path_model, pruning_set, cluster_set)

    global dim_list
    dim_list = [_ for _ in read_l(cfg, 'model', 'dimension') if _ != 0]

    global n_tasks
    n_tasks = read_i(cfg, 'task', 'n_models')

    global structure
    structure = [_ for _ in read_l(cfg, 'model', 'structure') if _.startswith('c') or _.startswith('f') and _ != 'fla']

    # Cluster
    cluster_res_dict = get_cluster_res(cfg)

    # Save cfg
    with open(cfg['path']['path_cfg'], 'w') as file:
        cfg.write(file)

    # Retrain
    retrain_model(cfg, cluster_res_dict, plan_retrain)
