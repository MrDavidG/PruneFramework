# encoding: utf-8

from models.model_combined_res_gate import Model_Combined
from models.model_res import Model
from layers.stage_layer import StageLayer
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
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
global dim_dict
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
    sess.run(model.test_init)

    # block_c1, block输出
    layers_name = list()
    layers_output_tf = list()
    for layer in model.layers:
        if type(layer) == ConvLayer:
            layers_name.append(layer.layer_name)
            layers_output_tf.append(layer.layer_output)
        elif type(layer) == StageLayer:
            for idx, block in enumerate(layer.blocks):
                # blcok 第一个conv层输出
                layers_name.append('%s/b%d/c1' % (layer.layer_name, idx + 1))
                layers_output_tf.append(block[0].layer_output)
                # 该block 的输出
                layers_name.append('%s/b%d' % (layer.layer_name, idx + 1))
                layers_output_tf.append(layer.blocks_output[idx])
        else:
            pass

    layers_output, labels = sess.run([layers_output_tf] + [model.Y], feed_dict={model.is_training: False})

    if with_relu:
        layers_output = [x * (x > 0) for x in layers_output]

    return dict(zip(layers_name, layers_output)), labels


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
        # 记录哪些位置是1,哪些位置是-1
        labelixs = {}
        labelixs[0] = y[:, label_index] == -1
        labelixs[1] = y[:, label_index] == 1
        labelixs_x.append(labelixs)

        # 各个维度为1的概率
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
        # 保存的是与该任务无关的neurons
        neurons_list_selected = list()
        count = 1

        # F用来保留和task有关系的神经元
        # 也就是F需要进行变化
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
    global dim_dict, n_tasks, structure
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
    # TODO: 记录cluster结果的方式需要变化
    cluster_res_dict = dict()
    for layer_name in layers_output_list[0].keys():
        layer_dim = dim_dict[layer_name[:2]]
        n_neuorns_total = layer_dim * n_tasks
        clu_layer_dict = dict()
        for task_index in range(n_tasks):
            clu_layer_dict[t2c(task_index)] = list()
        # CEN
        clu_layer_dict['CEN'] = np.arange(n_neuorns_total).tolist()

        cluster_res_dict[layer_name] = clu_layer_dict

    # Output layer
    labels_str = read_s(cfg, 'task', 'labels_task').strip().replace('],[', ']$[')[1:-1]

    labels = dict()
    for ind, item in enumerate(labels_str.split('$')):  # [1-10]/[1,2,3]
        if '-' in item:
            s, e = [int(_) for _ in item[1:-1].split('-')]
            labels[t2c(ind)] = np.arange(s, e).tolist()
        else:
            labels[t2c(ind)] = json.loads(item)
    labels['CEN'] = list()
    cluster_res_dict[structure[-1]] = labels

    # Main loop
    for layer_name in cluster_layer_range:
        alpha_threshold = cluster_threshold_dict[layer_name[:2]]

        log_t('Cluster layer %s, alpha threshold = %f' % (layer_name, alpha_threshold))

        # 所有task的model在这一层的输出
        layer_output_list = [layers_output_list[task_index][layer_name] for task_index in range(n_tasks)]

        # 如果是conv层的输出的话
        if len(layer_output_list[0]) == 4:
            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            layer_output_list = list(map(lambda x: np.max(x, axis=(1, 2)), layer_output_list))

        layer_output_all = np.concatenate(layer_output_list, axis=-1)

        neurons_cen, neurons_part_list = extract_unrelevance_neurons(layer_output_all,
                                                                     dim_dict[layer_name[:2]] * n_tasks,
                                                                     labelixs_list, labelprobs_list,
                                                                     entropy_func_upper,
                                                                     alpha_threshold)

        # Save results
        for task_index in range(n_tasks):
            cluster_res_dict[layer_name][t2c(task_index)] = list(neurons_part_list[task_index])
        cluster_res_dict[layer_name]['CEN'] = list(neurons_cen)

        path = path_save + '/cluster_results/cluster_results_layer-%s' % str(layer_name.replace('/', '_'))
        pickle.dump(cluster_res_dict, open(path, 'wb'))

    # Save the final result
    path = path_save + '/cluster_results/cluster_result_threshold-%s' % str(cluster_threshold_dict).replace('\'',
                                                                                                            '').replace(
        '\"', '')
    pickle.dump(cluster_res_dict, open(path, 'wb'))
    return cluster_res_dict, path


def model_summary(cluster_res_dict):
    global dim_dict, n_tasks, structure

    for ind, [name, layer] in enumerate(cluster_res_dict.items()):
        n_list = [len(layer[t2c(task_index)]) for task_index in range(n_tasks)]
        n_cen = len(layer['CEN'])

        if name != structure[-1]:
            n_pruned = dim_dict[name[:2]] * n_tasks - np.sum(n_list) - n_cen
        else:
            n_pruned = 0

        log('Layer %s: n_tasks: %s, n_CEN: %d, n_pruned: %d' % (name, n_list, n_cen, n_pruned))


def get_cluster_res(cfg):
    path_cluster_res = read_s(cfg, 'cluster', 'path_cluster_res')
    if path_cluster_res == 'None':
        print('[%s] Get the layer outputs of the models' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        layers_output_list, labels_list = list(), list()

        model_path_list = read_l(cfg, 'task', 'path_models')
        for model_path in model_path_list:
            layers_output, labels = get_layers_output(model_path, read_i(cfg, 'cluster', 'batch_size'))
            layers_output_list.append(layers_output)
            labels_list.append(labels)

        cluster_layer_range = read_l(cfg, 'cluster', 'cluster_layer_range')
        if cluster_layer_range is None:
            cluster_layer_range = list(layers_output_list[0].keys())

        cluster_res_dict, path = cluster_neurons(cfg, labels_list, layers_output_list,
                                                 read_l(cfg, 'cluster', 'cluster_threshold_dict'),
                                                 cluster_layer_range,
                                                 read_s(cfg, 'path', 'path_save'))

    else:
        cluster_res_dict = pickle.load(open(path_cluster_res.replace('\"', '\''), 'rb'))
        path = path_cluster_res

    cfg.set('path', 'path_cluster_res', path)

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


def prune_model(cfg, cluster_res_dict, plan_prune, plan_fine):
    log_t('Loading weights from models a and b ...')

    weight_list = list()
    for model_path in read_l(cfg, 'task', 'path_models'):
        weight_list.append(pickle.load(open(model_path, 'rb')))
    log_t('Done')

    # 获取各个单元连接信号
    signal_list = get_connection_signal(cluster_res_dict)

    global n_tasks

    # prune
    for plan in plan_prune:
        for epoch in range(plan['n_epochs']):
            # 重新加载模型，然后进行剪枝
            log_l('Iteration %d, lr %f' % (epoch, plan['lr']))

            tf.reset_default_graph()
            sess = init_tf()

            model = Model_Combined(cfg, n_tasks, weight_list, cluster_res_dict, signal_list)
            sess.run(tf.global_variables_initializer())

            model.get_cr_task(sess)

            # log_t('Pre test')
            # model.eval_once(sess, model.test_init, -2)
            # log_t('')

            path = model.prune(sess, n_epoch=epoch, lr=plan['lr'])

            cfg['path']['path_load'] = path

            sess.close()
            weight_list = None

    # fine tuning
    tf.reset_default_graph()
    sess = init_tf()

    model = Model_Combined(cfg, n_tasks, weight_list, cluster_res_dict, signal_list)
    sess.run(tf.global_variables_initializer())

    log_t('Pre test')
    model.eval_once(sess, model.test_init, -2)
    log_t('')

    model.get_cr_task(sess)

    for plan in plan_fine:
        for epoch in range(plan['n_epochs']):
            log_l('FINE TUNING %d, LR %.4f' % (epoch, plan['lr']))

            model.fine(sess, n_epoch=epoch, lr=plan['lr'])

            model.eval_once(sess, model.test_init, -2)
            log_l('')

    model.save_cfg()


def obtain_cfg(task_name, model_name, data_name, path_model, cluster_set, suffix=None):
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    cfg = get_cfg_rdnet(task_name, model_name, data_name, time_stamp, path_model, suffix=suffix)

    # Cluster
    cfg.add_section('cluster')
    for option in cluster_set.keys():
        cfg.set('cluster', option, str(cluster_set[option]))

    return cfg


def init_global(cfg):
    global dim_list, dim_dict
    # dim_list = [_ for _ in read_l(cfg, 'model', 'dimension') if _ != 0]
    dim_dict = dict(zip(read_l(cfg, 'model', 'structure'), read_l(cfg, 'model', 'dimension')))

    global n_tasks
    n_tasks = read_i(cfg, 'task', 'n_models')

    global structure
    structure = read_l(cfg, 'model', 'structure')


def pruning(task_name, model_name, data_name, path_model, cluster_set, plan_prune, plan_fine, suffix=None,
            batch_size=None):
    # Config
    cfg = obtain_cfg(task_name, model_name, data_name, path_model, cluster_set, suffix)

    if batch_size is not None:
        cfg.set('train', 'batch_size', str(batch_size))

    # Global
    init_global(cfg)

    # Cluster
    cluster_res_dict = get_cluster_res(cfg)

    # Save cfg
    with open(read_s(cfg, 'path', 'path_cfg'), 'w') as file:
        cfg.write(file)

    # Retrain
    prune_model(cfg, cluster_res_dict, plan_prune, plan_fine)
