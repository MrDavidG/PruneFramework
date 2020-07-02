# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: prune_cvpr_rdnet
@time: 2020/5/8 12:28 下午

Description. 
"""

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

from models.model_combined_gate import Model_Combined
from utils.configer import get_cfg_rdnet
from utils.logger import *
from utils.json import read_l
from utils.json import read_i
from utils.json import read_s
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle

global dim_list
global structure
global n_tasks


def init_tf():
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    return tf.Session(config=gpu_config)


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
            model.build()
            sess.run(tf.global_variables_initializer())

            log_t('Pre test')
            model.eval_once(sess, model.test_init, -2)
            log_t('')

            path = model.prune(sess, n_epoch=epoch, lr=plan['lr'])

            cfg['path']['path_load'] = path

            sess.close()
            weight_list = None

    # fine tuning
    tf.reset_default_graph()
    sess = init_tf()

    model = Model_Combined(cfg, n_tasks, weight_list, cluster_res_dict, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_t('Pre test')
    model.eval_once(sess, model.test_init, -2)
    log_t('')

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
    global dim_list
    dim_list = [_ for _ in read_l(cfg, 'model', 'dimension') if _ != 0]

    global n_tasks
    n_tasks = read_i(cfg, 'task', 'n_models')

    global structure
    structure = [_ for _ in read_l(cfg, 'model', 'structure') if _.startswith('c') or _.startswith('f') and _ != 'fla']


def get_cluster_res(cfg):
    path_cluster_res = read_s(cfg, 'cluster', 'path_cluster_res')
    if path_cluster_res == 'None':
        print('Cannot find the cluster results!')
    else:
        cluster_res_dict = pickle.load(open(path_cluster_res.replace('\"', '\''), 'rb'))
        path = path_cluster_res

    cfg.set('path', 'path_cluster_res', path)

    model_summary(cluster_res_dict)

    return cluster_res_dict


def prune(task_name, model_name, data_name, path_model, cluster_set, plan_prune, plan_fine, suffix=None):
    # Config
    cfg = obtain_cfg(task_name, model_name, data_name, path_model, cluster_set, suffix)

    # Global
    init_global(cfg)

    # Cluster
    cluster_res_dict = get_cluster_res(cfg)

    # Save cfg
    with open(read_s(cfg, 'path', 'path_cfg'), 'w') as file:
        cfg.write(file)

    # Retrain
    prune_model(cfg, cluster_res_dict, plan_prune, plan_fine)
