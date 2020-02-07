# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: exp_scenario2
@time: 2019/12/23 2:12 下午

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from utils.logger import logger
from utils.logger import log_l
from utils.configer import load_cfg
from utils.json import read_i, read_s
from pruning_algorithms.rdnet_multi import get_connection_signal
from pruning_algorithms.rdnet_multi import init_tf, init_global
from models.model_combined import Model_Combined
from utils.mi_gpu import kde_in_gpu
from utils.mi_gpu import get_K_function

from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

global entropy_func_upper


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


def cal_mi(lab_a, lab_b):
    """
    Calculate the mi between lab_a and lab_b
    Args:
        lab_a: [batch_size, dim_a]
        lab_b: [batch_size, dim_b]

    Returns:
        mi, a float-point number
    """
    global entropy_func_upper
    labelixs_a, labelprobs_a = get_label_info(lab_a)
    _, mi = kde_in_gpu(lab_b, labelixs_a, labelprobs_a, entropy_func_upper)
    return mi


def create_cluster_res():
    res = dict()

    # res['c1'] = {
    #     'A': [_ for _ in range(0, 4)],
    #     'CEN': [_ for _ in range(4, 8)],
    #     'B': [_ for _ in range(8, 12)]
    # }

    res['c1'] = {
        'A': [],
        'CEN': [_ for _ in range(0, 12)],
        'B': []
    }

    # res['c2'] = {
    #     'A': [_ for _ in range(0, 10)],
    #     'CEN': [_ for _ in range(10, 22)],
    #     'B': [_ for _ in range(22, 32)]
    # }

    res['c2'] = {
        'A': [],
        'CEN': [_ for _ in range(0, 32)],
        'B': []
    }

    # res['f3'] = {
    #     'A': [_ for _ in range(0, 80)],
    #     'CEN': [_ for _ in range(80, 160)],
    #     'B': [_ for _ in range(160, 240)]
    # }

    res['f3'] = {
        'A': [],
        'CEN': [_ for _ in range(0, 240)],
        'B': []
    }

    # res['f4'] = {
    #     'A': [_ for _ in range(0, 56)],
    #     'CEN': [_ for _ in range(56, 84)] + [_ for _ in range(140, 168)],
    #     'B': [_ for _ in range(84, 140)]
    # }

    # res['f4'] = {
    #     'A': [_ for _ in range(0, 56)],
    #     'CEN': [_ for _ in range(56, 112)],
    #     'B': [_ for _ in range(112, 168)]
    # }

    res['f4'] = {
        'A': [],
        'CEN': [_ for _ in range(0, 168)],
        'B': []
    }

    res['f5'] = {
        'A': [_ for _ in range(0, 5)],
        'CEN': list(),
        'B': [_ for _ in range(5, 10)]
    }

    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(time_stamp)
    os.mkdir('../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-%s-scenario2' % time_stamp)
    os.mkdir('../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-%s-scenario2/cluster_results' % time_stamp)

    pickle.dump(res, open(
        '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-%s-scenario2/cluster_results/cluster_result' % time_stamp,
        'wb'))


def exp(path_model, layer_name, suffix, a=-1, b=-1):
    logger.record_log = False

    # 需要先训练一个rdnet，只做vib，不做cluster
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)
    cfg.set('train', 'batch_size', str(1280))

    cluster_res_dict = pickle.load(open(read_s(cfg, 'path', 'path_cluster_res'), 'rb'))

    init_global(cfg)
    signal_list = get_connection_signal(cluster_res_dict)

    tf.reset_default_graph()
    sess = init_tf()
    model = Model_Combined(cfg, read_i(cfg, 'task', 'n_models'), None, cluster_res_dict, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    sess.run(model.test_init)
    y_A, y_B = sess.run([model.get_layer_by_name(layer_name + '/A').layer_output,
                         model.get_layer_by_name(layer_name + '/B').layer_output], {model.is_training: False})

    # init
    global entropy_func_upper
    entropy_func_upper = get_K_function()

    # 寻找相关值最大的一组，结果为0，44
    # 在vib的结果里面为14，53
    max_mi = 0
    mi_a, mi_b = a, b
    if a == -1:
        for ind_a in [7, 8, 14, 21, 26]:
            y_a = y_A[..., ind_a]
            for ind_b in [8, 28, 33, 41, 53]:
                y_b = y_B[..., ind_b]

                # mi = cal_mi(y_a, y_b)
                mi = np.corrcoef(y_a, y_b)[0][1]
                if mi > max_mi:
                    max_mi = mi
                    mi_a = ind_a
                    mi_b = ind_b
                    print(ind_a, ind_b, mi)
    print(max_mi)

    y_a = y_A[..., mi_a]
    y_b = y_B[..., mi_b]

    plt.scatter(y_a, y_b, marker='o')

    plt.xlabel('Outputs of the selected task-$A$-exclusive neuron')
    plt.ylabel('Outputs of the selected task-$B$-exclusive neuron')

    plt.savefig(
        '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/scatter_replace_a' + str(
            mi_a) + '_b' + str(mi_b) + '_' + suffix + '.pdf')

    pickle.dump({'a': y_a, 'b': y_b}, open(
        '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/xy_array' + str(
            mi_a) + '-' + str(mi_b) + '_' + suffix, 'wb'))


def vib(path_model):
    logger.record_log = False

    # 需要先训练一个rdnet，只做vib，不做cluster
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)
    cfg.set('train', 'batch_size', str(32))
    cfg.set('basic', 'pruning_method', 'info_bottle')

    cfg.set('pruning', 'gamma_conv', str(0.1))
    cfg.set('pruning', 'gamma_fc', str(30.))
    cfg.set('pruning', 'kl_mult', str([0.125, 0, 0.7, 0, 0, 10., 8.3]))

    cluster_res_dict = pickle.load(open(read_s(cfg, 'path', 'path_cluster_res'), 'rb'))

    init_global(cfg)
    signal_list = get_connection_signal(cluster_res_dict)

    tf.reset_default_graph()
    sess = init_tf()
    model = Model_Combined(cfg, read_i(cfg, 'task', 'n_models'), None, cluster_res_dict, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, model.test_init, -2)
    if model.pruning:
        model.get_CR(sess, cluster_res_dict)
    log_l('')

    for plan in [
        {'kl_factor': 3e-6,
         'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        {'kl_factor': 1e-7,
         'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
    ]:
        model.set_kl_factor(plan['kl_factor'])

        for set_ in plan['train']:
            model.train(sess=sess, n_epochs=set_['n_epochs'], lr=set_['lr'], type=set_.get('type', 'normal'),
                        save_clean=set_.get('save_clean', False))

    # 获取mask的结果信息
    mask_A = sess.run(model.get_layer_by_name('f4_vib/A').get_mask(0.01))
    mask_B = sess.run(model.get_layer_by_name('f4_vib/B').get_mask(0.01))

    list_a = list()
    for ind, _ in enumerate(mask_A):
        if _ == 1.:
            list_a.append(ind)
            print('A', ind)

    print(list_a)

    list_b = list()
    for ind, _ in enumerate(mask_B):
        if _ == 1.:
            list_b.append(ind)
            print('B', ind)

    print(list_b)


def replace(path_model, layer_name, ind_a, ind_b):
    wd = pickle.load(open(path_model, 'rb'))

    wd[layer_name + '/B/w'][..., ind_a] = wd[layer_name + '/A/w'][..., ind_b]
    wd[layer_name + '/B/b'][ind_a] = wd[layer_name + '/A/b'][ind_b]

    pickle.dump(wd, open(path_model + '-replace_a%d_b%d' % (ind_a, ind_b), 'wb'))


if __name__ == '__main__':
    # create_cluster_res()

    # exp(
    #     '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580',
    #     'f4'
    # )

    # vib(
    #     '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580')

    # exp(
    #     '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr02-epo010-cr0.0348-acc0.9564',
    #     # '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580',
    #     'f4'
    # )

    # 这个是替换的实验
    # replace(
    #     '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580',
    #     'f4',
    #     14,
    #     53
    # )
    # vib('../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580-replace_a14_b53')
    # exp(
    #     # before
    #     '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr01-epo010-acc0.9580-replace_a14_b53',
    #     # vib
    #     # '../exp_files/rdnet_fashionmnist_scenario2-rdnet_lenet5-rdnet-2019-12-23_13-45-48-scenario2/tr02-epo010-cr0.0313-acc0.9572',
    #     'f4'
    # )

    # f4，重新用一个单一的task来模拟两个task的实验
    # create_cluster_res()
    # exp(
    #     '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-acc0.9640',
    #     'f4'
    # )
    # vib(
    #     '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-acc0.9640')
    # 保存vib之后21，28两个神经元的分布情况
    # exp('../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-cr0.0141-acc0.9539','f4')
    # exp('../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-acc0.9640', 'f4')

    # 重复试验
    # vib('/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task')
    exp(
        '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-cr0.0183-acc0.9524',
        'f4',
        'vib',
        8, 8
    )

    # exp(
    #     '../exp_files/rdnet_fashionmnist_scenario2_-rdnet_lenet5-rdnet-2019-12-26_04-48-32-scenario2_same_task/tr01-epo010-acc0.9640',
    #     'f4',
    #     'before',
    #     8, 8
    # )
