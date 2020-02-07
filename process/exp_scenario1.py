# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: exp_scenario1
@time: 2019/12/20 4:01 下午

Description. 
"""

from models.model_combined import Model_Combined
from utils.logger import logger, log_l
from utils.configer import load_cfg
from utils.json import read_i
from utils.json import read_s
from pruning_algorithms.rdnet_multi import get_connection_signal
from pruning_algorithms.rdnet_multi import init_tf, init_global
from utils.mi_gpu import kde_in_gpu
from utils.mi_gpu import get_K_function
import pickle

import tensorflow as tf
import numpy as np

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


def exp(path_model, layer_name):
    # 不写入log
    logger.record_log = False

    # 需要先训练一个rdnet，只做vib，不做cluster
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)
    cfg.set('train', 'batch_size', str(128))

    cluster_res_dict = pickle.load(open(read_s(cfg, 'path', 'path_cluster_res'), 'rb'))

    init_global(cfg)
    signal_list = get_connection_signal(cluster_res_dict)

    tf.reset_default_graph()
    sess = init_tf()
    model = Model_Combined(cfg, read_i(cfg, 'task', 'n_models'), None, cluster_res_dict, signal_list)
    model.build()
    sess.run(tf.global_variables_initializer())

    sess.run(model.test_init)
    y_A, y_B, layer_output = sess.run(
        [tf.gather(model.Y, model.Y_list[0], axis=-1),
         tf.gather(model.Y, model.Y_list[1], axis=-1),
         model.get_layer_by_name(layer_name).layer_output], {model.is_training: False})

    # 找一个只对a有作用，对b没作用的
    # init
    global entropy_func_upper
    entropy_func_upper = get_K_function()

    # 卷积层特殊处理下
    if len(layer_output.shape) == 4:
        layer_output = np.max(layer_output, axis=(1, 2))

    for i in range(layer_output.shape[-1]):
        mi_A = cal_mi(y_A, layer_output[..., [i]])
        mi_B = cal_mi(y_B, layer_output[..., [i]])
        print(i, mi_A, mi_B, mi_A - mi_B)


def del_neurons(path_weight, name, name_next, inds):
    w = pickle.load(open(path_weight, 'rb'))

    mask = np.ones(shape=w[name + '/CEN/b'].shape, dtype=np.bool)
    for ind in inds:
        mask[ind] = False

    w[name + '/CEN/w'] = w[name + '/CEN/w'][..., mask]
    w[name + '/CEN/b'] = w[name + '/CEN/b'][mask]
    w[name + '_vib/CEN/mu'] = w[name + '_vib/CEN/mu'][mask]
    w[name + '_vib/CEN/logD'] = w[name + '_vib/CEN/logD'][mask]

    w[name_next + '/A/w'] = w[name_next + '/A/w'][..., mask, :]
    w[name_next + '/B/w'] = w[name_next + '/B/w'][..., mask, :]

    path_save = '%s-del_%s%s' % (path_weight, name, str(inds))
    pickle.dump(w, open(path_save, 'wb'))


def del_filter(path_weight, name, name_next, ind):
    w = pickle.load(open(path_weight, 'rb'))

    mask = np.ones(shape=w[name + '/CEN/b'].shape, dtype=np.bool)
    mask[ind] = False

    # c->f
    times = w[name_next + '/CEN/w'].shape[0] / w[name + '/CEN/w'].shape[-1]

    w[name + '/CEN/w'] = w[name + '/CEN/w'][..., mask]
    w[name + '/CEN/b'] = w[name + '/CEN/b'][mask]
    w[name + '_vib/CEN/mu'] = w[name + '_vib/CEN/mu'][mask]
    w[name + '_vib/CEN/logD'] = w[name + '_vib/CEN/logD'][mask]

    mask = np.concatenate([mask for _ in range(int(times))], axis=0)

    w[name_next + '/CEN/w'] = w[name_next + '/CEN/w'][..., mask, :]

    path_save = '%s-del_%s%s' % (path_weight, name, str(ind))
    pickle.dump(w, open(path_save, 'wb'))


if __name__ == '__main__':
    # exp(
    #     path_model='../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN',
    #     layer_name='c2/CEN'
    # )

    del_neurons(
        '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN',
        'f4',
        'f5',
        [12, 22, 18, 4, 26, 14, 13])

    # del_filter(
    #     '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN',
    #     'c2',
    #     'f3',
    #     3
    # )
