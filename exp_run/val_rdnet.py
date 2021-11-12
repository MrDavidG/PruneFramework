# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: test_rdnet
@time: 2019-12-04 14:21

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from pruning_algorithms.rdnet_multi import get_connection_signal
from pruning_algorithms.rdnet_multi import init_tf, init_global
from models.model_combined import Model_Combined
from utils.configer import load_cfg
from utils.logger import logger
from utils.logger import log_l
from utils.json import read_s
from utils.json import read_i

import tensorflow as tf

import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def val_rdnet(path_model):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)

    # not log
    logger.record_log = False

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


if __name__ == '__main__':
    paths = [
        ''
    ]

    for path_model in paths:
        val_rdnet(path_model)
