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

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def val_rdnet(path_model):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)

    # 不写入log
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

    # model.save_weight_clean(sess, '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN2')

    #

if __name__ == '__main__':
    paths = [
        # rdnet
        # '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-09_15-45-23/tr03-epo010-cr0.0207-acc0.9610',

        # baseline
        # '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-09_08-27-44/tr03-epo010-cr0.0243-acc0.9551',

        # rdnet lfw15
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_lfw15-vgg128-rdnet-2019-12-09_03-58-19/tr01-epo010-cr0.0543-acc0.8513',
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f412'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_c23'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22]'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22, 18]'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22, 18, 4]'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22, 18, 4, 26]'
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22, 18, 4, 26, 14]',
        # '/local/home/david/Remote/PruneFramework/exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_09-41-59-scenario1/tr02-epo010-cr0.0196-acc0.9619-CLEAN-del_f4[12, 22, 18, 4, 26, 14, 13]'
    ]

    for path_model in paths:
        val_rdnet(path_model)
