# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: val_model
@time: 2019/12/11 1:23 下午

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from utils.configer import load_cfg
from utils.logger import logger
from utils.logger import log_l
from models.model import Model

import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def val_model(path_model):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)

    # 不写入log
    logger.record_log = False

    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    model = Model(cfg)
    model.build()
    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, model.test_init, -2)
    if model.pruning:
        model.get_CR(sess)
    log_l('')


if __name__ == '__main__':
    path_models = [
        # vib_a
        # '/local/home/david/Remote/PruneFramework/exp_files/fashionmnist_a-lenet5-vib-2019-12-09_12-34-31/tr01-epo010-cr0.0447-fl0.1177-acc0.9590',
        # vib_b
        # '/local/home/david/Remote/PruneFramework/exp_files/fashionmnist_b-lenet5-vib-2019-12-10_15-21-57/tr01-epo010-cr0.0306-fl0.0336-acc0.9625',

        '/local/home/david/Remote/PruneFramework/exp_files/lfw15_4-vgg128-vib-2019-12-13_09-14-19/tr01-epo010-cr0.0008-fl0.0016-acc0.8661'
    ]

    for path in path_models:
        val_model(path)
