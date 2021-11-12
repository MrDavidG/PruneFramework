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


def val_model(path_model):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('path', 'path_load', path_model)

    # not record
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
        ''
    ]

    for path in path_models:
        val_model(path)
