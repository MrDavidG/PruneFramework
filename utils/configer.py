# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: configer.py.py
@time: 2019-03-27 10:58

Description here.
"""

from utils.logger import logger
from utils.logger import log_t
from utils.json import read_l

import configparser
import os


def get_cfg_rdnet(task_name, model_name, data_name, time_stamp, path_model=None, file_cfg_global='global.cfg',
                  suffix=None):
    cfg_base, cfg_task, cfg_data, cfg_model = load_cfg('../config/' + file_cfg_global), \
                                              load_cfg('../config/task.cfg'), \
                                              load_cfg('../config/data.cfg'), \
                                              load_cfg('../config/model.cfg')

    # 以global为主体
    cfg_base['basic']['time_stamp'] = time_stamp

    # task
    combine_cfg(cfg_base, cfg_task, 'task', task_name)
    # data
    combine_cfg(cfg_base, cfg_data, 'data', data_name)
    # model
    combine_cfg(cfg_base, cfg_model, 'model', model_name)

    cfg_base['path']['path_load'] = str(path_model)
    if suffix is None:
        cfg_base['path']['path_save'] += '%s-%s-rdnet-%s' % (task_name, model_name, time_stamp)
    else:
        cfg_base['path']['path_save'] += '%s-%s-rdnet-%s-%s' % (task_name, model_name, time_stamp, suffix)
    cfg_base['path']['path_cfg'] = cfg_base['path']['path_save'] + '/cfg.ini'
    cfg_base['path']['path_log'] = cfg_base['path']['path_save'] + '/log.log'
    cfg_base['path']['path_dataset'] = cfg_base['path']['path_dataset'] + data_name + '/'

    # activation要变化，同时还需要指出opt
    activation_task = cfg_base['task']['activation']
    activation_model = read_l(cfg_base, 'model', 'activation')
    cfg_base.set('model', 'activation', str(activation_model + [activation_task]))

    # 登录log
    logger(cfg_base['path']['path_log'])

    # create dir
    if not os.path.exists(cfg_base['path']['path_save']):
        os.mkdir(cfg_base['path']['path_save'])
        log_t('Create directory %s' % cfg_base['path']['path_save'])

    return cfg_base


def get_cfg(task_name, model_name, data_name, time_stamp, path_model=None, file_cfg_global='global.cfg', suffix=None):
    # global, task, data, model
    cfg_base, cfg_task, cfg_data, cfg_model = load_cfg('../config/' + file_cfg_global), \
                                              load_cfg('../config/task.cfg'), \
                                              load_cfg('../config/data.cfg'), \
                                              load_cfg('../config/model.cfg')

    # 以global为主体
    cfg_base['basic']['time_stamp'] = time_stamp

    # task
    combine_cfg(cfg_base, cfg_task, 'task', task_name)
    # data
    combine_cfg(cfg_base, cfg_data, 'data', data_name)
    # model
    combine_cfg(cfg_base, cfg_model, 'model', model_name)

    # path_cfg, path_log
    cfg_base['path']['path_load'] = str(path_model)
    if suffix is None:
        cfg_base['path']['path_save'] += '%s-%s-%s' % (task_name, model_name, time_stamp)
    else:
        cfg_base['path']['path_save'] += '%s-%s-%s-%s' % (task_name, model_name, suffix, time_stamp)
    cfg_base['path']['path_cfg'] = cfg_base['path']['path_save'] + '/cfg.ini'
    cfg_base['path']['path_log'] = cfg_base['path']['path_save'] + '/log.log'
    cfg_base['path']['path_dataset'] = cfg_base['path']['path_dataset'] + data_name + '/'

    # activation, dimension三个部分都要变化，同时还需要指出opt
    activation_task = cfg_base['task']['activation']
    activation_model = read_l(cfg_base, 'model', 'activation')
    cfg_base.set('model', 'activation', str(activation_model + [activation_task]))

    # TODO: 这里在rdnet里面是不对的，需要考虑一下怎么处理，最后可能放到外面来或者重新写一个get_cfg
    n_labels_task = cfg_base['task'].getint('n_labels')
    n_labels_model = read_l(cfg_base, 'model', 'dimension')
    cfg_base.set('model', 'dimension', str(n_labels_model + [n_labels_task]))

    # 登录log
    logger(cfg_base['path']['path_log'])

    # create dir
    if not os.path.exists(cfg_base['path']['path_save']):
        os.mkdir(cfg_base['path']['path_save'])
        log_t('Create directory %s' % cfg_base['path']['path_save'])

    return cfg_base


def get_cfg_data(data_name, file_cfg_global='global.cfg'):
    """
    Create cfg object for data analysis rather than training
    Args:
        data_name: The name of dataset, used to load dataset

    Returns:
        Cfg object
    """
    cfg_base, cfg_data = load_cfg('../config/' + file_cfg_global), load_cfg('../config/data.cfg')

    combine_cfg(cfg_base, cfg_data, 'data', data_name)

    return cfg_base


def load_cfg(path_cfg):
    cfg = configparser.ConfigParser()
    cfg.read(path_cfg)
    return cfg


def combine_cfg(cfg_base, cfg, key, value):
    if value is not None:
        cfg_base.add_section(key)
        cfg_base.set(key, 'name', value)
        for option in cfg.options(value):
            cfg_base.set(key, option, cfg[value][option])
