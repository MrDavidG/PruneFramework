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
import configparser


def get_cfg(dataset_name, time_stamp, file_cfg_model=None, file_cfg_global='global.cfg', suffix=''):
    cfg_global = load_cfg('../config/' + file_cfg_global)

    # Get model cfg
    if file_cfg_model is not None:
        cfg_model = load_cfg('../config/' + file_cfg_model)

        for section in cfg_model.sections():
            for option in cfg_model.options(section):
                if section not in cfg_global.sections():
                    cfg_global.add_section(section)
                cfg_global.set(section, option, cfg_model[section][option])

    # Get data cfg
    cfg_data = load_cfg('../config/data.cfg')

    cfg_global.add_section('data')

    cfg_global.set('data', 'dataset_name', dataset_name)
    for option in cfg_data.options(dataset_name):
        cfg_global.set('data', option, cfg_data[dataset_name][option])

    cfg_global['basic']['task_name'] = dataset_name
    cfg_global['basic']['time_stamp'] = time_stamp
    if suffix is None:
        cfg_global['path']['path_save'] += '%s-%s' % (dataset_name, time_stamp)
    else:
        cfg_global['path']['path_save'] += '%s-%s-%s' % (dataset_name, suffix, time_stamp)
    cfg_global['path']['path_cfg'] = cfg_global['path']['path_save'] + '/cfg.ini'
    cfg_global['path']['path_log'] = cfg_global['path']['path_save'] + '/log.log'
    cfg_global['path']['path_dataset'] = cfg_global['path']['path_dataset'] + cfg_global['basic']['task_name'] + '/'

    logger(cfg_global['path']['path_log'])

    return cfg_global


def load_cfg(path_cfg):
    cfg = configparser.ConfigParser()
    cfg.read(path_cfg)
    return cfg
