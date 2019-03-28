# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: config.py.py
@time: 2019-03-27 10:58

Description here.
"""

import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    # create a path for summary of the experiment
    config.summary_dir = os.path.join('.../experiments', config.exp_name, 'summary/')
    # create a path for checkpoint of the experiment
    config.checkpoint_dir = os.path.join('../experiments', config.exp_name, 'checkpoint/')
    return config
