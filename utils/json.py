# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: json
@time: 2019-11-21 14:38

Description. 
"""

import json


def read_f(cfg, sec, opt):
    return cfg[sec].getfloat(opt)


def read_i(cfg, sec, opt):
    return cfg[sec].getint(opt)


def read_l(cfg, sec, opt):
    return json.loads(cfg[sec][opt].replace('\'', '\"'))


def read_s(cfg, sec, opt):
    return cfg[sec][opt]
