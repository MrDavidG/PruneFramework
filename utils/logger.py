# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: logger.py
@time: 2019-03-27 10:59

Description here.
"""

from datetime import datetime


class logger:
    path_log = None
    record_log = True

    def __init__(self, path_log):
        logger.path_log = path_log

    @staticmethod
    def get_path_log():
        return logger.path_log


def print_t(str):
    print('[%s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str))


def print_l(str):
    print('{:-^60}'.format(str))


def log(str, need_print=True, end='\n'):
    if logger.record_log:
        with open(logger.path_log, 'a') as f:
            f.write(str + end)
    if need_print:
        print(str, end=end)


def log_t(str):
    if logger.record_log:
        with open(logger.path_log, 'a') as f:
            f.write('[%s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str))
    print_t(str)


def log_l(str):
    if logger.record_log:
        with open(logger.path_log, 'a') as f:
            f.write('{:-^60}\n'.format(str))
    print_l(str)
