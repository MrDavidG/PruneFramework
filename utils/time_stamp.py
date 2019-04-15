# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: time_stamp.py
@time: 2019-03-27 10:59

Description here.
"""

import datetime


def print_with_time_stamp(str):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time_now + ': ' + str + '...')


def print_with_time_stamp_line(str):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\r' + time_now + ': ' + str + '...', end=' ')
