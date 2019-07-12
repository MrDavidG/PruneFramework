# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: test
@time: 2019-04-16 10:22

Description.
"""
import pickle

if __name__ == '__main__':
    w = pickle.load(open('datasets_mean_std.pickle', 'rb'))

    print(w)
