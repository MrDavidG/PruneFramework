# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: label_mi
@time: 2019/12/18 10:13 上午

Calculate the relevance among different labels.
"""
import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from utils.mi_gpu import get_K_function
from utils.mi_gpu import kde_in_gpu

import tensorflow as tf
import pandas as pd
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

global entropy_func_upper


def get_label_info(y):
    labelixs_x, labelprobs_x = list(), list()
    for label_index in range(y.shape[1]):
        # 记录哪些位置是1,哪些位置是-1
        labelixs = {}
        labelixs[0] = y[:, label_index] == -1
        labelixs[1] = y[:, label_index] == 1
        labelixs_x.append(labelixs)

        # 各个维度为1的概率
        prob_label = np.mean((y[:, label_index] == 1).astype(np.float32), axis=0)
        labelprobs = np.array([1 - prob_label, prob_label])
        labelprobs_x.append(labelprobs)
    return labelixs_x, labelprobs_x


def cal_mi(lab_a, lab_b):
    """
    Calculate the mi between lab_a and lab_b
    Args:
        lab_a: [batch_size, dim_a]
        lab_b: [batch_size, dim_b]

    Returns:
        mi, a float-point number
    """
    global entropy_func_upper
    labelixs_a, labelprobs_a = get_label_info(lab_a)
    _, mi = kde_in_gpu(lab_b, labelixs_a, labelprobs_a, entropy_func_upper)
    return mi


def run(data_name, batch_size=128):
    # init
    global entropy_func_upper
    entropy_func_upper = get_K_function()
    # Obtain the labels
    # Here only consider the labels saved in csv format, like lfw and celebA
    data = pd.read_csv('/local/scratch/labels_deepfunneled.txt', delim_whitespace=True, header=None).values
    labels = np.array(data[:batch_size, 1:], dtype=np.float32)

    n_labels = labels.shape[-1]

    # Choose 10 basic labels
    matrix_mi = np.zeros(shape=[n_labels, n_labels])
    for i in range(n_labels):
        for j in range(i, n_labels):
            mi = cal_mi(labels[:, [i]], labels[:, [j]])
            matrix_mi[i][j] = mi
            matrix_mi[j][i] = mi
    base = np.argmax(np.sum(matrix_mi, axis=1))
    list_sort = np.argsort(matrix_mi[base]).tolist()
    if base in list_sort[-10:]:
        labels_basic = list_sort[-10:]
    else:
        labels_basic = list_sort[-9:] + [base]
    labels_basic = np.sort(labels_basic).tolist()
    print('labels_basic', labels_basic)

    # Calculate mi between
    # 直接获得一个不相关排序
    mi_list = list()
    for i in range(n_labels):
        mi_list.append(cal_mi(labels[:, labels_basic], labels[:, [i]]))
    res_sort = np.argsort(mi_list).tolist()

    # 开始替换
    labels_list = list()
    for i in range(1, 11):
        labels_list.append(np.sort(res_sort[:i] + labels_basic[i:]).tolist())

    # 计算一下10组mi
    print(-1, labels_basic, cal_mi(labels[:, labels_basic], labels[:, labels_basic]))
    for ind, inds in enumerate(labels_list):
        print(ind, inds, cal_mi(labels[:, labels_basic], labels[:, inds]))


if __name__ == '__main__':
    run('lfw')

# labels_basic [7, 16, 30, 34, 40, 45, 46, 48, 57, 64]

# -1 [7, 16, 30, 34, 40, 45, 46, 48, 57, 64] 6.108839079737663
# 0 [10, 16, 30, 34, 40, 45, 46, 48, 57, 64] 5.912561628967524
# 1 [4, 10, 30, 34, 40, 45, 46, 48, 57, 64] 5.800254814326763
# 2 [4, 10, 11, 34, 40, 45, 46, 48, 57, 64] 5.519734341651201
# 3 [4, 10, 11, 40, 45, 46, 48, 57, 62, 64] 5.2800874300301075
# 4 [4, 10, 11, 29, 45, 46, 48, 57, 62, 64] 5.026301991194487
# 5 [4, 9, 10, 11, 29, 46, 48, 57, 62, 64] 4.565149061381817
# 6 [4, 9, 10, 11, 29, 48, 49, 57, 62, 64] 3.9970237463712692
# 7 [4, 9, 10, 11, 29, 31, 49, 57, 62, 64] 3.3925466760993004
# 8 [4, 9, 10, 11, 29, 31, 49, 54, 62, 64] 3.012278448790312
# 9 [4, 9, 10, 11, 29, 31, 47, 49, 54, 62] 2.064763691276312
