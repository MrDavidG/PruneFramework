# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: measure_relevancy
@time: 2019-07-26 14:35

Description. 
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

def get_mi(x, y):
    n_x = np.shape(x)[0]
    n_y = np.shape(y)[0]

    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)

    mi_xy = 0

    for ind_x, val_x in enumerate(unique_x):
        for ind_y, val_y in enumerate(unique_y):
            p_x = counts_x[ind_x] / n_x
            p_y = counts_y[ind_y] / n_y
            p_xy = np.sum(((x == val_x).astype(np.float32) + (y == val_y).astype(np.float32)) == 2) / n_x
            if p_x == 0 or p_y == 0 or p_xy == 0:
                mi_xy += 0
            else:
                mi_xy += p_xy * np.log(p_xy / (p_x * p_y))
            # print('p_x: %f, p_y: %f, p_xy: %f' % (p_x, p_y, p_xy))

    # print('\n\n')
    return mi_xy


def measure_relevancy(path_data, ):
    # 获取所有的label(lfw: 73个)
    labels = pd.read_csv(path_data, delim_whitespace=True, header=None).values[:, 1:]

    n_classes = np.shape(labels)[1]
    # 两两一组衡量相关性

    res_rele = list()

    index = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # rele = get_mi(labels[:, i], labels[:, j])
            rele = mutual_info_score(labels[:,i], labels[:, j])
            # if index in [1597, 741, 1275, 1166, 1112, 597, 2610, 65, 1192]:
            #     print('%d: %d-%d' % (index, i, j))
            res_rele.append(rele)
            index += 1

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(res_rele)).tolist(), np.sort(res_rele))
    plt.savefig('./relevance.png')
    plt.title('MI distribution of paired labels')
    plt.ylabel('MI')
    plt.xlabel('Index of pairs')
    print('debug')


if __name__ == '__main__':
    measure_relevancy(
        path_data='/local/scratch/labels_deepfunneled.txt'
    )
