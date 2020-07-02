# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: get_volume
@time: 2020/5/17 3:49 下午

Description. 
"""

import numpy as np
from copy import deepcopy


def t2c(task_index):
    if task_index == -1:
        return 'CEN'
    else:
        return chr(ord('A') + task_index)


def i2c(array):
    cs = list()
    for i in array:
        cs.append(chr(i + ord('a')))
    return cs


def get_volume(n_tasks, h_, w_, channel, kernel_size, net_origin, structure):
    # net_origin = dict()
    # for name in structure:
    #     if name.startswith('c') or name.startswith('f') and name != 'fla':
    #         layer = [len(cluster_res_dict[name][t2c(task_index)]) for task_index in range(n_tasks)]
    #         layer.append(len(cluster_res_dict[name][t2c(-1)]))
    #         net_origin[name] = layer

    def count(net_dict):
        # 读入图片的大小
        h, w = h_, w_
        params, flops = 0, 0
        prod_kernel = kernel_size * kernel_size

        for ind, name in enumerate(structure):
            if name == 'p':
                h, w = h // 2, w // 2
            elif name.startswith('c'):
                # 默认第一层为卷积层
                if ind == 0:
                    # 第一层，需要特殊处理
                    for task_index in range(-1, n_tasks):
                        params += prod_kernel * channel * net_dict[name][task_index]
                        flops += (2 * prod_kernel * channel - 1) * h * w * net_dict[name][task_index]
                else:
                    n_CEN_last = n_dict_last[-1]
                    for task_index in range(n_tasks):
                        n_in = n_CEN_last + n_dict_last[task_index]

                        params += prod_kernel * n_in * net_dict[name][task_index]
                        flops += (2 * prod_kernel * n_in - 1) * h * w * net_dict[name][task_index]

                    # CEN单独处理
                    params += prod_kernel * n_CEN_last * net_dict[name][-1]
                    flops += (2 * prod_kernel * n_CEN_last - 1) * h * w * net_dict[name][-1]

                n_dict_last = net_dict[name]
            elif name.startswith('f') and name != 'fla':
                n_CEN_last = n_dict_last[-1]
                for task_index in range(n_tasks):
                    n_in = n_CEN_last + n_dict_last[task_index]

                    params += n_in * net_dict[name][task_index]
                    flops += (2 * n_in - 1) * net_dict[name][task_index]

                params += n_CEN_last * net_dict[name][-1]
                flops += (2 * n_CEN_last - 1) * net_dict[name][-1]
                n_dict_last = net_dict[name]
            elif name == 'fla':
                n_dict_last = [_ * h * w for _ in n_dict_last]

        return params, flops

    total_params, total_flops = count(net_origin)

    # print('Parameters: %d,\tFLOPs: %d' % (total_params, total_flops))

    return total_params, total_flops


if __name__ == '__main__':
    task_lenet5 = {
        'n_tasks': 2,
        'h_': 124,
        'w_': 496,
        'channel': 1,
        'kernel_size': 5,
        'structure': ["c1", "p", "c2", "p", "fla", "f3", "f4", "f5"],

        'net_origin': {'c1': [1, 1, 2], 'c2': [2, 2, 3], 'f3': [6, 6, 10], 'f4': [4, 3, 9], 'f5': [5, 5, 0]}
    }

    task_celeba = {
        'n_tasks': 2,
        'h_': 72,
        'w_': 72,
        'channel': 3,
        'kernel_size': 3,
        'structure': ["c1_1", "c1_2", "p", "c2_1", "c2_2", "p", "c3_1", "c3_2", "c3_3", "p", "c4_1", "c4_2", "c4_3",
                      "p", "c5_1", "c5_2", "c5_3", "p", "fla", "f6", "f7", "f8"],

        'net_origin': {'c1_1': [0, 0, 26], 'c1_2': [0, 0, 26], 'c2_1': [0, 0, 27], 'c2_2': [0, 0, 38],
                       'c3_1': [0, 0, 40], 'c3_2': [0, 0, 44], 'c3_3': [0, 0, 43], 'c4_1': [0, 0, 60],
                       'c4_2': [2, 2, 52], 'c4_3': [5, 5, 59], 'c5_1': [9, 9, 40], 'c5_2': [4, 4, 40],
                       'c5_3': [7, 9, 48], 'f6': [27, 16, 58], 'f7': [7, 5, 82], 'f8': [20, 20, 0]}
    }

    task_lfw15 = {
        'n_tasks': 5,
        'h_': 72,
        'w_': 72,
        'channel': 3,
        'kernel_size': 3,
        'structure': ["c1_1", "c1_2", "p", "c2_1", "c2_2", "p", "c3_1", "c3_2", "c3_3", "p", "c4_1", "c4_2", "c4_3",
                      "p", "c5_1", "c5_2", "c5_3", "p", "fla", "f6", "f7", "f8"],
        'net_origin': {'c1_1': [0, 0, 0, 0, 0, 15], 'c1_2': [0, 0, 0, 0, 0, 10], 'c2_1': [0, 0, 0, 0, 0, 10],
                       'c2_2': [0, 0, 0, 0, 0, 8], 'c3_1': [0, 0, 0, 0, 0, 5], 'c3_2': [0, 0, 0, 0, 0, 5],
                       'c3_3': [0, 0, 0, 0, 0, 6], 'c4_1': [0, 0, 0, 0, 0, 7], 'c4_2': [0, 0, 0, 0, 0, 8],
                       'c4_3': [0, 0, 0, 0, 0, 12], 'c5_1': [1, 2, 2, 2, 1, 12], 'c5_2': [2, 1, 1, 1, 1, 16],
                       'c5_3': [3, 5, 3, 2, 2, 16], 'f6': [8, 2, 1, 4, 3, 39], 'f7': [9, 6, 2, 10, 7, 37],
                       'f8': [15, 15, 15, 15, 13, 0]}
    }

    for task in [task_lenet5]:
        if task['n_tasks'] == 2:
            per = [[0], [1], [0, 1]]
        else:
            per = [[0], [1], [2], [3], [4],
                   [0, 1], [0, 2], [0, 3], [0, 4],
                   [1, 2], [1, 3], [1, 4],
                   [2, 3], [2, 4],
                   [3, 4],
                   [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4],
                   [1, 2, 3], [1, 2, 4], [1, 3, 4],
                   [2, 3, 4],
                   [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4],
                   [0, 1, 2, 3, 4]]

        for net in per:
            net_copy = deepcopy(task['net_origin'])
            for i in range(task['n_tasks']):
                if i not in net:
                    for key in net_copy.keys():
                        net_copy[key][i] = 0

            a, b = get_volume(
                n_tasks=task['n_tasks'],
                h_=task['h_'],
                w_=task['w_'],
                channel=task['channel'],
                kernel_size=task['kernel_size'],
                net_origin=net_copy,
                structure=task['structure']
            )
            print('\t'.join([''.join(i2c(net)), str(a), str(b)]))

    # get_volume(
    #     n_tasks=2,
    #     h_=72,
    #     w_=72,
    #     channel=3,
    #     kernel_size=3,
    #     net_origin={'c1_1': [0, 0, 24], 'c1_2': [0, 0, 26], 'c2_1': [0, 0, 30], 'c2_2': [0, 0, 34], 'c3_1': [0, 0, 40],
    #                 'c3_2': [0, 0, 47], 'c3_3': [0, 0, 52], 'c4_1': [0, 0, 57], 'c4_2': [0, 0, 54], 'c4_3': [0, 0, 56],
    #                 'c5_1': [0, 0, 44], 'c5_2': [0, 0, 45], 'c5_3': [0, 0, 53], 'f6': [29, 0, 40], 'f7': [16, 0, 30],
    #                 'f8': [0, 0, 0]},
    #     structure=["c1_1", "c1_2", "p", "c2_1", "c2_2", "p", "c3_1", "c3_2", "c3_3", "p", "c4_1", "c4_2", "c4_3", "p",
    #                "c5_1", "c5_2", "c5_3", "p", "fla", "f6", "f7", "f8"]
    # )

    # lenet5
    # get_volume(
    #     n_tasks=2,
    #     h_=124,
    #     w_=496,
    #     channel=1,
    #     kernel_size=5,
    #     net_origin={'c1': [0, 0, 2], 'c2': [0, 2, 4], 'f3': [0, 6, 30], 'f4': [0, 3, 16], 'f5': [0, 5, 0]},
    #     structure=["c1", "p", "c2", "p", "fla", "f3", "f4", "f5"]
    # )

    # net_remain = {'c1_1': [0, 0, 0, 0, 0, 14], 'c1_2': [0, 0, 0, 0, 0, 10], 'c2_1': [0, 0, 0, 0, 0, 9],
    #               'c2_2': [0, 0, 0, 0, 0, 6], 'c3_1': [0, 0, 0, 0, 0, 11], 'c3_2': [0, 0, 0, 0, 0, 9],
    #               'c3_3': [0, 0, 0, 0, 0, 9], 'c4_1': [0, 0, 0, 0, 0, 9], 'c4_2': [0, 0, 0, 0, 0, 11],
    #               'c4_3': [0, 0, 0, 0, 0, 12], 'c5_1': [0, 0, 0, 0, 0, 14], 'c5_2': [2, 1, 1, 0, 1, 15],
    #               'c5_3': [3, 5, 3, 2, 0, 19], 'f6': [8, 2, 1, 4, 3, 29], 'f7': [9, 6, 2, 10, 7, 32],
    #               'f8': [15, 15, 15, 15, 13, 0]}
    # net_remain = {'c1_1': [0, 0, 0, 0, 0, 18], 'c1_2': [0, 0, 0, 0, 0, 15], 'c2_1': [0, 0, 0, 0, 0, 14],
    #               'c2_2': [0, 0, 0, 0, 0, 12],
    #               'c3_1': [0, 0, 0, 0, 0, 12], 'c3_2': [0, 0, 0, 0, 0, 14], 'c3_3': [0, 0, 0, 0, 0, 12],
    #               'c4_1': [0, 0, 0, 0, 0, 14],
    #               'c4_2': [0, 0, 0, 0, 0, 14], 'c4_3': [0, 0, 0, 0, 0, 16], 'c5_1': [0, 0, 0, 0, 0, 21],
    #               'c5_2': [0, 0, 0, 0, 0, 23],
    #               'c5_3': [0, 0, 0, 0, 0, 28], 'f6': [0, 0, 0, 0, 0, 54], 'f7': [0, 0, 0, 0, 0, 59],
    #               'f8': [15, 15, 15, 15, 13, 0]}
    #
    # for net in [[0], [1], [2], [3], [4],
    #             [0, 1], [0, 2], [0, 3], [0, 4],
    #             [1, 2], [1, 3], [1, 4],
    #             [2, 3], [2, 4],
    #             [3, 4],
    #             [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4],
    #             [1, 2, 3], [1, 2, 4], [1, 3, 4],
    #             [2, 3, 4],
    #             [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4],
    #             [0, 1, 2, 3, 4]]:
    #     net_copy = deepcopy(net_remain)
    #     for i in [0, 1, 2, 3, 4]:
    #         if i not in net:
    #             net_copy['c5_2'][i] = 0
    #             net_copy['c5_3'][i] = 0
    #             net_copy['f6'][i] = 0
    #             net_copy['f7'][i] = 0
    #             net_copy['f8'][i] = 0
    #
    #     a, b = get_volume(
    #         n_tasks=5,
    #         h_=72,
    #         w_=72,
    #         channel=3,
    #         kernel_size=3,
    #         net_origin=net_copy,
    #         structure=["c1_1", "c1_2", "p", "c2_1", "c2_2", "p", "c3_1", "c3_2", "c3_3", "p", "c4_1", "c4_2", "c4_3",
    #                    "p",
    #                    "c5_1", "c5_2", "c5_3", "p", "fla", "f6", "f7", "f8"]
    #     )
    #     print('\t'.join([''.join(i2c(net)), str(a), str(b)]))
