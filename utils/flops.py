# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: flops
@time: 2019-05-16 15:06

Description. 
"""

import numpy as np


def get_flops(sess, model):
    # Obtain all masks
    masks = list()
    layers_type = list()
    for layer in model.layers:
        if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib':
            layers_type += [layer.layer_type]
            masks += [layer.get_mask(threshold=model.prune_threshold)]

    masks = sess.run(masks)

    # how many channels/dims are prune in each layer
    prune_state = [np.sum(mask == 0) for mask in masks]

    # 原来的计算量
    total_flops, remain_flops, pruned_flops = 0, 0, 0

    for layer_index, num_mask in enumerate(prune_state):
        C_in = model.layers[layer_index].weight_tensors[0].shape.as_list()[-2]
        C_out = model.layers[layer_index].weight_tensors[0].shape.as_list()[-1]
        if layers_type[layer_index] == 'C_ib':
            # Feature map 大小
            M = model.layers[layer_index].layer_output.shape.as_list()[1]
            total_flops += 9 * C_in * M * M * C_out
            pruned_flops += 9 * C_in * M * M * num_mask
        elif layers_type[layer_index] == 'F_ib':
            total_flops += (2 * C_in - 1) * C_out
            pruned_flops += (2 * C_in - 1) * num_mask

    print('Total Flops: {}, Pruned Flops: {}, Remaining Flops:{}, Remain/Total Flops:{}, '
          'Each layer pruned: {}'.format(total_flops, pruned_flops, remain_flops,
                                         float(total_flops - pruned_flops) / total_flops, prune_state))
