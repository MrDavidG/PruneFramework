# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: mi_pruning
@time: 2019-05-09 10:58

Description. 
"""
from utils.mutual_information import kde_mi, bin_mi
from layers.conv_layer import ConvLayer

import tensorflow as tf
import numpy as np
import pickle
import os

def build_model(cluster_res_list, weight_a, weight_b):
    x = tf.placeholder()
    # Layer 1
    y = ConvLayer(x, )


def argmin_mi(layer_output, labelixs, method_mi, binsize, labelprobs=None):
    """
    Get the neuron in layer_output who has the minimal mi with label
    :param layer_output:
    :param labelixs:
    :param method_mi:
    :param binsize:
    :param labelprobs:
    :return: the index, and mi of the neuron with minimal MI
    """
    mi_min = 0
    min_index_neuron = -1
    for index_neuron in range(layer_output.shape[1]):
        if method_mi == 'bin':
            mi_neuron = bin_mi(layer_output[index_neuron], labelixs=labelixs, binsize=binsize)
        elif method_mi == 'kde':
            mi_neuron = kde_mi(layer_output[index_neuron], labelixs=labelixs, labelprobs=labelprobs)

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    return min_index_neuron, mi_min


def argmin_marginal_mi(layer_output, F, neuron_list_previous, labelixs, method_mi, binsize, labelprobs=None):
    """
    Get the neuron in F who has the minimal marginal MI w.r.t neuron_list_previous
    :param layer_output:
    :param F:
    :param neuron_list_previous:
    :param labelixs:
    :param method_mi:
    :param binsize:
    :param labelprobs:
    :return:
    """
    mi_min = 0
    min_index_neuron = -1
    for index_neuron in F:
        neuron_list = neuron_list_previous + [index_neuron]

        # Sort or not?
        if method_mi == 'bin':
            mi_neuron = bin_mi(layer_output[:, neuron_list], labelixs=labelixs, binsize=binsize)
        elif method_mi == 'kde':
            mi_neuron = kde_mi(layer_output[:, neuron_list], labelixs=labelixs, labelprobs=labelprobs)

        if mi_neuron < mi_min or min_index_neuron == -1:
            mi_min = mi_neuron
            min_index_neuron = index_neuron

    return min_index_neuron, mi_min


def combine_models(x, y_a, y_b, model_1, model_2, alpha_threshold, N_total, method_mi, binsize=0.5):
    num_layer = 15
    num_label_a = y_a.shape[1]
    num_label_b = y_b.shape[1]

    # 每得到一次新的y_b就需要重新算一下，需要看一下放在哪里
    # labelixs[index_label]: [batch_size], the sample that has label will be true
    labelixs_a = {}
    labelixs_b = {}
    for index_label in range(num_label_a):
        labelixs_a[index_label] = y_a[:, index_label] == 1
    for index_label in range(num_label_b):
        labelixs_b[index_label] = y_b[:, index_label] == 1

    if method_mi == 'kde':
        labelprobs_a = np.mean((y_a == 1).astype(np.float32), axis=0)
        labelprobs_b = np.mean((y_b == 1).astype(np.float32), axis=0)

    # Record list of clusters for all layers
    cluster_res_list = list()

    for layer_index in range(num_layer):
        shape_layer_model_1 = model_1.layers[layer_index].wegiht_tensors[0].shape
        shape_layer_model_2 = model_2.layers[layer_index].wegiht_tensors[0].shape

        # Total number of neurons
        if len(shape_layer_model_1.shape) == 4 and len(shape_layer_model_2.shape) == 4:
            num_neuron_total = shape_layer_model_1[3] + shape_layer_model_2[3]
        elif len(shape_layer_model_1.shape) == 2 and len(shape_layer_model_2.shape) == 2:
            num_neuron_total = shape_layer_model_1[1] + shape_layer_model_2[1]

        # Store clusters for each layer
        cluster_layer_dict = dict()
        cluster_layer_dict['A'] = list()
        cluster_layer_dict['B'] = list()
        cluster_layer_dict['AB'] = np.arange(num_neuron_total).tolist()
        cluster_res_list += [cluster_layer_dict]

    for k in range(N_total):
        # Obtain layer output
        layer_output_list_1 = list()
        layer_output_list_2 = list()

        for layer_index in range(num_layer):
            layer_output_1 = layer_output_list_1[layer_index]
            layer_output_2 = layer_output_list_2[layer_index]

            # [batch_size,h,w,channel_size] --> [batch_size,channel_size] for conv
            if layer_output_1.shape == 4 and layer_output_2.shape == 4:
                layer_output_1, layer_output_2 = list(
                    map(lambda x: np.mean(x, axis=(1, 2)), [layer_output_1, layer_output_2]))

            # Have the same number of neurons
            assert (layer_output_1.shape[1] == layer_output_2.shape[1])

            # All neurons
            F_A = cluster_res_list[layer_index]['AB']
            F_B = cluster_res_list[layer_index]['AB']
            alpha_sum = 0

            # Init with the neuron of A that has the  minimal MI with Y_B
            min_index_neuron, _ = argmin_mi(layer_output_1[:, cluster_res_list[layer_index]['A']], labelixs_b,
                                            method_mi, binsize, labelprobs_b)

            # Lines 11-13
            F_A.remove(min_index_neuron)
            cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
            cluster_res_list[layer_index]['A'].append(min_index_neuron)

            while alpha_sum <= alpha_threshold:
                # Traverse neurons in F_A and find the neuron with the minimal mi with Y_B
                min_index_neuron, mi_min = argmin_marginal_mi(layer_output_1, cluster_res_list[layer_index]['A'],
                                                              labelixs_b, method_mi, binsize, labelprobs=labelprobs_b)

                # Lines 17-19
                F_A = F_A.remove(min_index_neuron)
                cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
                cluster_res_list[layer_index]['A'].append(min_index_neuron)

                alpha_sum += mi_min

            alpha_sum = 0

            min_index_neuron, _ = argmin_mi(layer_output_1[:, cluster_res_list[layer_index]['AB']], labelixs_a,
                                            method_mi, binsize, labelprobs_a)
            # Line 24-26
            cluster_res_list[layer_index]['B'].remove(min_index_neuron)
            cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
            cluster_res_list[layer_index]['B'].append(min_index_neuron)

            while alpha_sum <= alpha_threshold:
                # Traverse neurons in F_B to find the neuron with the minimal mi with Y_A
                min_index_neuron, mi_min = argmin_marginal_mi(layer_output_2, cluster_res_list[layer_index]['B'],
                                                              labelixs_a, method_mi, binsize, labelprobs=labelprobs_a)

                # Lines 30-36
                cluster_res_list[layer_index]['AB'].remove(min_index_neuron)
                if min_index_neuron in cluster_res_list[layer_index]['A']:
                    cluster_res_list[layer_index]['A'].remove(min_index_neuron)
                else:
                    cluster_res_list[layer_index]['B'].append(min_index_neuron)
                    alpha_sum += mi_min

            # Reconnect neurons
            if layer_index >= 2:
                # Rebuild the model
                build_model(cluster_res_list, weight_a, weight_b)


if __name__ == '__main__':
    run()
