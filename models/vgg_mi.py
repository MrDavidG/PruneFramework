# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: vgg_model
@time: 2019-03-27 15:31

Description.
"""
import sys

sys.path.append(r"/local/home/david/Remote/")
from models.base_model import BaseModel
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
from layers.ib_layer import InformationBottleneckLayer
from utils.config import process_config
from datetime import datetime

import numpy as np
import pickle
import time

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VGG_Combined(BaseModel):
    def __init__(self, config, task_name, weight_a, weight_b, cluster_res_list, signal_list, musk=False, gamma=1.,
                 ib_threshold=None):
        super(VGG_Combined, self).__init__(config)

        self.imgs_path = self.config.dataset_path + 'celeba/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name
        self.is_musked = musk
        self.gamma = gamma

        if self.prune_method == 'info_bottle' and ib_threshold is not None:
            self.prune_threshold = ib_threshold

        self.op_logits_a = None
        self.op_logits_b = None

        self.op_opt_a = None
        self.op_opt_b = None

        self.op_accuracy_a = None
        self.op_accuracy_b = None

        self.op_loss_a = None
        self.op_loss_b = None

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        self.regularizer_decay = None

        self.signal_list = signal_list

        self.weight_dict = self.construct_initial_weights(weight_a, weight_b, cluster_res_list)

        if self.prune_method == 'info_bottle' and 'conv1_1/info_bottle/mu' not in self.weight_dict.keys():
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib(cluster_res_list))

    def construct_initial_weights(self, weight_dict_a, weight_dict_b, cluster_res_list):
        dim_list = [64, 64,
                    128, 128,
                    256, 256, 256,
                    512, 512, 512,
                    512, 512, 512,
                    4096, 4096, self.n_classes]

        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        def weight_variable(shape, local=0, scale=1e-2):
            # return np.random.normal(loc=local, scale=scale, size=shape).astype(dtype=np.float32)
            return np.zeros(shape=shape).astype(dtype=np.float32)

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

        def get_expand(array, step=4):
            res_list = list()
            for value in array:
                res_list += [value * step + x for x in range(step)]
            return res_list

        def del_offset(list_, offset):
            return list(np.array(list_) - offset)

        weight_dict = dict()

        for layer_index, layer_name in enumerate(['conv1_1', 'conv1_2',
                                                  'conv2_1', 'conv2_2',
                                                  'conv3_1', 'conv3_2', 'conv3_3',
                                                  'conv4_1', 'conv4_2', 'conv4_3',
                                                  'conv5_1', 'conv5_2', 'conv5_3',
                                                  'fc6', 'fc7', 'fc8']):
            # All weights and biases
            weight = np.concatenate((weight_dict_a[layer_name + '/weights'], weight_dict_b[layer_name + '/weights']),
                                    axis=-1)
            bias = np.concatenate((weight_dict_a[layer_name + '/biases'], weight_dict_b[layer_name + '/biases']))

            # Obtain neuron list
            A = cluster_res_list[layer_index]['A']
            AB = cluster_res_list[layer_index]['AB']
            B = cluster_res_list[layer_index]['B']

            if layer_index == 0:
                # The first layer
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/weights'] = weight[:, :, :, A]
                    weight_dict[layer_name + '/A/biases'] = bias[A]
                if get_signal(layer_index, 'AB'):
                    weight_dict[layer_name + '/AB/weights'] = weight[:, :, :, AB]
                    weight_dict[layer_name + '/AB/biases'] = bias[AB]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/weights'] = weight[:, :, :, B]
                    weight_dict[layer_name + '/B/biases'] = bias[B]
            else:
                # Obtain neuron list of last layer
                A_last = cluster_res_list[layer_index - 1]['A']
                AB_last = cluster_res_list[layer_index - 1]['AB']
                B_last = cluster_res_list[layer_index - 1]['B']

                # Number of neurons in last layer in combined model
                num_A_last_layer = len(A_last)
                num_B_last_layer = len(B_last)
                num_AB_from_a_last_layer = (np.array(AB_last) < dim_list[layer_index - 1]).sum()
                num_AB_from_b_last_layer = len(AB_last) - num_AB_from_a_last_layer

                # Number of neurons in this layer
                num_A = len(A)
                num_B = len(B)
                num_AB_from_a = (np.array(AB) < dim_list[layer_index]).sum()
                num_AB_from_b = len(AB) - num_AB_from_a

                # Init weights
                if layer_name.startswith('conv'):
                    # A
                    if get_signal(layer_index, 'A'):
                        in_list = A_last + AB_last[:num_AB_from_a_last_layer]
                        weight_dict[layer_name + '/A/weights'] = np.concatenate((weight[:, :, in_list, :][:, :, :, A],
                                                                                 weight_variable((3, 3,
                                                                                                  num_AB_from_b_last_layer,
                                                                                                  num_A))), axis=2)

                    # AB
                    if get_signal(layer_index, 'AB'):
                        # From A to AB
                        if get_signal(layer_index, 'fromA'):
                            weight_dict[layer_name + '/AB/A/weights'] = np.concatenate(
                                (weight[:, :, A_last, :][:, :, :,
                                 AB[:num_AB_from_a]],
                                 weight_variable((3, 3,
                                                  num_A_last_layer,
                                                  num_AB_from_b))),
                                axis=-1)

                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_AB_part_a = np.concatenate(
                                (weight[:, :, AB_last[:num_AB_from_a_last_layer], :][:, :, :,
                                 AB[:num_AB_from_a]],
                                 weight_variable(
                                     (3, 3, num_AB_from_b_last_layer, num_AB_from_a))),
                                axis=2)

                            # 必须是在第3个维度，并且是b model的部分的时候需要del_offset
                            weight_AB_part_b = np.concatenate((weight_variable(
                                (3, 3, num_AB_from_a_last_layer, num_AB_from_b)), weight[:, :, del_offset(
                                AB_last[num_AB_from_a_last_layer:], dim_list[layer_index - 1]), :][:, :, :,
                                                                                  AB[num_AB_from_a:]]), axis=2)
                            weight_dict[layer_name + '/AB/AB/weights'] = np.concatenate(
                                (weight_AB_part_a, weight_AB_part_b), axis=-1)

                        # From B to AB
                        if get_signal(layer_index, 'fromB'):
                            weight_dict[layer_name + '/AB/B/weights'] = np.concatenate((weight_variable(
                                (3, 3, num_B_last_layer, num_AB_from_a)), weight[:, :,
                                                                          del_offset(B_last, dim_list[layer_index - 1]),
                                                                          :][:, :, :, AB[num_AB_from_a:]]), axis=-1)

                    # B
                    if get_signal(layer_index, 'B'):
                        in_list = del_offset(AB_last[num_AB_from_a_last_layer:] + B_last, dim_list[layer_index - 1])
                        weight_dict[layer_name + '/B/weights'] = np.concatenate((weight_variable(
                            (3, 3, num_AB_from_a_last_layer, num_B)), weight[:, :, in_list, :][:, :, :, B]), axis=2)

                elif layer_name.startswith('fc'):
                    # Fc layer

                    if layer_name == 'fc6':
                        # From conv to fc, times h*w
                        # !!!!!从conv层到全连接层的时候，原来的512->2048,而且其中的index=i, 投影变成了[4*i, 4i+4]...整体是变成了4倍
                        # 最后一层产生的feature map边长的平方
                        length = 4
                        num_A_last_layer *= length
                        num_B_last_layer *= length
                        num_AB_from_a_last_layer *= length
                        num_AB_from_b_last_layer *= length

                        A_last = get_expand(A_last)
                        AB_last = get_expand(AB_last)
                        B_last = get_expand(B_last)

                    # New weights for neurons from A and AB to A
                    if get_signal(layer_index, 'A'):
                        in_list = A_last + AB_last[:num_AB_from_a_last_layer]
                        weight_dict[layer_name + '/A/weights'] = np.concatenate(
                            (weight[in_list, :][:, A], weight_variable((num_AB_from_b_last_layer, num_A))), axis=0)

                    # The output layer does not have AB
                    if get_signal(layer_index, 'AB'):
                        # New weights for neurons from last layer to AB
                        # From A to AB
                        if get_signal(layer_index, 'fromA'):
                            weight_dict[layer_name + '/AB/A/weights'] = np.concatenate((weight[A_last, :][:,
                                                                                        AB[:num_AB_from_a]],
                                                                                        weight_variable(
                                                                                            (num_A_last_layer,
                                                                                             num_AB_from_b))),
                                                                                       axis=-1)

                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_AB_part_a = np.concatenate((weight[AB_last[:num_AB_from_a_last_layer], :][:,
                                                               AB[:num_AB_from_a]], weight_variable(
                                (num_AB_from_b_last_layer, num_AB_from_a))), axis=0)

                            offset = dim_list[layer_index - 1]
                            if layer_name == 'fc6':
                                # 如果是fc6,conv转化fc的话offset也得变
                                offset *= 4
                            weight_AB_part_b = np.concatenate((weight_variable(
                                (num_AB_from_a_last_layer, num_AB_from_b)),
                                                               weight[
                                                               del_offset(AB_last[num_AB_from_a_last_layer:], offset),
                                                               :][:,
                                                               AB[num_AB_from_a:]]), axis=0)

                            weight_dict[layer_name + '/AB/AB/weights'] = np.concatenate(
                                (weight_AB_part_a, weight_AB_part_b), axis=-1)

                        # From B to AB
                        if get_signal(layer_index, 'fromB'):
                            weight_dict[layer_name + '/AB/B/weights'] = np.concatenate((weight_variable(
                                (num_B_last_layer, num_AB_from_a)), weight[
                                                                    del_offset(B_last, dim_list[layer_index - 1]), :][:,
                                                                    AB[num_AB_from_a:]]), axis=-1)

                    # New weights for neurons from AB and B to B
                    if get_signal(layer_index, 'B'):
                        in_list = del_offset(AB_last[num_AB_from_a_last_layer:] + B_last, dim_list[layer_index - 1])
                        weight_dict[layer_name + '/B/weights'] = np.concatenate(
                            (weight_variable((num_AB_from_a_last_layer, num_B)), weight[in_list, :][:, B]), axis=0)

                # Biases
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/biases'] = bias[cluster_res_list[layer_index]['A']]
                if get_signal(layer_index, 'AB'):
                    if get_signal(layer_index, 'AB'):
                        weight_dict[layer_name + '/AB/AB/biases'] = bias[cluster_res_list[layer_index]['AB']]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/biases'] = bias[cluster_res_list[layer_index]['B']]

        return weight_dict

    def construct_initial_weights_ib(self, cluster_res_list):
        weight_dict = dict()
        # parameters of the information bottleneck
        for layer_index, name_layer in enumerate(['conv1_1', 'conv1_2',
                                                  'conv2_1', 'conv2_2',
                                                  'conv3_1', 'conv3_2', 'conv3_3',
                                                  'conv4_1', 'conv4_2', 'conv4_3',
                                                  'conv5_1', 'conv5_2', 'conv5_3',
                                                  'fc6', 'fc7']):
            # Number of neurons in this layer
            num_A = len(cluster_res_list[layer_index]['A'])
            num_B = len(cluster_res_list[layer_index]['B'])
            num_AB = len(cluster_res_list[layer_index]['AB'])

            # A
            weight_dict[name_layer + '/A/info_bottle/mu'] = np.random.normal(loc=1, scale=0.01,
                                                                             size=[num_A]).astype(np.float32)

            weight_dict[name_layer + '/A/info_bottle/logD'] = np.random.normal(loc=-9, scale=0.01,
                                                                               size=[num_A]).astype(np.float32)

            # AB
            weight_dict[name_layer + '/AB/info_bottle/mu'] = np.random.normal(loc=1, scale=0.01,
                                                                              size=[num_AB]).astype(np.float32)

            weight_dict[name_layer + '/AB/info_bottle/logD'] = np.random.normal(loc=-9, scale=0.01,
                                                                                size=[num_AB]).astype(np.float32)

            # B
            weight_dict[name_layer + '/B/info_bottle/mu'] = np.random.normal(loc=1, scale=0.01,
                                                                             size=[num_B]).astype(np.float32)

            weight_dict[name_layer + '/B/info_bottle/logD'] = np.random.normal(loc=-9, scale=0.01,
                                                                               size=[num_B]).astype(np.float32)
        return weight_dict

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

        def get_conv(x, regu_conv, has_bias=True):
            return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
                             is_musked=self.is_musked, regularizer_conv=regu_conv,
                             is_merge_bn=self.meta_val('is_merge_bn'), has_bias=has_bias)

        def get_ib(y, layer_type, kl_mult):
            if self.prune_method == 'info_bottle':
                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult,
                                                      mask_threshold=self.prune_threshold)
                self.layers.append(ib_layer)
                y, ib_kld = ib_layer.layer_output
                self.kl_total += ib_kld
            return y

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total = 0.

            layer_index = 0
            # the name of the layer and the coefficient of the kl divergence
            for layer_set in [('conv1_1', 1.0 / 32), ('conv1_2', 1.0 / 32), 'pooling',
                              ('conv2_1', 1.0 / 16), ('conv2_2', 1.0 / 16), 'pooling',
                              ('conv3_1', 1.0 / 8), ('conv3_2', 1.0 / 8), ('conv3_3', 1.0 / 8), 'pooling',
                              ('conv4_1', 1.0 / 4), ('conv4_2', 1.0 / 4), ('conv4_3', 1.0 / 4), 'pooling',
                              ('conv5_1', 1.0 / 2), ('conv5_2', 1.0 / 2), ('conv5_3', 1.0 / 2), 'pooling']:
                if layer_index == 0:
                    conv_name, kl_mult = layer_set

                    # 如果有A部分的话才进行连接
                    if get_signal(layer_index, 'A'):
                        with tf.variable_scope(conv_name + '/A'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_A = tf.nn.relu(conv.layer_output)

                            y_A = get_ib(y_A, 'C_ib', kl_mult)

                    if get_signal(layer_index, 'AB'):
                        with tf.variable_scope(conv_name + '/AB'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_AB = tf.nn.relu(conv.layer_output)

                            y_AB = get_ib(y_AB, 'C_ib', kl_mult)

                    if get_signal(layer_index, 'B'):
                        with tf.variable_scope(conv_name + '/B'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_B = tf.nn.relu(conv.layer_output)

                            y_B = get_ib(y_B, 'C_ib', kl_mult)

                elif layer_set != 'pooling':
                    conv_name, kl_mult = layer_set

                    # A
                    if get_signal(layer_index, 'A'):
                        with tf.variable_scope(conv_name + '/A'):

                            if get_signal(layer_index, 'fromA') and get_signal(layer_index, 'fromAB'):
                                # From A and AB to A
                                conv = get_conv(tf.concat((y_A_last, y_AB_last), axis=-1), self.regularizer_conv)
                            elif get_signal(layer_index, 'fromAB'):
                                # Only AB to A
                                conv = get_conv(y_AB_last, self.regularizer_conv)
                            elif get_signal(layer_index, 'fromA'):
                                # Only A to A
                                conv = get_conv(y_A_last, self.regularizer_conv)
                            self.layers.append(conv)
                            y_A = tf.nn.relu(conv.layer_output)

                            y_A = get_ib(y_A, 'C_ib', kl_mult)

                    # AB
                    if get_signal(layer_index, 'AB'):
                        with tf.variable_scope(conv_name + '/AB'):
                            # y_from_A, y_from_AB, y_from_B = 0, 0, 0
                            # if get_signal(layer_index, 'fromA'):
                            #     with tf.variable_scope('A'):
                            #         # From A to AB
                            #         conv_A = get_conv(y_A_last, self.regularizer_decay, has_bias=False)
                            #         self.layers.append(conv_A)
                            #         y_from_A = conv_A.layer_output

                            if get_signal(layer_index, 'fromAB'):
                                with tf.variable_scope('AB'):
                                    # From AB to AB
                                    conv_AB = get_conv(y_AB_last, self.regularizer_conv)
                                    self.layers.append(conv_AB)
                                    y_from_AB = conv_AB.layer_output

                            # if get_signal(layer_index, 'fromB'):
                            #     with tf.variable_scope('B'):
                            #         # From B to AB
                            #         conv_B = get_conv(y_B_last, self.regularizer_decay, has_bias=False)
                            #         self.layers.append(conv_B)
                            #         y_from_B = conv_B.layer_output

                            # y_AB = tf.nn.relu(y_from_A + y_from_AB + y_from_B)
                            y_AB = tf.nn.relu(y_from_AB)

                            y_AB = get_ib(y_AB, 'C_ib', kl_mult)

                    # B
                    if get_signal(layer_index, 'B'):
                        with tf.variable_scope(conv_name + '/B'):
                            if get_signal(layer_index, 'fromAB') and get_signal(layer_index, 'fromB'):
                                # From AB and B to B
                                conv = get_conv(tf.concat((y_AB_last, y_B_last), axis=-1), self.regularizer_conv)
                            elif get_signal(layer_index, 'fromAB'):
                                # Only AB to B
                                conv = get_conv(y_AB_last, self.regularizer_conv)
                            elif get_signal(layer_index, 'fromB'):
                                # Only B to B
                                conv = get_conv(y_A_last, self.regularizer_conv)
                            self.layers.append(conv)
                            y_B = tf.nn.relu(conv.layer_output)

                            y_B = get_ib(y_B, 'C_ib', kl_mult)

                elif layer_set == 'pooling':
                    if get_signal(layer_index, 'A'):
                        y_A = tf.nn.max_pool(y_A_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index, 'AB'):
                        y_AB = tf.nn.max_pool(y_AB_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index, 'A'):
                        y_B = tf.nn.max_pool(y_B_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # Record the output of this layer
                if get_signal(layer_index, 'A'):
                    y_A_last = y_A
                if get_signal(layer_index, 'AB'):
                    y_AB_last = y_AB
                if get_signal(layer_index, 'B'):
                    y_B_last = y_B

                if layer_set != 'pooling':
                    layer_index += 1

            # From conv to fc layer
            if get_signal(layer_index, 'A'):
                y_A_last = tf.contrib.layers.flatten(y_A_last)
            if get_signal(layer_index, 'AB'):
                y_AB_last = tf.contrib.layers.flatten(y_AB_last)
            if get_signal(layer_index, 'B'):
                y_B_last = tf.contrib.layers.flatten(y_B_last)

            for fc_name in ['fc6', 'fc7']:
                # A
                if get_signal(layer_index, 'A'):
                    with tf.variable_scope(fc_name + '/A'):
                        if get_signal(layer_index, 'fromA') and get_signal(layer_index, 'fromAB'):
                            fc_layer = FullConnectedLayer(tf.concat((y_A_last, y_AB_last), axis=-1), self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)
                        elif get_signal(layer_index, 'fromAB'):
                            fc_layer = FullConnectedLayer(y_AB_last, self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)
                        elif get_signal(layer_index, 'fromA'):
                            fc_layer = FullConnectedLayer(y_A_last, self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)

                        self.layers.append(fc_layer)
                        y_A = tf.nn.relu(fc_layer.layer_output)

                        y_A = get_ib(y_A, 'F_ib', self.gamma)

                # AB
                if get_signal(layer_index, 'AB'):
                    with tf.variable_scope(fc_name + '/AB'):
                        y_from_A, y_from_AB, y_from_B = 0, 0, 0
                        # if get_signal(layer_index, 'fromA'):
                        #     with tf.variable_scope('A'):
                        #         # From A to AB
                        #         fc_layer_A = FullConnectedLayer(y_A_last, self.weight_dict,
                        #                                         regularizer_fc=self.regularizer_decay, has_bias=False)
                        #         self.layers.append(fc_layer_A)
                        #         y_from_A = fc_layer_A.layer_output

                        if get_signal(layer_index, 'fromAB'):
                            with tf.variable_scope('AB'):
                                # From AB to AB
                                fc_layer_AB = FullConnectedLayer(y_AB_last, self.weight_dict,
                                                                 regularizer_fc=self.regularizer_fc)
                                self.layers.append(fc_layer_AB)
                                y_from_AB = fc_layer_AB.layer_output

                        # if get_signal(layer_index, 'fromB'):
                        #     with tf.variable_scope('B'):
                        #         # From B to AB
                        #         fc_layer_B = FullConnectedLayer(y_B_last, self.weight_dict,
                        #                                         regularizer_fc=self.regularizer_decay, has_bias=False)
                        #         self.layers.append(fc_layer_B)
                        #         y_from_B = fc_layer_B.layer_output

                        y_AB = tf.nn.relu(y_from_A + y_from_AB + y_from_B)

                        y_AB = get_ib(y_AB, 'F_ib', self.gamma)

                # B
                if get_signal(layer_index, 'B'):
                    with tf.variable_scope(fc_name + '/B'):
                        if get_signal(layer_index, 'fromAB') and get_signal(layer_index, 'fromB'):
                            fc_layer = FullConnectedLayer(tf.concat((y_AB_last, y_B_last), axis=-1), self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)
                        elif get_signal(layer_index, 'fromAB'):
                            fc_layer = FullConnectedLayer(y_AB_last, self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)
                        elif get_signal(layer_index, 'fromB'):
                            fc_layer = FullConnectedLayer(y_B_last, self.weight_dict,
                                                          regularizer_fc=self.regularizer_fc)
                        y_B = tf.nn.relu(fc_layer.layer_output)

                        y_B = get_ib(y_B, 'F_ib', self.gamma)

                # Record the output of last layer
                if get_signal(layer_index, 'A'):
                    y_A_last = y_A
                if get_signal(layer_index, 'AB'):
                    y_AB_last = y_AB
                if get_signal(layer_index, 'B'):
                    y_B_last = y_B

                layer_index += 1

            with tf.variable_scope('fc8'):
                # A
                with tf.variable_scope('A'):
                    if get_signal(layer_index, 'fromA') and get_signal(layer_index, 'fromAB'):
                        fc_layer = FullConnectedLayer(tf.concat((y_A_last, y_AB_last), axis=-1), self.weight_dict,
                                                      regularizer_fc=self.regularizer_fc)
                    elif get_signal(layer_index, 'fromA'):
                        fc_layer = FullConnectedLayer(y_A_last, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    elif get_signal(layer_index, 'fromAB'):
                        fc_layer = FullConnectedLayer(y_AB_last, self.weight_dict, regularizer_fc=self.regularizer_fc)

                    self.layers.append(fc_layer)
                    y_A = fc_layer.layer_output

                # B
                with tf.variable_scope('B'):
                    if get_signal(layer_index, 'fromAB') and get_signal(layer_index, 'fromB'):
                        fc_layer = FullConnectedLayer(tf.concat((y_AB_last, y_B_last), axis=-1), self.weight_dict,
                                                      regularizer_fc=self.regularizer_fc)
                    elif get_signal(layer_index, 'fromAB'):
                        fc_layer = FullConnectedLayer(y_AB_last, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    elif get_signal(layer_index, 'fromB'):
                        fc_layer = FullConnectedLayer(y_B_last, self.weight_dict, regularizer_fc=self.regularizer_fc)

                    self.layers.append(fc_layer)
                    y_B = fc_layer.layer_output

                self.op_logits = tf.nn.tanh(tf.concat((y_A, y_B), axis=1))
                self.op_logits_a = tf.nn.tanh(y_A)
                self.op_logits_b = tf.nn.tanh(y_B)

    def loss(self):
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / 2, tf.int32)
        mae_loss = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
        mae_loss_a = tf.losses.mean_squared_error(labels=self.Y[:, :dim_label_single], predictions=self.op_logits_a)
        mae_loss_b = tf.losses.mean_squared_error(labels=self.Y[:, dim_label_single:], predictions=self.op_logits_b)

        l2_loss = tf.losses.get_regularization_loss()

        # for the pruning method
        if self.prune_method == 'info_bottle':
            self.op_loss = mae_loss + l2_loss + self.kl_factor * self.kl_total
            self.op_loss_a = mae_loss_a + l2_loss + self.kl_factor * self.kl_total
            self.op_loss_b = mae_loss_b + l2_loss + self.kl_factor * self.kl_total
        else:
            self.op_loss = mae_loss + l2_loss
            self.op_loss_a = mae_loss_a + l2_loss
            self.op_loss_b = mae_loss_b + l2_loss

    def get_params(self):
        param_AB_list = list()
        param_A_list = list()
        param_B_list = list()
        layer_name_list = ['conv1_1', 'conv1_2',
                           'conv2_1', 'conv2_2',
                           'conv3_1', 'conv3_2', 'conv3_3',
                           'conv4_1', 'conv4_2', 'conv4_3',
                           'conv5_1', 'conv5_2', 'conv5_3',
                           'fc6', 'fc7', 'fc8']
        for layer in self.layers:
            if layer.layer_name in [name + '/A' for name in layer_name_list]:
                param_A_list += layer.weight_tensors
            elif layer.layer_name in [name + '/B' for name in layer_name_list]:
                param_B_list += layer.weight_tensors
            elif layer.layer_name.endswith('/AB') or layer.layer_name.endswith('/AB/A') or layer.layer_name.endswith(
                    '/AB/AB') or layer.layer_name.endswith('/AB/B'):
                param_AB_list += layer.weight_tensors

        return param_A_list, param_AB_list, param_B_list

    def optimize(self):
        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # Create a optimizer
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=0.9,
                                                      use_nesterov=True)

                self.op_opt = self.opt.minimize(self.op_loss)

                # grads_and_vars is a list of tuples
                params_A, params_AB, params_B = self.get_params()

                grads_vars_a = self.opt.compute_gradients(self.op_loss, params_A + params_AB)
                grads_vars_b = self.opt.compute_gradients(self.op_loss, params_AB + params_B)

                # Ask the optimizer to apply the capped gradients
                self.op_opt_a = self.opt.apply_gradients(grads_vars_a)
                self.op_opt_b = self.opt.apply_gradients(grads_vars_b)

    def evaluate(self):
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / 2, tf.int32)
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            correct_preds_a = tf.equal(self.Y[:, :dim_label_single], tf.sign(self.op_logits_a))
            correct_preds_b = tf.equal(self.Y[:, dim_label_single:], tf.sign(self.op_logits_b))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
                                                                                           tf.float32)
            self.op_accuracy_a = tf.reduce_sum(tf.cast(correct_preds_a, tf.float32)) / tf.cast(tf.shape(self.Y)[1] / 2,
                                                                                               tf.float32)
            self.op_accuracy_b = tf.reduce_sum(tf.cast(correct_preds_b, tf.float32)) / tf.cast(tf.shape(self.Y)[1] / 2,
                                                                                               tf.float32)

    def build(self, weight_dict=None, share_scope=None, is_merge_bn=False):
        # dont need to merge bn
        if is_merge_bn and not self.meta_val('is_merge_bn'):
            self.merge_batch_norm_to_conv()
            self.set_meta_val('is_merge_bn', is_merge_bn)

        if weight_dict:
            self.weight_dict = weight_dict
            self.share_scope = share_scope
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()

    def train_one_epoch(self, sess, init, epoch, step, task_name):
        sess.run(init)
        total_loss = 0
        total_kl = 0
        total_correct_preds = 0
        total_correct_preds_a = 0
        total_correct_preds_b = 0
        n_batches = 0

        if task_name == 'A':
            op_opt = self.op_opt_a
        elif task_name == 'B':
            op_opt = self.op_opt_b
        else:
            op_opt = self.op_opt

        time_last = time.time()

        try:
            while True:
                if self.prune_method == 'info_bottle':
                    _, loss, accu_batch, accu_batch_a, accu_batch_b, kl = sess.run(
                        [op_opt, self.op_loss, self.op_accuracy, self.op_accuracy_a, self.op_accuracy_b,
                         self.kl_total],
                        feed_dict={self.is_training: True})
                    total_kl += kl
                else:
                    _, loss, accu_batch, accu_batch_a, accu_batch_b = sess.run(
                        [self.op_opt, self.op_loss, self.op_accuracy, self.op_accuracy_a, self.op_accuracy_b],
                        feed_dict={self.is_training: True})
                step += 1
                total_loss += loss
                total_correct_preds += accu_batch
                total_correct_preds_a += accu_batch_a
                total_correct_preds_b += accu_batch_b
                n_batches += 1

                if n_batches % 5 == 0:
                    print(
                        '\repoch={:d}, batch={:d}/{:d}, curr_loss={:f}, train_acc={:%} | a={:%} | b={:%},  train_kl={:f}, used_time:{:.2f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.total_batches_train,
                            total_loss / n_batches,
                            total_correct_preds / (n_batches * self.config.batch_size),
                            total_correct_preds_a / (n_batches * self.config.batch_size),
                            total_correct_preds_b / (n_batches * self.config.batch_size),
                            total_kl / n_batches,
                            time.time() - time_last),
                        end=' ')

                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass
        return step

    def set_global_tensor(self, training_tensor, regu_conv, regu_decay, regu_fc):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
        self.regularizer_decay = regu_decay
        self.regularizer_fc = regu_fc

    def fetch_weight(self, sess):
        """
        get all the parameters, including the
        :param sess:
        :return:
        """
        weight_dict = dict()
        weight_list = list()
        for layer in self.layers:
            weight_list.append(layer.get_params(sess))
        for params_dict in weight_list:
            for k, v in params_dict.items():
                weight_dict[k.split(':')[0]] = v
        for meta_key in self.meta_keys_with_default_val.keys():
            meta_key_in_weight = meta_key
            weight_dict[meta_key_in_weight] = self.meta_val(meta_key)
        return weight_dict

    def eval_once(self, sess, init, epoch):
        sess.run(init)
        total_loss = 0
        total_correct_preds = 0
        total_correct_preds_a = 0
        total_correct_preds_b = 0
        n_batches = 0
        time_start = time.time()
        try:
            while True:
                loss_batch, accuracy_batch, accuracy_batch_a, accuracy_batch_b = sess.run(
                    [self.op_loss, self.op_accuracy, self.op_accuracy_a, self.op_accuracy_b],
                    feed_dict={self.is_training: False})

                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                total_correct_preds_a += accuracy_batch_a
                total_correct_preds_b += accuracy_batch_b
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        accu_a = total_correct_preds_a / self.n_samples_val
        accu_b = total_correct_preds_b / self.n_samples_val
        print('\nEpoch:{:d}, val_acc={:%} | a={:%} | b={:%}, val_loss={:f}, used_time:{:.2f}s'.format(
            epoch + 1, accu, accu_a, accu_b, total_loss / n_batches, time_end - time_start))

        return accu, accu_a, accu_b

    def train(self, sess, n_epochs, task_name, lr=None):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step, task_name)
            accu, accu_a, accu_b = self.eval_once(sess, self.test_init, epoch)

            if (epoch + 1) % 1 == 0:
                if self.prune_method == 'info_bottle':
                    self.get_CR(sess)

            if (epoch + 1) % 10 == 0:
                if self.prune_method == 'info_bottle':
                    self.get_CR(sess)
                    save_path = '/local/home/david/Remote/models/model_weights/vgg_ib_' + self.task_name + '_' + str(
                        self.prune_threshold) + '_' + str(np.around(accu, decimals=6))
                else:
                    save_path = '/local/home/david/Remote/models/model_weights/vgg_' + self.task_name + '_' + str(
                        np.around(accu, decimals=6))
                self.save_weight(sess, save_path)

    def test(self, sess):
        sess.run(self.test_init)
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0

        try:
            while True:
                loss_batch, accuracy_batch = sess.run([self.op_loss, self.op_accuracy],
                                                      feed_dict={self.is_training: False})
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print(
            '\nTesting {}, val_acc={:%}, val_loss={:f}'.format(self.task_name, total_correct_preds / self.n_samples_val,
                                                               total_loss / n_batches))

    def get_CR(self, sess):
        # Obtain all masks
        masks = list()
        for layer in self.layers:
            if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib':
                masks += [layer.get_mask(threshold=self.prune_threshold)]

        masks = sess.run(masks)
        n_classes = self.Y.shape.as_list()[1]

        # how many channels/dims are prune in each layer
        prune_state = [np.sum(mask == 0) for mask in masks]

        total_params, pruned_params, remain_params = 0, 0, 0
        # for conv layers
        in_channels, in_pruned = 3, 0
        for n, n_out in enumerate([64, 64,
                                   128, 128,
                                   256, 256, 256,
                                   512, 512, 512,
                                   512, 512, 512]):
            # params between this and last layers
            n_params = in_channels * n_out * 9
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_state[n]) * 9
            remain_params += n_remain
            pruned_params += n_params - n_remain
            # for next layer
            in_channels = n_out
            in_pruned = prune_state[n]
        # for fc layers
        offset = len(prune_state) - 2
        for n, n_out in enumerate([4096, 4096]):
            n_params = in_channels * n_out
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_state[n + offset])
            remain_params += n_remain
            pruned_params += n_params - n_remain
            # for next layer
            in_channels = n_out
            in_pruned = prune_state[n + offset]
        total_params += in_channels * n_classes
        remain_params += (in_channels - in_pruned) * n_classes
        pruned_params += in_pruned * n_classes

        print('Total parameters: {}, Pruned parameters: {}, Remaining params:{}, Remain/Total params:{}, '
              'Each layer pruned: {}'.format(total_params, pruned_params, remain_params,
                                             float(total_params - pruned_params) / total_params, prune_state))
