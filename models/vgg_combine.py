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
                 ib_threshold=None, model_path=None):
        super(VGG_Combined, self).__init__(config)

        if task_name in ['celeba', 'celeba1', 'celeba2']:
            self.imgs_path = self.config.dataset_path + 'celeba/'
        else:
            self.imgs_path = self.config.dataset_path + task_name + '/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.cluster_res_list = cluster_res_list

        self.task_name = task_name
        self.is_musked = musk
        self.gamma = gamma

        if self.prune_method == 'info_bottle' and ib_threshold is not None:
            self.prune_threshold = ib_threshold

        if self.prune_method == 'info_bottle':
            self.kl_total_a = None
            self.kl_total_b = None

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

        if model_path and os.path.exists(model_path):
            # Directly load all weights
            self.weight_dict = pickle.load(open(model_path, 'rb'))
            print('[%s] Loading weight matrix in %s' % (datetime.now(), model_path))
            self.initial_weight = False
        else:
            # Use pre-train weights in conv, but init weights in fc
            self.weight_dict = self.construct_initial_weights(weight_a, weight_b, cluster_res_list)
            print('[%s] Initialize weight matrix' % (datetime.now()))
            self.initial_weight = True

        if self.prune_method == 'info_bottle' and 'conv1_1/AB/info_bottle/mu' not in self.weight_dict.keys():
            print('-----------------------初始化ib层权重-----------------------')
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib(cluster_res_list))

    def construct_initial_weights(self, weight_dict_a, weight_dict_b, cluster_res_list):
        dim_list = [64, 64,
                    128, 128,
                    256, 256, 256,
                    512, 512, 512,
                    512, 512, 512,
                    512, 512, self.n_classes]

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

        def get_expand(array, h=2, w=2, original_channel_num_a=512):
            res_list = list()
            step = h * w
            for i in range(step):
                for value in array:
                    if value < original_channel_num_a:
                        res_list += [value + i * original_channel_num_a]
                    else:
                        res_list += [
                            (value - original_channel_num_a) + (i + step) * original_channel_num_a]
            return res_list

        weight_dict = dict()

        for layer_index, layer_name in enumerate(['conv1_1', 'conv1_2',
                                                  'conv2_1', 'conv2_2',
                                                  'conv3_1', 'conv3_2', 'conv3_3',
                                                  'conv4_1', 'conv4_2', 'conv4_3',
                                                  'conv5_1', 'conv5_2', 'conv5_3',
                                                  'fc6', 'fc7', 'fc8']):
            # All bias
            bias = np.concatenate(
                (weight_dict_a[layer_name + '/biases'], weight_dict_b[layer_name + '/biases'])).astype(np.float32)

            # Obtain neuron list
            A = cluster_res_list[layer_index]['A']
            AB = cluster_res_list[layer_index]['AB']
            B = cluster_res_list[layer_index]['B']

            if layer_index == 0:
                weight = np.concatenate(
                    (weight_dict_a[layer_name + '/weights'], weight_dict_b[layer_name + '/weights']), axis=-1).astype(
                    np.float32)
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
                # Get all weights
                weight_a = weight_dict_a[layer_name + '/weights']
                weight_b = weight_dict_b[layer_name + '/weights']

                shape_weight_a = np.shape(weight_a)
                shape_weight_b = np.shape(weight_b)

                if len(shape_weight_a) == 4:
                    matrix_zero_left_down = np.zeros(shape=(3, 3, shape_weight_b[-2], shape_weight_a[-1]))
                    matrix_zero_right_up = np.zeros(shape=(3, 3, shape_weight_a[-2], shape_weight_b[-1]))
                else:
                    matrix_zero_left_down = np.zeros(shape=(shape_weight_b[-2], shape_weight_a[-1]))
                    matrix_zero_right_up = np.zeros(shape=(shape_weight_a[-2], shape_weight_b[-1]))

                weight_up = np.concatenate((weight_a, matrix_zero_right_up), axis=-1)
                weight_down = np.concatenate((matrix_zero_left_down, weight_b), axis=-1)
                weight = np.concatenate((weight_up, weight_down), axis=-2).astype(np.float32)

                # Obtain neuron list of last layer
                A_last = cluster_res_list[layer_index - 1]['A']
                AB_last = cluster_res_list[layer_index - 1]['AB']
                B_last = cluster_res_list[layer_index - 1]['B']

                # Init weights
                if layer_name.startswith('conv'):
                    # A
                    if get_signal(layer_index, 'A'):
                        weight_dict[layer_name + '/A/weights'] = weight[:, :, A_last + AB_last, :][:, :, :, A]

                    # AB
                    if get_signal(layer_index, 'AB'):
                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_dict[layer_name + '/AB/weights'] = weight[:, :, AB_last, :][:, :, :, AB]

                    # B
                    if get_signal(layer_index, 'B'):
                        weight_dict[layer_name + '/B/weights'] = weight[:, :, AB_last + B_last, :][:, :, :, B]

                elif layer_name.startswith('fc'):
                    # Fc layer

                    if layer_name == 'fc6':
                        # From conv to fc, times h*w
                        # !!!!!从conv层到全连接层的时候，原来的512->2048,而且其中的index=i, 投影变成了[4*i, 4i+4]...整体是变成了4倍
                        # 最后一层产生的feature map边长的平方
                        A_last = get_expand(A_last)
                        AB_last = get_expand(AB_last)
                        B_last = get_expand(B_last)

                    # New weights for neurons from A and AB to A
                    if get_signal(layer_index, 'A'):
                        weight_dict[layer_name + '/A/weights'] = weight[A_last + AB_last, :][:, A]

                    # The output layer does not have AB
                    if get_signal(layer_index, 'AB'):

                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_dict[layer_name + '/AB/weights'] = weight[AB_last, :][:, AB]

                    # New weights for neurons from AB and B to B
                    if get_signal(layer_index, 'B'):
                        weight_dict[layer_name + '/B/weights'] = weight[AB_last + B_last, :][:, B]

                # Biases
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/biases'] = bias[A]
                if get_signal(layer_index, 'AB'):
                    if get_signal(layer_index, 'AB'):
                        weight_dict[layer_name + '/AB/biases'] = bias[AB]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/biases'] = bias[B]

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
            if num_A != 0:
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
            if num_B != 0:
                weight_dict[name_layer + '/B/info_bottle/mu'] = np.random.normal(loc=1, scale=0.01,
                                                                                 size=[num_B]).astype(np.float32)

                weight_dict[name_layer + '/B/info_bottle/logD'] = np.random.normal(loc=-9, scale=0.01,
                                                                                   size=[num_B]).astype(np.float32)
        return weight_dict

    def get_conv(self, x, regu_conv):
        return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
                         is_musked=self.is_musked, regularizer_conv=regu_conv,
                         is_merge_bn=self.meta_val('is_merge_bn'))

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

        def get_conv(x, regu_conv):
            return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
                             is_musked=self.is_musked, regularizer_conv=regu_conv,
                             is_merge_bn=self.meta_val('is_merge_bn'))

        def get_ib(y, layer_type, kl_mult, partition_name):
            if self.prune_method == 'info_bottle':
                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult,
                                                      mask_threshold=self.prune_threshold)
                self.layers.append(ib_layer)
                y, ib_kld = ib_layer.layer_output
                self.kl_total += ib_kld

                if partition_name == 'A':
                    self.kl_total_a += ib_kld
                if partition_name == 'AB':
                    self.kl_total_a += ib_kld
                    self.kl_total_b += ib_kld
                if partition_name == 'B':
                    self.kl_total_a += ib_kld

            return y

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total = 0.
            self.kl_total_a = 0.
            self.kl_total_b = 0.

            layer_index = 0
            # the name of the layer and the coefficient of the kl divergence
            for layer_set in [('conv1_1', 1.0 / 32), ('conv1_2', 1.0 / 32), 'pooling',
                              ('conv2_1', 1.0 / 16), ('conv2_2', 1.0 / 16), 'pooling',
                              ('conv3_1', 1.0 / 8), ('conv3_2', 1.0 / 8), ('conv3_3', 1.0 / 8), 'pooling',
                              ('conv4_1', 1.0 / 4), ('conv4_2', 1.0 / 4), ('conv4_3', 1.0 / 4), 'pooling',
                              ('conv5_1', 4.0 / 2), ('conv5_2', 4.0 / 2), ('conv5_3', 4.0 / 2), 'pooling']:
                if layer_index == 0:
                    conv_name, kl_mult = layer_set

                    # 如果有A部分的话才进行连接
                    if get_signal(layer_index, 'A'):
                        with tf.variable_scope(conv_name + '/A'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_A = tf.nn.relu(conv.layer_output)

                            y_A = get_ib(y_A, 'C_ib', kl_mult, 'A')

                    if get_signal(layer_index, 'AB'):
                        with tf.variable_scope(conv_name + '/AB'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_AB = tf.nn.relu(conv.layer_output)

                            y_AB = get_ib(y_AB, 'C_ib', kl_mult, 'AB')

                    if get_signal(layer_index, 'B'):
                        with tf.variable_scope(conv_name + '/B'):
                            conv = get_conv(x, regu_conv=self.regularizer_conv)
                            self.layers.append(conv)
                            y_B = tf.nn.relu(conv.layer_output)

                            y_B = get_ib(y_B, 'C_ib', kl_mult, 'B')

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

                            y_A = get_ib(y_A, 'C_ib', kl_mult, 'A')

                    # AB
                    if get_signal(layer_index, 'AB'):
                        with tf.variable_scope(conv_name + '/AB'):

                            if get_signal(layer_index, 'fromAB'):
                                # From AB to AB
                                conv_AB = get_conv(y_AB_last, self.regularizer_conv)
                                self.layers.append(conv_AB)

                                y_AB = tf.nn.relu(conv_AB.layer_output)

                                y_AB = get_ib(y_AB, 'C_ib', kl_mult, 'AB')

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
                                conv = get_conv(y_B_last, self.regularizer_conv)
                            self.layers.append(conv)
                            y_B = tf.nn.relu(conv.layer_output)

                            y_B = get_ib(y_B, 'C_ib', kl_mult, 'B')

                elif layer_set == 'pooling':
                    # 因为在上一层已经出现了layer_index+=1,所以应该查看上一层是否出现了A,AB,B
                    if get_signal(layer_index - 1, 'A'):
                        y_A = tf.nn.max_pool(y_A_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index - 1, 'AB'):
                        y_AB = tf.nn.max_pool(y_AB_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index - 1, 'B'):
                        y_B = tf.nn.max_pool(y_B_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 如果是pooling的时候layer_index指向下一层，所以需要变成判断上一层是不是有A，AB，B
                y_A_last = None
                y_AB_last = None
                y_B_last = None
                if layer_set == 'pooling':
                    # Record the output of this layer
                    if get_signal(layer_index - 1, 'A'):
                        y_A_last = y_A
                    if get_signal(layer_index - 1, 'AB'):
                        y_AB_last = y_AB
                    if get_signal(layer_index - 1, 'B'):
                        y_B_last = y_B
                else:
                    # 如果是正常层的话，layer_index对应，所以可以正常判断
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
            # 到这一层的时候layer_index刚刚经过pooling，所以指向的是fc层的状态，这里面应该是判断上一层
            if get_signal(layer_index - 1, 'A'):
                y_A_last = tf.contrib.layers.flatten(y_A_last)
            if get_signal(layer_index - 1, 'AB'):
                y_AB_last = tf.contrib.layers.flatten(y_AB_last)
            if get_signal(layer_index - 1, 'B'):
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

                        y_A = get_ib(y_A, 'F_ib', self.gamma, 'A')

                # AB
                if get_signal(layer_index, 'AB'):
                    with tf.variable_scope(fc_name + '/AB'):
                        if get_signal(layer_index, 'fromAB'):
                            # From AB to AB
                            fc_layer_AB = FullConnectedLayer(y_AB_last, self.weight_dict,
                                                             regularizer_fc=self.regularizer_fc)
                            self.layers.append(fc_layer_AB)

                            y_AB = tf.nn.relu(fc_layer_AB.layer_output)

                            y_AB = get_ib(y_AB, 'F_ib', self.gamma, 'AB')

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
                        self.layers.append(fc_layer)
                        y_B = tf.nn.relu(fc_layer.layer_output)

                        y_B = get_ib(y_B, 'F_ib', self.gamma, 'B')

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
            self.op_loss_a = mae_loss_a + l2_loss + self.kl_factor * self.kl_total_a
            self.op_loss_b = mae_loss_b + l2_loss + self.kl_factor * self.kl_total_b
        else:
            self.op_loss = mae_loss + l2_loss
            self.op_loss_a = mae_loss_a + l2_loss
            self.op_loss_b = mae_loss_b + l2_loss

    def optimize(self):
        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # Create a optimizer
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=0.9,
                                                      use_nesterov=True)

                self.op_opt = self.opt.minimize(self.op_loss)

                self.op_opt_a = self.opt.minimize(self.op_loss_a)

                self.op_opt_b = self.opt.minimize(self.op_loss_b)

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

    def train_one_epoch(self, sess, init, epoch, step, task_name, time_stamp):
        sess.run(init)
        total_loss = 0
        total_kl = 0
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
                    _, loss, kl = sess.run([op_opt, self.op_loss, self.kl_total], feed_dict={self.is_training: True})
                    total_kl += kl
                else:
                    _, loss = sess.run([op_opt, self.op_loss], feed_dict={self.is_training: True})
                step += 1
                total_loss += loss
                n_batches += 1

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, train_kl={:f}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        total_loss / n_batches,
                        total_kl / n_batches,
                        time.time() - time_last)
                    print('\r' + str_, end=' ')

                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass

        # 写文件
        with open('/local/home/david/Remote/models/model_weights/log_vgg_combine_' + time_stamp, 'a+') as f:
            f.write(str_ + '\n')

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

    def eval_once(self, sess, init, epoch, time_stamp=None):
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

        str_ = 'Epoch:{:d}, val_acc={:%} | a={:%} | b={:%}, val_loss={:f}, used_time:{:.2f}s'.format(
            epoch + 1, accu, accu_a, accu_b, total_loss / n_batches, time_end - time_start)

        print('\n' + str_)

        # 写文件
        if time_stamp is not None:
            with open('/local/home/david/Remote/models/model_weights/log_vgg_combine_' + time_stamp, 'a+') as f:
                f.write(str_ + '\n')

        return accu, accu_a, accu_b

    def train_one_epoch_individual(self, sess, init, epoch, step, time_stamp):
        sess.run(init)
        total_loss_a = 0
        total_loss_b = 0
        total_kl_a = 0
        total_kl_b = 0
        n_batches = 0

        time_last = time.time()

        try:
            while True:
                if self.prune_method == 'info_bottle':
                    _, loss_a, kl_a = sess.run([self.op_opt_a, self.op_loss_a, self.kl_total_a],
                                               feed_dict={self.is_training: True})
                    total_kl_a += kl_a
                    _, loss_b, kl_b = sess.run([self.op_opt_b, self.op_loss_b, self.kl_total_b],
                                               feed_dict={self.is_training: True})
                    total_kl_b += kl_b

                else:
                    _, loss_a = sess.run([self.op_opt_a, self.op_loss_a], feed_dict={self.is_training: True})
                    _, loss_b = sess.run([self.op_opt_b, self.op_loss_b], feed_dict={self.is_training: True})
                step += 2
                total_loss_a += loss_a
                total_loss_b += loss_b
                n_batches += 2

                if n_batches % 10 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss_a={:f}, curr_loss_b={:f}, train_kl_a={:f}, train_kl_b={:f}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        total_loss_a / n_batches * 2,
                        total_loss_b / n_batches * 2,
                        total_kl_a / n_batches * 2,
                        total_kl_b / n_batches * 2,
                        time.time() - time_last)
                    print('\r' + str_, end=' ')

                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass

        # 写文件
        with open('/local/home/david/Remote/models/model_weights/log_vgg_combine_' + time_stamp, 'a+') as f:
            f.write(str_ + '\n')

        return step

    def train_individual(self, sess, n_epochs, lr, time_stamp):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch_individual(sess, self.train_init, epoch, step, time_stamp)
            accu, accu_a, accu_b = self.eval_once(sess, self.test_init, epoch, time_stamp)

            if self.prune_method == 'info_bottle':
                cr = self.get_CR(sess, self.cluster_res_list, time_stamp)

            if (epoch + 1) % 10 == 0:
                if self.prune_method == 'info_bottle':
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_combine_ib_' + self.task_name + '_' + str(
                        self.prune_threshold) + '_' + str(
                        np.around(accu, decimals=4)) + '-' + str(np.around(accu_a, decimals=4)) + '-' + str(
                        np.around(accu_b,
                                  decimals=4)) + '_cr-' + str(
                        np.around(cr, decimals=4))
                else:
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_combine_' + self.task_name + '_' + str(
                        np.around(accu, decimals=4))
                self.save_weight(sess, save_path)

    def train(self, sess, n_epochs, task_name, lr, time_stamp):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        count_86 = 4

        # 为了在没加vib的时候方便保存参数
        cr = 1.

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step, task_name, time_stamp)
            accu, accu_a, accu_b = self.eval_once(sess, self.test_init, epoch, time_stamp)

            if self.prune_method == 'info_bottle':
                cr = self.get_CR(sess, self.cluster_res_list, time_stamp)

            if (epoch + 1) % 10 == 0:
                if self.prune_method == 'info_bottle':
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_combine_ib_' + self.task_name + '_' + str(
                        self.prune_threshold) + '_' + str(
                        np.around(accu, decimals=4)) + '-' + str(np.around(accu_a, decimals=4)) + '-' + str(
                        np.around(accu_b,
                                  decimals=4)) + '_cr-' + str(
                        np.around(cr, decimals=4))
                else:
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_combine_' + self.task_name + '_' + str(
                        np.around(accu, decimals=4))
                self.save_weight(sess, save_path)

            # if accu < 0.875:
            #     count_86 -= 1
            #     if count_86 <= 0:
            #         break

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

    def get_CR(self, sess, cluster_res_list, time_stamp):
        layers_name = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']

        name_len_dict = {'conv1_1': 72, 'conv1_2': 72,
                         'conv2_1': 36, 'conv2_2': 36,
                         'conv3_1': 18, 'conv3_2': 18, 'conv3_3': 18,
                         'conv4_1': 9, 'conv4_2': 9, 'conv4_3': 9,
                         'conv5_1': 4, 'conv5_2': 4, 'conv5_3': 4}

        # 都是做mask之前的入度和出度
        num_out_channel_dict = dict()
        num_in_channel_dict = dict()

        for layer_index, cluster_layer in enumerate(cluster_res_list):
            layer_name = layers_name[layer_index]
            num_A = len(cluster_layer['A'])
            num_out_channel_dict[layer_name + '/A'] = num_A

            num_AB = len(cluster_layer['AB'])
            num_out_channel_dict[layer_name + '/AB'] = num_AB

            num_B = len(cluster_layer['B'])
            num_out_channel_dict[layer_name + '/B'] = num_B

            if layer_index == 0:
                num_in_channel_dict[layer_name + '/A'] = 3
                num_in_channel_dict[layer_name + '/AB'] = 3
                num_in_channel_dict[layer_name + '/B'] = 3
            else:
                num_A_last = len(cluster_res_list[layer_index - 1]['A'])
                num_AB_last = len(cluster_res_list[layer_index - 1]['AB'])
                num_B_last = len(cluster_res_list[layer_index - 1]['B'])

                num_in_channel_dict[layer_name + '/A'] = num_A_last + num_AB_last
                num_in_channel_dict[layer_name + '/AB'] = num_AB_last
                num_in_channel_dict[layer_name + '/B'] = num_AB_last + num_B_last

        # 输出被prune的数量
        masks = list()
        layers_type = list()
        layers_name_list = list()
        for layer in self.layers:
            if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib':
                # 和musks是一一对应的关系
                layers_name_list += [layer.layer_name]
                layers_type += [layer.layer_type]
                masks += [layer.get_mask(threshold=self.prune_threshold)]

        # 获得具体的mask
        masks = sess.run(masks)

        # how many channels/dims are prune in each layer
        prune_state = [np.sum(mask == 0) for mask in masks]
        original_state = [len(mask) for mask in masks]

        # 记录一下每一层的出度被剪枝了多少
        out_prune_dict = dict()
        for i, layer_name in enumerate(layers_name_list):
            # 这一层被剪枝了多少
            out_prune_dict[layer_name] = prune_state[i]

        # 记录这一层被剪枝了多少个神经元
        in_prune_dict = dict()
        for i, layer_name in enumerate(layers_name):
            if i == 0:
                in_prune_dict[layer_name + '/A'] = 0
                in_prune_dict[layer_name + '/AB'] = 0
                in_prune_dict[layer_name + '/B'] = 0
                continue

            layer_name_last = layers_name_list[i - 1]

            # 输入被剪枝掉了多少!!!!
            in_prune_dict[layer_name + '/A'] = out_prune_dict.get(layer_name_last + '/A', 0) + out_prune_dict.get(
                layer_name_last + '/AB', 0)
            if layer_name != 'fc8':
                in_prune_dict[layer_name + '/AB'] = out_prune_dict.get(layer_name_last + '/AB', 0)
            in_prune_dict[layer_name + '/B'] = out_prune_dict.get(layer_name_last + '/AB', 0) + out_prune_dict.get(
                layer_name_last + '/B', 0)

        total_params, pruned_params, remain_params = 0, 0, 0
        total_flops, remain_flops, pruned_flops = 0, 0, 0

        for index, layer_name in enumerate(layers_name_list):
            num_in = num_in_channel_dict[layer_name]
            num_out = num_out_channel_dict[layer_name]

            num_out_prune = out_prune_dict.get(layer_name)
            num_in_prune = in_prune_dict.get(layer_name)

            if layers_type[index] == 'C_ib':
                total_params += num_in * num_out * 9
                remain_params += (num_in - num_in_prune) * (num_out - num_out_prune) * 9

                # FLOPs
                for key in name_len_dict.keys():
                    if layer_name.startswith(key):
                        M = name_len_dict[key]
                        break
                total_flops += 2 * (9 * num_in + 1) * M * M * num_out
                remain_flops += 2 * (9 * (num_in - num_in_prune) + 1) * M * M * (num_out - num_out_prune)

            elif layers_type[index] == 'F_ib':
                total_params += num_in * num_out
                remain_params += (num_in - num_in_prune) * (num_out - num_out_prune)

                # FLOPs
                total_flops += (2 * num_in - 1) * num_out
                remain_flops += (2 * (num_in - num_in_prune) - 1) * (num_out - num_out_prune)

        # output layer
        total_params += num_in_channel_dict['fc8/A'] * 20
        remain_params += (num_in_channel_dict['fc8/A'] - in_prune_dict['fc8/A']) * 20
        total_params += num_in_channel_dict['fc8/B'] * 20
        remain_params += (num_in_channel_dict['fc8/B'] - in_prune_dict['fc8/B']) * 20

        # FLOPs
        total_flops += (2 * num_in_channel_dict['fc8/A'] - 1) * 20
        remain_flops += (2 * (num_in_channel_dict['fc8/A'] - in_prune_dict['fc8/A']) - 1) * 20
        total_flops += (2 * num_in_channel_dict['fc8/B'] - 1) * 20
        remain_flops += (2 * (num_in_channel_dict['fc8/B'] - in_prune_dict['fc8/B']) - 1) * 20

        pruned_params = total_params - remain_params
        pruned_flops = total_flops - remain_flops

        str_1 = 'Total parameters: {}, Pruned parameters: {}, Remaining params:{}, Remain/Total params:{}'.format(
            total_params, pruned_params, remain_params,
            np.around(float(total_params - pruned_params) / total_params, decimals=5))

        res_each_layer = list()
        for i in range(len(prune_state)):
            res_each_layer += [str(prune_state[i]) + '/' + str(original_state[i])]

        str_2 = 'Each layer pruned: {}'.format(res_each_layer).replace("'", "")

        str_3 = 'Total FLOPs: {}, Pruned FLOPs: {}, Remaining FLOPs: {}, Remain/Total FLOPs:{}'.format(total_flops,
                                                                                                       pruned_flops,
                                                                                                       remain_flops,
                                                                                                       np.around(
                                                                                                           float(
                                                                                                               total_flops - pruned_flops) / total_flops,
                                                                                                           decimals=5))
        print(str_1)
        print(str_2)
        print(str_3)

        if time_stamp is not None:
            with open('/local/home/david/Remote/models/model_weights/log_vgg_combine_' + time_stamp, 'a+') as f:
                f.write(str_1 + '\n')
                f.write(str_2 + '\n')
                f.write(str_3 + '\n')

        return np.around(float(total_params - pruned_params) / total_params, decimals=5)
