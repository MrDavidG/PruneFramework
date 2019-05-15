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

from layers.fc_layer import FullConnectedLayer
from data_loader.image_data_generator import ImageDataGenerator
import numpy as np

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VGG_Combined():

    def __init__(self, config, task_name, weight_a, weight_b, cluster_res_list, signal_list, musk, gamma=1.,
                 ib_threshold=None):

        self.config = config
        self.imgs_path = self.config.dataset_path + 'celeba/'

        self.task_name = task_name

        self.gamma = gamma

        self.op_logits_a = None
        self.op_logits_b = None

        self.layers = list()

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        self.regularizer_decay = None
        self.is_training = None
        self.regularizer_conv = None
        self.regularizer_fc = None

        self.signal_list = signal_list

        self.weight_dict = self.construct_initial_weights(weight_a, weight_b, cluster_res_list)


    def set_global_tensor(self, training_tensor, regu_conv, regu_fc,regularizer_decay):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
        self.regularizer_fc = regu_fc

    def load_dataset(self):
        dataset_train, dataset_val, self.total_batches_train, self.n_samples_train, self.n_samples_val = ImageDataGenerator.load_dataset(
            self.config.batch_size, self.config.cpu_cores, self.task_name, self.imgs_path)
        self.train_init, self.test_init, self.X, self.Y = ImageDataGenerator.dataset_iterator(
            dataset_train,
            dataset_val)

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
                        # From A to AB
                        if get_signal(layer_index, 'fromA'):
                            weight_dict[layer_name + '/AB/A/weights'] = weight[:, :, A_last, :][:, :, :, AB]

                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_dict[layer_name + '/AB/AB/weights'] = weight[:, :, AB_last, :][:, :, :, AB]

                        # From B to AB
                        if get_signal(layer_index, 'fromB'):
                            weight_dict[layer_name + '/AB/B/weights'] = weight[:, :, B_last, :][:, :, :AB]

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
                        # New weights for neurons from last layer to AB
                        # From A to AB
                        if get_signal(layer_index, 'fromA'):
                            weight_dict[layer_name + '/AB/A/weights'] = weight[A_last, :][:, AB]

                        # From AB to AB
                        if get_signal(layer_index, 'fromAB'):
                            weight_dict[layer_name + '/AB/AB/weights'] = weight[AB_last, :][:, AB]

                        # From B to AB
                        if get_signal(layer_index, 'fromB'):
                            weight_dict[layer_name + '/AB/B/weights'] = weight[B_last, :][:, AB]

                    # New weights for neurons from AB and B to B
                    if get_signal(layer_index, 'B'):
                        weight_dict[layer_name + '/B/weights'] = weight[AB_last + B_last, :][:, B]

                # Biases
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/biases'] = bias[A]
                if get_signal(layer_index, 'AB'):
                    if get_signal(layer_index, 'AB'):
                        weight_dict[layer_name + '/AB/AB/biases'] = bias[AB]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/biases'] = bias[B]

        return weight_dict

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

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

                    conv = tf.nn.conv2d(x, self.weight_dict['conv1_1/AB/weights'], [1, 1, 1, 1], padding='SAME') + self.weight_dict['conv1_1/AB/biases']

                    self.layers.append(conv)

                    y_AB = tf.nn.relu(conv)

                    layer_index+=1
                elif layer_set != 'pooling':
                    conv_name, kl_mult = layer_set

                    conv_AB = tf.nn.conv2d(y_AB_last, self.weight_dict[conv_name + '/AB/AB/weights'], [1, 1, 1, 1], padding='SAME') + self.weight_dict[conv_name + '/AB/AB/biases']

                    self.layers.append(conv_AB)
                    y_AB = tf.nn.relu(conv_AB)

                    layer_index+=1
                elif layer_set == 'pooling':

                    y_AB = tf.nn.max_pool(y_AB_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


                y_AB_last = y_AB

