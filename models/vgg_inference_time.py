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

# gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'


class VGG_InferenceTime(BaseModel):
    def __init__(self, config, task_name, weight_original, mask_res_list, musk=False, ib_threshold=None):
        super(VGG_InferenceTime, self).__init__(config)

        if task_name in ['celeba', 'celeba1', 'celeba2']:
            self.imgs_path = self.config.dataset_path + 'celeba/'
        elif task_name in ['deepfashion', 'deepfashion1', 'deepfashion2']:
            self.img_path = self.config.dataset_path + 'deepfashion/'
        else:
            self.imgs_path = self.config.dataset_path + task_name + '/'

        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name
        self.is_musked = musk

        if self.prune_method == 'info_bottle':
            self.prune_threshold = ib_threshold

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        self.weight_dict = self.construct_initial_weights(weight_original, mask_res_list)

    def construct_initial_weights(self, weight_original, mask_res_list):

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

            # Obtain neuron list
            neurons = mask_res_list[layer_index]

            # All weight and bias
            weight = weight_original[layer_name + '/weights'].astype(np.float32)
            bias = weight_original[layer_name + '/biases'].astype(np.float32)

            if layer_index == 0:
                # The first layer
                weight_dict[layer_name + '/weights'] = weight[:, :, :, neurons]

            else:
                # Obtain neuron list of last layer
                neurons_last = mask_res_list[layer_index - 1]

                # Init weights
                if layer_name.startswith('conv'):
                    weight_dict[layer_name + '/weights'] = weight[:, :, neurons_last, :][:, :, :, neurons]
                elif layer_name.startswith('fc'):
                    # Fc layer
                    if layer_name == 'fc6':
                        # From conv to fc, times h*w
                        neurons_last = get_expand(neurons_last)
                    weight_dict[layer_name + '/weights'] = weight[neurons_last, :][:, neurons]

            # Biases
            weight_dict[layer_name + '/biases'] = bias[neurons]

            # mu, logD
            if layer_index != 15:
                mu = weight_original[layer_name + '/info_bottle/mu'].astype(np.float32)
                logD = weight_original[layer_name + '/info_bottle/logD'].astype(np.float32)
                weight_dict[layer_name + '/info_bottle/mu'] = mu[neurons]
                weight_dict[layer_name + '/info_bottle/logD'] = logD[neurons]

        return weight_dict

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_conv(x, regu_conv):
            return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
                             is_musked=self.is_musked, regularizer_conv=regu_conv,
                             is_merge_bn=self.meta_val('is_merge_bn'))

        def get_ib(y, layer_type, kl_mult):
            if self.prune_method == 'info_bottle':
                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult,
                                                      mask_threshold=self.prune_threshold)
                self.layers.append(ib_layer)
                y, _ = ib_layer.layer_output
            return y

        self.layers.clear()

        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            y = self.X

            # the name of the layer and the coefficient of the kl divergence
            for layer_name in ['conv1_1', 'conv1_2', 'pooling',
                               'conv2_1', 'conv2_2', 'pooling',
                               'conv3_1', 'conv3_2', 'conv3_3', 'pooling',
                               'conv4_1', 'conv4_2', 'conv4_3', 'pooling',
                               'conv5_1', 'conv5_2', 'conv5_3', 'pooling']:
                if layer_name != 'pooling':
                    with tf.variable_scope(layer_name):
                        conv = get_conv(y, regu_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)
                        y = get_ib(y, 'C_ib', 1.)
                else:
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # From conv to fc layer
            y = tf.contrib.layers.flatten(y)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
                    y = get_ib(y, 'F_ib', 1.)

            with tf.variable_scope('fc8'):
                fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                y = fc_layer.layer_output

            self.op_logits = tf.nn.tanh(y)

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
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
        self.evaluate()

    def set_global_tensor(self, training_tensor, regu_conv, regu_fc):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
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
        total_correct_preds = 0
        n_batches = 0
        time_start = time.time()
        try:
            while True:
                accuracy_batch = sess.run(self.op_accuracy, feed_dict={self.is_training: False})

                total_correct_preds += accuracy_batch
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        print('\nEpoch:{:d}, val_acc={:%}, used_time:{:.2f}s'.format(epoch + 1, accu, time_end - time_start))
        return accu


class VGG_InferenceTime_Combine(BaseModel):
    def __init__(self, config, task_name, weight_dict_1, weight_dict_2, mask_res_list_1, mask_res_list_2, musk=False,
                 ib_threshold=None):
        super(VGG_InferenceTime_Combine, self).__init__(config)

        if task_name in ['celeba', 'celeba1', 'celeba2']:
            self.imgs_path = self.config.dataset_path + 'celeba/'
        elif task_name in ['deepfashion', 'deepfashion1', 'deepfashion2']:
            self.img_path = self.config.dataset_path + 'deepfashion/'
        else:
            self.imgs_path = self.config.dataset_path + task_name + '/'

        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name
        self.is_musked = musk

        self.op_logits_1 = None
        self.op_logits_2 = None

        self.op_accuracy_1 = None
        self.op_accuracy_2 = None

        if self.prune_method == 'info_bottle':
            self.prune_threshold = ib_threshold

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        self.weight_dict = self.construct_initial_weights(weight_dict_1, weight_dict_2,
                                                          mask_res_list_1,
                                                          mask_res_list_2)

    def construct_initial_weights(self, weight_dict_1, weight_dict_2, mask_res_list_1, mask_res_list_2):
        # 分别得到两个模型进行mask之后的结果
        w1 = self.get_weight_dict_after_mask(weight_dict_1, mask_res_list_1)
        w2 = self.get_weight_dict_after_mask(weight_dict_2, mask_res_list_2)
        # 这里是已经运行完了mask之后的结果

        # Rename
        weight_dict = dict()
        for layer in ['conv1_1', 'conv1_2',
                      'conv2_1', 'conv2_2',
                      'conv3_1', 'conv3_2', 'conv3_3',
                      'conv4_1', 'conv4_2', 'conv4_3',
                      'conv5_1', 'conv5_2', 'conv5_3',
                      'fc6', 'fc7', 'fc8']:
            weight_dict[layer + '/1/weights'] = w1[layer + '/weights']
            weight_dict[layer + '/1/biases'] = w1[layer + '/biases']
            if layer != 'fc8':
                weight_dict[layer + '/1/info_bottle/mu'] = w1[layer + '/info_bottle/mu']
                weight_dict[layer + '/1/info_bottle/logD'] = w1[layer + '/info_bottle/logD']

        for layer in ['conv1_1', 'conv1_2',
                      'conv2_1', 'conv2_2',
                      'conv3_1', 'conv3_2', 'conv3_3',
                      'conv4_1', 'conv4_2', 'conv4_3',
                      'conv5_1', 'conv5_2', 'conv5_3',
                      'fc6', 'fc7', 'fc8']:
            weight_dict[layer + '/2/weights'] = w2[layer + '/weights']
            weight_dict[layer + '/2/biases'] = w2[layer + '/biases']
            if layer != 'fc8':
                weight_dict[layer + '/2/info_bottle/mu'] = w2[layer + '/info_bottle/mu']
                weight_dict[layer + '/2/info_bottle/logD'] = w2[layer + '/info_bottle/logD']

        weight_dict['is_merge_bn'] = True

        return weight_dict

    def get_weight_dict_after_mask(self, weight_original, mask_res_list):

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

            # Obtain neuron list
            neurons = mask_res_list[layer_index]

            # All weight and bias
            weight = weight_original[layer_name + '/weights'].astype(np.float32)
            bias = weight_original[layer_name + '/biases'].astype(np.float32)

            if layer_index == 0:
                # The first layer
                weight_dict[layer_name + '/weights'] = weight[:, :, :, neurons]

            else:
                # Obtain neuron list of last layer
                neurons_last = mask_res_list[layer_index - 1]

                # Init weights
                if layer_name.startswith('conv'):
                    weight_dict[layer_name + '/weights'] = weight[:, :, neurons_last, :][:, :, :, neurons]
                elif layer_name.startswith('fc'):
                    # Fc layer
                    if layer_name == 'fc6':
                        # From conv to fc, times h*w
                        neurons_last = get_expand(neurons_last)
                    weight_dict[layer_name + '/weights'] = weight[neurons_last, :][:, neurons]

            # Biases
            weight_dict[layer_name + '/biases'] = bias[neurons]

            # mu, logD
            if layer_index != 15:
                mu = weight_original[layer_name + '/info_bottle/mu'].astype(np.float32)
                logD = weight_original[layer_name + '/info_bottle/logD'].astype(np.float32)
                weight_dict[layer_name + '/info_bottle/mu'] = mu[neurons]
                weight_dict[layer_name + '/info_bottle/logD'] = logD[neurons]

        return weight_dict

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_conv(x, regu_conv):
            return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
                             is_musked=self.is_musked, regularizer_conv=regu_conv,
                             is_merge_bn=self.meta_val('is_merge_bn'))

        def get_ib(y, layer_type, kl_mult):
            if self.prune_method == 'info_bottle':
                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult,
                                                      mask_threshold=self.prune_threshold)
                self.layers.append(ib_layer)
                y, _ = ib_layer.layer_output
            return y

        self.layers.clear()

        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            # model 1
            y1 = self.X
            # the name of the layer and the coefficient of the kl divergence
            for layer_name in ['conv1_1', 'conv1_2', 'pooling',
                               'conv2_1', 'conv2_2', 'pooling',
                               'conv3_1', 'conv3_2', 'conv3_3', 'pooling',
                               'conv4_1', 'conv4_2', 'conv4_3', 'pooling',
                               'conv5_1', 'conv5_2', 'conv5_3', 'pooling']:
                if layer_name != 'pooling':
                    with tf.variable_scope(layer_name + '/1'):
                        conv = get_conv(y1, self.regularizer_conv)
                        self.layers.append(conv)
                        y1 = tf.nn.relu(conv.layer_output)
                        y1 = get_ib(y1, 'C_ib', 1.)
                else:
                    y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # From conv to fc layer
            y1 = tf.contrib.layers.flatten(y1)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name + '/1'):
                    fc_layer = FullConnectedLayer(y1, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y1 = tf.nn.relu(fc_layer.layer_output)
                    y1 = get_ib(y1, 'F_ib', 1.)

            with tf.variable_scope('fc8' + '/1'):
                fc_layer = FullConnectedLayer(y1, self.weight_dict, regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                y1 = fc_layer.layer_output

            self.op_logits_1 = tf.nn.tanh(y1)

            # model 2
            y2 = self.X
            # the name of the layer and the coefficient of the kl divergence
            for layer_name in ['conv1_1', 'conv1_2', 'pooling',
                               'conv2_1', 'conv2_2', 'pooling',
                               'conv3_1', 'conv3_2', 'conv3_3', 'pooling',
                               'conv4_1', 'conv4_2', 'conv4_3', 'pooling',
                               'conv5_1', 'conv5_2', 'conv5_3', 'pooling']:
                if layer_name != 'pooling':
                    with tf.variable_scope(layer_name + '/2'):
                        conv = get_conv(y2, self.regularizer_conv)
                        self.layers.append(conv)
                        y2 = tf.nn.relu(conv.layer_output)
                        y2 = get_ib(y2, 'C_ib', 1.)
                else:
                    y2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # From conv to fc layer
            y2 = tf.contrib.layers.flatten(y2)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name + '/2'):
                    fc_layer = FullConnectedLayer(y2, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y2 = tf.nn.relu(fc_layer.layer_output)
                    y2 = get_ib(y2, 'F_ib', 1.)

            with tf.variable_scope('fc8' + '/2'):
                fc_layer = FullConnectedLayer(y2, self.weight_dict, regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                y2 = fc_layer.layer_output

            self.op_logits_2 = tf.nn.tanh(y2)

            self.op_logits = tf.nn.tanh(tf.concat((y1, y2), axis=1))

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
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
        self.evaluate()

    def set_global_tensor(self, training_tensor, regu_conv, regu_fc):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
        self.regularizer_fc = regu_fc

    # def fetch_weight(self, sess):
    #     """
    #     get all the parameters, including the
    #     :param sess:
    #     :return:
    #     """
    #     weight_dict = dict()
    #     weight_list = list()
    #     for layer in self.layers:
    #         weight_list.append(layer.get_params(sess))
    #     for params_dict in weight_list:
    #         for k, v in params_dict.items():
    #             weight_dict[k.split(':')[0]] = v
    #     for meta_key in self.meta_keys_with_default_val.keys():
    #         meta_key_in_weight = meta_key
    #         weight_dict[meta_key_in_weight] = self.meta_val(meta_key)
    #     return weight_dict

    def eval_once(self, sess, init, epoch):
        sess.run(init)
        total_correct_preds = 0
        n_batches = 0
        time_start = time.time()
        try:
            while True:
                accuracy_batch = sess.run(self.op_accuracy, feed_dict={self.is_training: False})

                total_correct_preds += accuracy_batch
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        print('\nEpoch:{:d}, val_acc={:%}, used_time:{:.2f}s'.format(epoch + 1, accu, time_end - time_start))
        return accu


# get_mask的np版本
def get_mask(mu, logD, threshold=0):
    # logalpha: [dim]
    logalpha = logD - np.log(np.power(mu, 2) + 1e-8)
    mask = (logalpha < threshold).astype(np.float32)
    return mask


def get_mask_result(model_path, n_class, ib_threshold):
    """
    根据model_path获得对应的weight以及mask后剩下的神经元的序号
    :param model_path:
    :param n_class:
    :param ib_threshold:
    :return:
    """
    weight_dict = pickle.load(open(model_path, 'rb'))

    mask_res_list = list()
    for layer_index, layer_name in enumerate(['conv1_1', 'conv1_2',
                                              'conv2_1', 'conv2_2',
                                              'conv3_1', 'conv3_2', 'conv3_3',
                                              'conv4_1', 'conv4_2', 'conv4_3',
                                              'conv5_1', 'conv5_2', 'conv5_3',
                                              'fc6', 'fc7']):
        mu = weight_dict[layer_name + '/info_bottle/mu']
        logD = weight_dict[layer_name + '/info_bottle/logD']

        # 得到这一层的mask
        mask = get_mask(mu, logD, ib_threshold)

        index_list = list()
        for index, value in enumerate(mask):
            if value != 0:
                index_list += [index]
        # 留下来的神经元的序号(剩下来的序号）
        mask_res_list.append(index_list)

    mask_res_list.append(list(np.arange(n_class)))

    return weight_dict, mask_res_list


def get_inference_time(ib_threshold, n_class, task_name, model_path):
    """
    只能获得不分叉模型的inference time(celeba1/celeba2/celeba+vib)
    :param ib_threshold:
    :param n_class:
    :param task_name:
    :param model_path:
    :return:
    """

    weight_dict, mask_res_list = get_mask_result(model_path=model_path, n_class=n_class, ib_threshold=ib_threshold)

    # 接下来进行重新inference
    config = process_config("../configs/ib_vgg.json")

    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    print('Training on task {:s}'.format(task_name))
    tf.reset_default_graph()
    # session for training

    session = tf.Session(config=gpu_config)

    training = tf.placeholder(dtype=tf.bool, name='training')

    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.00)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.00)

    # Train
    model = VGG_InferenceTime(config, task_name, weight_dict, mask_res_list, musk=False, ib_threshold=10000)
    model.set_global_tensor(training, regularizer_conv, regularizer_fc)
    model.build()

    session.run(tf.global_variables_initializer())

    model.eval_once(session, model.test_init, -1)


def get_inference_time_two(ib_threshold, n_class, task_name, model_path_1, model_path_2):
    weight_dict_1, mask_res_list_1 = get_mask_result(model_path=model_path_1, n_class=n_class,
                                                     ib_threshold=ib_threshold)
    weight_dict_2, mask_res_list_2 = get_mask_result(model_path=model_path_2, n_class=n_class,
                                                     ib_threshold=ib_threshold)

    # 接下来进行重新inference
    config = process_config("../configs/ib_vgg.json")

    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    print('Training on task {:s}'.format(task_name))
    tf.reset_default_graph()
    # session for training

    session = tf.Session(config=gpu_config)

    training = tf.placeholder(dtype=tf.bool, name='training')

    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.00)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.00)

    # Train
    model = VGG_InferenceTime_Combine(config, task_name, weight_dict_1, weight_dict_2, mask_res_list_1, mask_res_list_2,
                                      musk=False, ib_threshold=10000)
    model.set_global_tensor(training, regularizer_conv, regularizer_fc)
    model.build()

    session.run(tf.global_variables_initializer())

    model.eval_once(session, model.test_init, -1)


if __name__ == '__main__':
    # 测试一个模型的效果
    get_inference_time(ib_threshold=0.01,
                       n_class=20,
                       task_name='celeba1',
                       # model_path='/local/home/david/Remote/models/model_weights/best_vgg512_ib_celeba1_0.01_0.895017_cr-0.01538')
    # model_path = '/local/home/david/Remote/models/model_weights/best_vgg512_ib_celeba2_0.01_0.879943_cr-0.01257')
    model_path = '/local/home/david/Remote/models/model_weights/best_vgg512_ib_celeba_0.01_0.884827_cr-0.01738')

    # 测试两个模型同时的效果
    # get_inference_time_two(ib_threshold=0.01,
    #                        n_class=20,
    #                        task_name='celeba',
    #                        model_path_1='/local/home/david/Remote/models/model_weights/best_vgg512_ib_celeba1_0.01_0.895017_cr-0.01538',
    #                        model_path_2='/local/home/david/Remote/models/model_weights/best_vgg512_ib_celeba2_0.01_0.879943_cr-0.01257')
