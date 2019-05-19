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


class VGG_InferenceTime(BaseModel):
    def __init__(self, config, task_name, weight_original, mask_res_list, musk=False,
                 ib_threshold=None):
        super(VGG_InferenceTime, self).__init__(config)

        if task_name in ['celeba', 'celeba1', 'celeba2']:
            self.imgs_path = self.config.dataset_path + 'celeba/'
        else:
            self.imgs_path = self.config.dataset_path + task_name + '/'

        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name
        self.is_musked = musk

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

            # All bias
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
                y, ib_kld = ib_layer.layer_output
                self.kl_total += ib_kld
            return y

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            y = self.X

            self.kl_total = 0.

            layer_index = 0
            # the name of the layer and the coefficient of the kl divergence
            for layer_name in ['conv1_1', 'conv1_2', 'pooling',
                               'conv2_1', 'conv2_2', 'pooling',
                               'conv3_1', 'conv3_2', 'conv3_3', 'pooling',
                               'conv4_1', 'conv4_2', 'conv4_3', 'pooling',
                               'conv5_1', 'conv5_2', 'conv5_3', 'pooling']:
                if layer_index == 0:
                    with tf.variable_scope(layer_name):
                        conv = get_conv(y, regu_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)
                        y = get_ib(y, 'C_ib', 1.)

                elif layer_name != 'pooling':
                    with tf.variable_scope(layer_name):
                        conv = get_conv(y, self.regularizer_conv)
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)
                        y = get_ib(y, 'C_ib', 1.)

                elif layer_name == 'pooling':
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                if layer_name != 'pooling':
                    layer_index += 1

            # From conv to fc layer
            y = tf.contrib.layers.flatten(y)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
                    y = get_ib(y, 'F_ib', 1.)
                layer_index += 1

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
                accuracy_batch = sess.run([self.op_accuracy], feed_dict={self.is_training: False})

                total_correct_preds += accuracy_batch
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        print('\nEpoch:{:d}, val_acc={:%}, used_time:{:.2f}s'.format(epoch + 1,
                                                                     accu,
                                                                     time_end - time_start))
        return accu


# get_mask的np版本
def get_mask(mu, logD, threshold=0):
    # logalpha: [dim]
    logalpha = logD - np.log(np.power(mu, 2) + 1e-8)
    mask = (logalpha < threshold).astype(np.float32)
    return mask


from models.vgg_celeba_512 import VGGNet

if __name__ == '__main__':
    threshold = 0.01
    n_class = 20
    path = '/local/home/david/Remote/models/model_weights/vgg512_ib_celeba1_0.01_0.894047_cr-0.13749'

    weight_dict = pickle.load(open(path, 'rb'))
    dim_list = [64, 64,
                128, 128,
                256, 256, 256,
                512, 512, 512,
                512, 512, 512,
                512, 512, n_class]

    mask_res_list = list()
    for layer_index, layer_name in enumerate(['conv1_1', 'conv1_2',
                                              'conv2_1', 'conv2_2',
                                              'conv3_1', 'conv3_2', 'conv3_3',
                                              'conv4_1', 'conv4_2', 'conv4_3',
                                              'conv5_1', 'conv5_2', 'conv5_3',
                                              'fc6', 'fc7']):
        mu = weight_dict[layer_name + '/info_bottle/mu']
        logD = weight_dict[layer_name + '/info_bottle/logD']

        mask = get_mask(mu, logD, threshold)

        index_list = list()
        for index, value in enumerate(mask):
            if value != 0:
                index_list += [index]
        # 留下来的神经元的序号
        mask_res_list.append(index_list)

    mask_res_list.append(list(np.arange(n_class)))

    # 接下来进行重新inference

    config = process_config("../configs/ib_vgg.json")

    # apply video memory dynamically
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['celeba1']:
        print('Training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training

        session = tf.Session(config=gpu_config)

        training = tf.placeholder(dtype=tf.bool, name='training')

        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.00)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.00)

        # Train
        model = VGGNet(config, task_name, musk=False, gamma=15, model_path=path)
        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        model.get_CR(session)
        model.eval_once(session, model.test_init, -1)

    print('get')
