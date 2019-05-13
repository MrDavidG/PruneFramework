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
    def __init__(self, config, task_name, weight_a, weight_b, cluster_res_list, musk=False, gamma=1.):
        super(VGG_Combined, self).__init__(config)

        self.imgs_path = self.config.dataset_path + 'celeba/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name
        self.is_musked = musk
        self.gamma = gamma

        self.op_logits_a = None
        self.op_logits_b = None

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        self.regularizer_decay = None

        self.weight_dict = self.construct_initial_weights(weight_a, weight_b, cluster_res_list)

        if self.prune_method == 'info_bottle' and 'conv1_1/info_bottle/mu' not in self.weight_dict.keys():
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib())

    def construct_initial_weights(self, weight_dict_a, weight_dict_b, cluster_res_list):
        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        def weight_variable(shape, local=0, scale=1e-2):
            return np.random.normal(local=local, scale=scale, size=shape).astype(dtype=np.float32)

        weight_dict = list()

        dim_list = [64, 64,
                    128, 128,
                    256, 256, 256,
                    512, 512, 512,
                    512, 512, 521,
                    4096, 4096, self.n_classes]

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

            if layer_index == 0:
                weight_dict[layer_name + '/A/weights'] = weight[:, :, :, cluster_res_list[layer_index]['A']]
                weight_dict[layer_name + '/A/biases'] = bias[cluster_res_list[layer_index]['A']]
                weight_dict[layer_name + '/AB/weights'] = weight[:, :, :, cluster_res_list[layer_index]['AB']]
                weight_dict[layer_name + '/AB/biases'] = bias[cluster_res_list[layer_index]['AB']]
                weight_dict[layer_name + '/B/weights'] = weight[:, :, :, cluster_res_list[layer_index]['B']]
                weight_dict[layer_name + '/B/biases'] = bias[cluster_res_list[layer_index]['B']]
            else:
                # Number of neurons in last layer in combined model
                num_A_last_layer = len(cluster_res_list[layer_index - 1]['A'])
                num_B_last_layer = len(cluster_res_list[layer_index - 1]['B'])
                num_AB_from_a_last_layer = (
                        np.array(cluster_res_list[layer_index - 1]['AB']) < dim_list[layer_index - 1]).sum()
                num_AB_from_b_last_layer = len(cluster_res_list[layer_index - 1]['AB']) - num_AB_from_a_last_layer

                # Number of neurons in this layer
                num_A = len(cluster_res_list[layer_index]['A'])
                num_B = len(cluster_res_list[layer_index]['B'])
                num_AB_from_a = (np.array(cluster_res_list[layer_index]['AB']) < dim_list[layer_index]).sum()
                num_AB_from_b = len(cluster_res_list[layer_index]['AB']) - num_AB_from_a_last_layer

                if layer_name.startswith('conv'):
                    # New weights for neurons from A and AB to A
                    weight_dict[layer_name + '/A/weights'] = np.concatenate((weight[:, :, :,
                                                                             cluster_res_list[layer_index]['A']],
                                                                             weight_variable((3, 3,
                                                                                              num_AB_from_b_last_layer,
                                                                                              num_A))), axis=2)

                    # New weights for neurons from last layer to AB
                    # From A to AB
                    weight_dict[layer_name + '/AB/A/weights'] = np.concatenate((weight[:, :,
                                                                                cluster_res_list[layer_index - 1]['A'],
                                                                                cluster_res_list[layer_index]['AB'][
                                                                                :num_AB_from_a]], weight_variable(
                        (3, 3, num_A_last_layer, num_AB_from_b))), axis=-1)

                    # From AB to AB
                    weight_AB_part_a = np.concatenate((weight[:, :, cluster_res_list[layer_index - 1]['AB'][
                                                                    :num_AB_from_a_last_layer],
                                                       cluster_res_list[layer_index]['AB'][:num_AB_from_a]],
                                                       weight_variable(
                                                           (3, 3, num_AB_from_b_last_layer, num_AB_from_a))), axis=2)
                    weight_AB_part_b = np.concatenate((weight_variable((3, 3, num_AB_from_a_last_layer, num_AB_from_b)),
                                                       weight[:, :, cluster_res_list[layer_index - 1]['AB'][
                                                                    num_AB_from_a_last_layer:],
                                                       cluster_res_list[layer_index]['AB'][num_AB_from_a:]]), axis=2)

                    weight_dict[layer_name + '/AB/AB/weights'] = np.concatenate((weight_AB_part_a, weight_AB_part_b),
                                                                                axis=-1)

                    # From B to AB
                    weight_dict[layer_name + '/AB/B/weights'] = np.concatenate((weight_variable(
                        (3, 3, num_B_last_layer, num_AB_from_a)), weight[:, :, cluster_res_list[layer_index - 1]['B'],
                                                                  cluster_res_list[layer_index]['AB'][num_AB_from_a:]]),
                        axis=-1)

                    # New weights for neurons from AB and B to B
                    weight_dict[layer_name + '/B/weights'] = np.concatenate((weight_variable(
                        (3, 3, num_AB_from_a_last_layer, num_B)), weight[:, :, :, cluster_res_list[layer_index]['B']]),
                        axis=2)
                else:
                    # New weights for neurons from A and AB to A
                    weight_dict[layer_name + '/A/weights'] = np.concatenate((weight[:,
                                                                             cluster_res_list[layer_index]['A']],
                                                                             weight_variable(
                                                                                 (num_AB_from_b_last_layer, num_A))),
                                                                            axis=1)

                    # The output layer does not have AB
                    if layer_name != 'fc8':
                        # New weights for neurons from last layer to AB
                        # From A to AB
                        weight_dict[layer_name + '/AB/A/weights'] = np.concatenate((weight[cluster_res_list[
                            layer_index - 1]['A']], cluster_res_list[layer_index]['AB'][:num_AB_from_a],
                                                                                    weight_variable((num_A_last_layer,
                                                                                                     num_AB_from_b))),
                                                                                   axis=-1)

                        # From AB to AB
                        weight_AB_part_a = np.concatenate((weight[cluster_res_list[layer_index - 1]['AB'][
                                                                  :num_AB_from_a_last_layer],
                                                                  cluster_res_list[layer_index]['AB'][:num_AB_from_a]],
                                                           weight_variable((num_AB_from_b_last_layer, num_AB_from_a))),
                                                          axis=0)
                        weight_AB_part_b = np.concatenate((weight_variable((num_AB_from_a_last_layer, num_AB_from_b)),
                                                           weight[cluster_res_list[layer_index - 1]['AB'][
                                                                  num_AB_from_a_last_layer:],
                                                                  cluster_res_list[layer_index]['AB'][num_AB_from_a:]]),
                                                          axis=0)

                        weight_dict[layer_name + '/AB/AB/weights'] = np.concatenate(
                            (weight_AB_part_a, weight_AB_part_b), axis=-1)

                        # From B to AB
                        weight_dict[layer_name + '/AB/B/weights'] = np.concatenate((weight_variable(
                            (num_B_last_layer, num_AB_from_a)), weight[cluster_res_list[layer_index - 1]['B'],
                                                                       cluster_res_list[layer_index]['AB'][
                                                                       num_AB_from_a:]]), axis=-1)

                    # New weights for neurons from AB and B to B
                    weight_dict[layer_name + '/B/weights'] = np.concatenate((weight_variable(
                        (num_AB_from_a_last_layer, num_B)), weight[:, cluster_res_list[layer_index]['B']]), axis=2)

                # Biases
                weight_dict[layer_name + '/A/biases'] = bias[cluster_res_list[layer_index]['A']]
                if layer_name != 'fc8':
                    weight_dict[layer_name + '/AB/biases'] = bias[cluster_res_list[layer_index]['AB']]
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

        def get_conv(x, regu_conv):
            return ConvLayer(x, self.weight_dict, self.config.dropout, self.is_training, self.is_musked,
                             regularizer_conv=regu_conv, is_merge_bn=self.meta_val('is_mrege_bn'))

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total = 0.
            # the name of the layer and the coefficient of the kl divergence
            for layer_index, layer_set in [('conv1_1', 1.0 / 32), ('conv1_2', 1.0 / 32), 'pooling',
                                           ('conv2_1', 1.0 / 16), ('conv2_2', 1.0 / 16), 'pooling',
                                           ('conv3_1', 1.0 / 8), ('conv3_2', 1.0 / 8), ('conv3_3', 1.0 / 8), 'pooling',
                                           ('conv4_1', 1.0 / 4), ('conv4_2', 1.0 / 4), ('conv4_3', 1.0 / 4), 'pooling',
                                           ('conv5_1', 1.0 / 2), ('conv5_2', 1.0 / 2), ('conv5_3', 1.0 / 2), 'pooling']:
                if layer_index == 0:
                    conv_name, kl_mult = layer_set

                    with tf.variable_scope(conv_name + '/A'):
                        conv = get_conv(x, regularizer_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y_A = conv.layer_output

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_A, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_A, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                    with tf.variable_scope(conv_name + '/AB'):
                        conv = get_conv(x, regularizer_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y_AB = conv.layer_output

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_AB, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_AB, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                    with tf.variable_scope(conv_name + '/B'):
                        conv = get_conv(x, regularizer_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y_B = conv.layer_output

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_B, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_B, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                elif layer_set != 'pooling':
                    conv_name, kl_mult = layer_set

                    # A
                    with tf.variable_scope(conv_name + '/A'):
                        # From A and AB to A
                        conv = get_conv(tf.concat((y_A_last, y_AB_last), axis=-1), self.regularizer_conv)
                        self.layers.append(conv)
                        y_A = tf.nn.relu(conv.layer_output)

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_A, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_A, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                    # AB
                    with tf.variable_scope(conv_name + '/AB'):
                        with tf.variable_scope(conv_name + '/A'):
                            # From A to AB
                            conv_A = get_conv(y_A_last, self.regularizer_decay)
                            self.layers.append(conv_A)
                            y_from_A = conv_A.layer_output

                        with tf.variable_scope(conv_name + '/AB'):
                            # From AB to AB
                            conv_AB = get_conv(y_AB_last, self.regularizer_conv)
                            self.layers.append(conv_AB)
                            y_from_AB = conv_AB.layer_output

                        with tf.variable_scope(conv_name + '/B'):
                            # From B to AB
                            conv_B = get_conv(y_B_last, self.regularizer_decay)
                            self.layers.append(conv_B)
                            y_from_B = conv_B.layer_output

                        y_AB = tf.nn.relu(y_from_A + y_from_AB + y_from_B)

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_AB, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_AB, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                    # B
                    with tf.variable_scope(conv_name + '/B'):
                        # From AB and B to B
                        conv = get_conv(tf.concat((y_AB_last, y_B_last), axis=-1), self.regularizer_conv)
                        self.layers.append(conv)
                        y_B = tf.nn.relu(conv.layer_output)

                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y_B, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y_B, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                else:
                    y_A = tf.nn.max_pool(y_A, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    y_AB = tf.nn.max_pool(y_AB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    y_B = tf.nn.max_pool(y_B, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # Record the output of last layer
                y_A_last, y_AB_last, y_B_last = y_A, y_AB, y_B

            # From conv to fc layer
            y_A_last, y_AB_last, y_B_last = [tf.contrib.layers.flatten(layer_output) for layer_output in
                                             [y_A_last, y_AB_last, y_B_last]]

            for fc_name in ['fc6', 'fc7']:
                # A
                with tf.variable_scope(fc_name + '/A'):
                    fc_layer = FullConnectedLayer(tf.concat((y_A_last, y_AB_last), axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y_A = tf.nn.relu(fc_layer.layer_output)

                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y_A, layer_type='F_ib', weight_dict=self.weight_dict,
                                                              is_training=self.is_training,
                                                              kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                        self.layers.append(ib_layer)
                        y_A, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

                # AB
                with tf.variable_scope(fc_name + '/AB'):
                    with tf.variable_scope(fc_name + '/A'):
                        # From A to AB
                        fc_layer_A = FullConnectedLayer(y_A_last, self.weight_dict,
                                                        regularizer_fc=self.regularizer_decay)
                        self.layers.append(fc_layer_A)
                        y_from_A = fc_layer_A.layer_output

                    with tf.variable_scope(fc_name + 'AB'):
                        # From AB to AB
                        fc_layer_AB = FullConnectedLayer(y_AB_last, self.weight_dict,
                                                         regularizer_fc=self.regularizer_fc)
                        self.layers.append(fc_layer_AB)
                        y_from_AB = fc_layer_AB.layer_output

                    with tf.variable_scope(fc_name + '/B'):
                        # From B to AB
                        fc_layer_B = FullConnectedLayer(y_B_last, self.weight_dict,
                                                        regularizer_fc=self.regularizer_decay)
                        self.layers.append(fc_layer_B)
                        y_from_B = fc_layer_B.layer_output

                    y_AB = tf.nn.relu(y_from_A + y_from_AB + y_from_B)

                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y_AB, layer_type='F_ib', weight_dict=self.weight_dict,
                                                              is_training=self.is_training,
                                                              kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                        self.layers.append(ib_layer)
                        y_AB, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

                # B
                with tf.variable_scope(fc_name + '/B'):
                    fc_layer = FullConnectedLayer(tf.concat((y_AB_last, y_B_last), axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    y_B = tf.nn.relu(fc_layer.layer_output)

                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y_B, layer_type='F_ib', weight_dict=self.weight_dict,
                                                              is_training=self.is_training,
                                                              kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                        self.layers.append(ib_layer)
                        y_B, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

                # Record the output of last layer
                y_A_last, y_AB_last, y_B_last = y_A, y_AB, y_B

            with tf.variable_scope('fc8'):
                # A
                with tf.variable_scope(fc_name + '/A'):
                    fc_layer = FullConnectedLayer(tf.concat((y_A_last, y_AB_last), axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y_A = fc_layer.layer_output

                # B
                with tf.variable_scope(fc_name + '/A'):
                    fc_layer = FullConnectedLayer(tf.concat((y_A_last, y_AB_last), axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y_B = fc_layer.layer_output

                self.op_logits = tf.nn.tanh(tf.concat((y_A, y_B)))
                self.op_logits_a = tf.nn.tanh(y_A)
                self.op_logits_b = tf.nn.tanh(y_B)

    def loss(self):
        mae_loss = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
        l2_loss = tf.losses.get_regularization_loss()

        # for the pruning method
        if self.prune_method == 'info_bottle':
            self.op_loss = mae_loss + l2_loss + self.kl_factor * self.kl_total
        else:
            self.op_loss = mae_loss + l2_loss

    def optimize(self):
        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=0.9,
                                                      use_nesterov=True)
                # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)

                self.op_opt = self.opt.minimize(self.op_loss)

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
        self.loss()
        self.optimize()
        self.evaluate()

    def train_one_epoch(self, sess, init, epoch, step):
        sess.run(init)
        total_loss = 0
        total_kl = 0
        total_correct_preds = 0
        n_batches = 0
        time_last = time.time()
        try:
            while True:
                if self.prune_method == 'info_bottle':
                    _, loss, accuracy_batch, kl = sess.run([self.op_opt, self.op_loss, self.op_accuracy, self.kl_total],
                                                           feed_dict={self.is_training: True})
                    total_kl += kl
                else:
                    _, loss, accuracy_batch = sess.run([self.op_opt, self.op_loss, self.op_accuracy],
                                                       feed_dict={self.is_training: True})
                step += 1
                total_loss += loss
                total_correct_preds += accuracy_batch
                n_batches += 1

                if n_batches % 5 == 0:
                    print(
                        '\repoch={:d}, batch={:d}/{:d}, curr_loss={:f}, train_acc={:%}, train_kl={:f}, used_time:{:.2f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.total_batches_train,
                            total_loss / n_batches,
                            total_correct_preds / (n_batches * self.config.batch_size),
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
        n_batches = 0
        time_start = time.time()
        try:
            while True:
                loss_batch, accuracy_batch = sess.run([self.op_loss, self.op_accuracy],
                                                      feed_dict={self.is_training: False})

                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        print('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}, used_time:{:.2f}s'.format(epoch + 1,
                                                                                    accu,
                                                                                    total_loss / n_batches,
                                                                                    time_end - time_start))
        return accu

    def train(self, sess, n_epochs, lr=None):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step)
            accu = self.eval_once(sess, self.test_init, epoch)

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
