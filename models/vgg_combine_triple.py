# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: vgg_triple
@time: 2019-09-18 14:05

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/")
from models.base_model import BaseModel
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
from layers.ib_layer import InformationBottleneckLayer
from utils.logger import *

import numpy as np
import pickle
import json
import time

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VGG_Combined(BaseModel):
    def __init__(self, config, weight_a, weight_b, weight_c, cluster_res_list, signal_list):

        super(VGG_Combined, self).__init__(config)

        self.cluster_res_list = cluster_res_list

        self.is_musked = False

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.kl_total_a = None
            self.kl_total_b = None
            self.kl_total_c = None

        self.op_logits_a = None
        self.op_logits_b = None
        self.op_logits_c = None

        self.op_opt_a = None
        self.op_opt_b = None
        self.op_opt_c = None

        self.op_accuracy_a = None
        self.op_accuracy_b = None
        self.op_accuracy_c = None

        self.op_loss_a = None
        self.op_loss_b = None
        self.op_loss_c = None

        self.load_dataset()

        self.n_classes = self.Y.shape[1]

        self.signal_list = signal_list

        if self.cfg['path']['path_load'] and os.path.exists(self.cfg['path']['path_load']):
            # Directly load all weights
            self.weight_dict = pickle.load(open(self.cfg['path']['path_load'], 'rb'))
            log_t('Loading weights in %s' % self.cfg['path']['path_load'])
            self.initial_weight = False
        else:
            # Use pre-train weights in conv, but init weights in fc
            self.weight_dict = self.construct_initial_weights(weight_a, weight_b, weight_c, cluster_res_list)
            log_t('Initialize weight matrix from weight a and b')
            self.initial_weight = True

        if self.cfg['basic'][
            'pruning_method'] == 'info_bottle' and 'conv1_1/AB/info_bottle/mu' not in self.weight_dict.keys():
            log_t('Initialize ib params')
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib(cluster_res_list))

        self.build()

    def construct_initial_weights(self, weight_dict_a, weight_dict_b, weight_dict_c, cluster_res_list):
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
            bias = np.concatenate((weight_dict_a[layer_name + '/biases'],
                                   weight_dict_b[layer_name + '/biases'],
                                   weight_dict_c[layer_name + '/biases'])).astype(np.float32)

            # Obtain neuron list
            A = cluster_res_list[layer_index]['A']
            B = cluster_res_list[layer_index]['B']
            C = cluster_res_list[layer_index]['C']
            ABC = cluster_res_list[layer_index]['ABC']

            if layer_index == 0:
                weight = np.concatenate((weight_dict_a[layer_name + '/weights'],
                                         weight_dict_b[layer_name + '/weights'],
                                         weight_dict_c[layer_name + '/weights']), axis=-1).astype(np.float32)
                # The first layer
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/weights'] = weight[:, :, :, A]
                    weight_dict[layer_name + '/A/biases'] = bias[A]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/weights'] = weight[:, :, :, B]
                    weight_dict[layer_name + '/B/biases'] = bias[B]
                if get_signal(layer_index, 'C'):
                    weight_dict[layer_name + '/C/weights'] = weight[:, :, :, C]
                    weight_dict[layer_name + '/C/biases'] = bias[C]
                if get_signal(layer_index, 'ABC'):
                    weight_dict[layer_name + '/ABC/weights'] = weight[:, :, :, ABC]
                    weight_dict[layer_name + '/ABC/biases'] = bias[ABC]
            else:
                # Get all weights
                weight_a = weight_dict_a[layer_name + '/weights']
                weight_b = weight_dict_b[layer_name + '/weights']
                weight_c = weight_dict_c[layer_name + '/weights']

                n_in_a, n_out_a = np.shape(weight_a)[-2:]
                n_in_b, n_out_b = np.shape(weight_b)[-2:]
                n_in_c, n_out_c = np.shape(weight_c)[-2:]

                if layer_index >= 13:
                    weight = np.zeros(shape=[3, 3, n_in_a + n_in_b + n_in_c, n_out_a + n_out_b + n_out_c]).astype(
                        np.float32)
                else:
                    weight = np.zeros(shape=[n_in_a + n_in_b + n_in_c, n_out_a + n_out_b + n_out_c]).astype(np.float32)

                weight[..., :n_in_a, :n_out_a] = weight_a
                weight[..., :n_in_a + n_in_b, n_out_a:n_out_a + n_out_b] = weight_b
                weight[..., -n_in_c:, -n_out_c:] = weight_c

                # Obtain neuron list of last layer
                A_last = cluster_res_list[layer_index - 1]['A']
                B_last = cluster_res_list[layer_index - 1]['B']
                C_last = cluster_res_list[layer_index - 1]['C']
                ABC_last = cluster_res_list[layer_index - 1]['ABC']

                if layer_name == 'fc6':
                    # From conv to fc, times h*w
                    # !!!!!从conv层到全连接层的时候，原来的512->2048,而且其中的index=i, 投影变成了[4*i, 4i+4]...整体是变成了4倍
                    # 最后一层产生的feature map边长的平方
                    A_last = get_expand(A_last)
                    B_last = get_expand(B_last)
                    C_last = get_expand(C_last)
                    ABC_last = get_expand(ABC_last)

                # Weights
                # 要不要排序？？？？？？？？
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/weights'] = weight[..., A_last + ABC_last, :][..., A]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/weights'] = weight[..., ABC_last + B_last, :][..., B]
                if get_signal(layer_index, 'C'):
                    weight_dict[layer_name + '/C/weights'] = weight[..., ABC_last + C_last, :][..., C]
                if get_signal(layer_index, 'ABC'):
                    weight_dict[layer_name + '/ABC/weights'] = weight[..., ABC_last, :][..., ABC]

                # Biases
                if get_signal(layer_index, 'A'):
                    weight_dict[layer_name + '/A/biases'] = bias[A]
                if get_signal(layer_index, 'B'):
                    weight_dict[layer_name + '/B/biases'] = bias[B]
                if get_signal(layer_index, 'C'):
                    weight_dict[layer_name + '/C/biases'] = bias[C]
                if get_signal(layer_index, 'ABC'):
                    weight_dict[layer_name + '/ABC/biases'] = bias[ABC]

        return weight_dict

    def construct_initial_weights_ib(self, cluster_res_list):
        weight_dict = dict()
        # parameters of the information bottleneck
        for layer_index, layer_name in enumerate(['conv1_1', 'conv1_2',
                                                  'conv2_1', 'conv2_2',
                                                  'conv3_1', 'conv3_2', 'conv3_3',
                                                  'conv4_1', 'conv4_2', 'conv4_3',
                                                  'conv5_1', 'conv5_2', 'conv5_3',
                                                  'fc6', 'fc7']):
            # Number of neurons in this layer
            num_A = len(cluster_res_list[layer_index]['A'])
            num_B = len(cluster_res_list[layer_index]['B'])
            num_C = len(cluster_res_list[layer_index]['C'])
            num_ABC = len(cluster_res_list[layer_index]['ABC'])

            for part_name, num in zip(['A', 'B', 'C', 'ABC'], [num_A, num_B, num_C, num_ABC]):
                if num != 0:
                    weight_dict['%s/%s/info_bottle/mu' % (layer_name, part_name)] = np.random.normal(loc=1, scale=0.01,
                                                                                                     size=[num]).astype(
                        np.float32)

                    weight_dict['%s/%s/info_bottle/logD' % (layer_name, part_name)] = np.random.normal(loc=-9,
                                                                                                       scale=0.01,
                                                                                                       size=[
                                                                                                           num]).astype(
                        np.float32)
        return weight_dict

    # def get_conv(self, x, regu_conv):
    #     return ConvLayer(x, self.weight_dict, is_dropout=False, is_training=self.is_training,
    #                      is_musked=self.is_musked, regularizer_conv=regu_conv)

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_signal(layer_index, key):
            return self.signal_list[layer_index][key]

        def get_conv(input, regu_conv):
            conv_layer = ConvLayer(input, self.weight_dict, is_dropout=False, is_training=self.is_training,
                                   is_musked=self.is_musked, regularizer_conv=regu_conv)
            self.layers.append(conv_layer)
            return tf.nn.relu(conv_layer.layer_output)

        def get_ib(y, layer_type, kl_mult, partition_name):
            if layer_type == 'C_ib':
                mask_threshold = self.cfg['pruning'].getfloat('ib_threshold_conv')
            elif layer_type == 'F_ib':
                mask_threshold = self.cfg['pruning'].getfloat('ib_threshold_fc')
            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult,
                                                      mask_threshold=mask_threshold)

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

        def create_conv_and_ib(input, conv_name, part_name):
            if get_signal(layer_index, part_name):
                with tf.variable_scope('%d/%d' % (conv_name, part_name)):
                    return get_ib(get_conv(tf.concat(input.remove(None), axis=-1), regu_conv=self.regularizer_conv),
                                  'C_ib', kl_mult, part_name)
            else:
                return None

        def create_fc_and_ib(input, layer_name, part_name):
            if get_signal(layer_index, part_name):
                with tf.variable_scope('%s/%s' % (layer_name, part_name)):
                    fc_layer = FullConnectedLayer(tf.concat(input.remove(None), axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
                    return get_ib(y, 'F_ib', self.cfg['pruning'].getfloat('gamma_fc'), part_name)
            else:
                return None

        def create_output(input, part_name):
            with tf.variable_scope('fc8'):
                fc_layer = FullConnectedLayer(tf.concat(input.remove(None), axis=-1), self.weight_dict,
                                              regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                return fc_layer.layer_output

        self.layers.clear()

        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total = 0.
            self.kl_total_a = 0.
            self.kl_total_b = 0.
            self.kl_total_c = 0.

            # 这里跟着每一层的输出结果，若这一层没有A，则y_A=None
            y_A_last, y_B_last, y_C_last, y_ABC_last = None, None, None, None

            # The name of the layer and the coefficient of the kl divergence
            for layer_index, layer_name, kl_mult in [(0, 'conv1_1', 1.0 / 32), (1, 'conv1_2', 1.0 / 32),
                                                     (1, 'pooling', 0),
                                                     (2, 'conv2_1', 1.0 / 16), (3, 'conv2_2', 1.0 / 16),
                                                     (3, 'pooling', 0),
                                                     (4, 'conv3_1', 1.0 / 8), (5, 'conv3_2', 1.0 / 8),
                                                     (6, 'conv3_3', 1.0 / 8),
                                                     (6, 'pooling', 0),
                                                     (7, 'conv4_1', 1.0 / 4), (8, 'conv4_2', 1.0 / 4),
                                                     (9, 'conv4_3', 1.0 / 4),
                                                     (9, 'pooling', 0),
                                                     (10, 'conv5_1', 2.0 / 1), (11, 'conv5_2', 2.0 / 1),
                                                     (12, 'conv5_3', 2.0 / 1),
                                                     (12, 'pooling', 0)]:

                y_A, y_B, y_C, y_ABC = None, None, None, None

                if layer_name.startswith('conv'):
                    if layer_index == 0:
                        y_A = create_conv_and_ib([x], layer_name, 'A')
                        y_B = create_conv_and_ib([x], layer_name, 'B')
                        y_C = create_conv_and_ib([x], layer_name, 'C')
                        y_ABC = create_conv_and_ib([x], layer_name, 'ABC')
                    else:
                        y_A = create_conv_and_ib([y_A_last, y_ABC_last], layer_name, 'A')
                        y_B = create_conv_and_ib([y_B_last, y_ABC_last], layer_name, 'B')
                        y_C = create_conv_and_ib([y_C_last, y_ABC_last], layer_name, 'C')
                        y_ABC = create_conv_and_ib([y_ABC_last], 'ABC')
                else:
                    if get_signal(layer_index, 'A'):
                        y_A = tf.nn.max_pool(y_A_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index, 'B'):
                        y_B = tf.nn.max_pool(y_B_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index, 'C'):
                        y_C = tf.nn.max_pool(y_C_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    if get_signal(layer_index, 'ABC'):
                        y_ABC = tf.nn.max_pool(y_ABC_last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 记录上一层的结果
                y_A_last, y_B_last, y_C_last, y_ABC_last = y_A, y_B, y_C, y_ABC

            # Flatten
            if y_A_last is not None:
                y_A_last = tf.contrib.layers.flatten(y_A_last)
            if y_B_last is not None:
                y_B_last = tf.contrib.layers.flatten(y_B_last)
            if y_C_last is not None:
                y_C_last = tf.contrib.layers.flatten(y_C_last)
            if y_ABC_last is not None:
                y_ABC_last = tf.contrib.layers.flatten(y_ABC_last)

            for layer_index, fc_name in zip([14, 15], ['fc6', 'fc7']):
                y_A = create_fc_and_ib([y_A_last, y_ABC_last], fc_name, 'A')
                y_B = create_fc_and_ib([y_B_last, y_ABC_last], fc_name, 'B')
                y_C = create_fc_and_ib([y_C_last, y_ABC_last], fc_name, 'C')
                y_ABC = create_fc_and_ib([y_ABC_last], 'ABC')

                # 记录上一层的结果
                y_A_last, y_B_last, y_C_last, y_ABC_last = y_A, y_B, y_C, y_ABC

            with tf.variable_scope('fc8'):
                y_A = create_output([y_A_last, y_ABC_last], 'A')
                y_B = create_output([y_B_last, y_ABC_last], 'B')
                y_C = create_output([y_C_last, y_ABC_last], 'C')

                self.op_logits = tf.nn.tanh(tf.concat((y_A, y_B, y_C), axis=1))
                self.op_logits_a = tf.nn.tanh(y_A)
                self.op_logits_b = tf.nn.tanh(y_B)
                self.op_logits_c = tf.nn.tanh(y_C)

    def loss(self):
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / 3, tf.int32)
        mae_loss = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
        mae_loss_a = tf.losses.mean_squared_error(labels=self.Y[:, :dim_label_single], predictions=self.op_logits_a)
        mae_loss_b = tf.losses.mean_squared_error(labels=self.Y[:, dim_label_single:2 * dim_label_single],
                                                  predictions=self.op_logits_b)
        mae_loss_c = tf.losses.mean_squared_error(labels=self.Y[:, 2 * dim_label_single:], predictions=self.op_logits_c)

        l2_loss = tf.losses.get_regularization_loss()

        # for the pruning method
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.op_loss = mae_loss + l2_loss + self.kl_factor * self.kl_total
            self.op_loss_a = mae_loss_a + l2_loss + self.kl_factor * self.kl_total_a
            self.op_loss_b = mae_loss_b + l2_loss + self.kl_factor * self.kl_total_b
            self.op_loss_c = mae_loss_c + l2_loss + self.kl_factor * self.kl_total_c
        else:
            self.op_loss = mae_loss + l2_loss
            self.op_loss_a = mae_loss_a + l2_loss
            self.op_loss_b = mae_loss_b + l2_loss
            self.op_loss_c = mae_loss_c + l2_loss

    def optimize(self, lr):
        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # Create a optimizer
                self.opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9,
                                                      use_nesterov=True)

                self.op_opt = self.opt.minimize(self.op_loss)

                self.op_opt_a = self.opt.minimize(self.op_loss_a)

                self.op_opt_b = self.opt.minimize(self.op_loss_b)

    def evaluate(self):
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / 3, tf.int32)
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            correct_preds_a = tf.equal(self.Y[:, :dim_label_single], tf.sign(self.op_logits_a))
            correct_preds_b = tf.equal(self.Y[:, dim_label_single:2 * dim_label_single], tf.sign(self.op_logits_b))
            correct_preds_c = tf.equal(self.Y[:, 2 * dim_label_single:], tf.sign(self.op_logits_b))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
                                                                                           tf.float32)
            self.op_accuracy_a = tf.reduce_sum(tf.cast(correct_preds_a, tf.float32)) / tf.cast(tf.shape(self.Y)[1] / 3,
                                                                                               tf.float32)
            self.op_accuracy_b = tf.reduce_sum(tf.cast(correct_preds_b, tf.float32)) / tf.cast(tf.shape(self.Y)[1] / 3,
                                                                                               tf.float32)
            self.op_accuracy_c = tf.reduct_sum(tf.cast(correct_preds_c, tf.float32)) / tf.cast(tf.shape(self.Y)[1] / 3)

    def set_kl_factor(self, kl_factor):
        log_l('kl_factor: %f' % kl_factor)
        self.kl_factor = kl_factor
        self.loss()
        # self.optimize 在train函数中调用

    def build(self):
        self.inference()
        self.loss()
        self.evaluate()

    def train_one_epoch(self, sess, init, epoch, task_name):
        sess.run(init)
        total_loss = 0
        total_kl = 0
        n_batches = 0

        if task_name == 'A':
            op_opt = self.op_opt_a
        elif task_name == 'B':
            op_opt = self.op_opt_b
        elif task_name == 'C':
            op_opt = self.op_opt_c
        else:
            op_opt = self.op_opt

        time_last = time.time()

        try:
            while True:
                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    _, loss, kl = sess.run([op_opt, self.op_loss, self.kl_total], feed_dict={self.is_training: True})
                    total_kl += kl * self.kl_factor
                else:
                    _, loss = sess.run([op_opt, self.op_loss], feed_dict={self.is_training: True})
                total_loss += loss
                n_batches += 1

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:.4f}, train_kl={:.4f}, used_time:{:.2f}s'.format(
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

        log(str_, need_print=False)

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
        return weight_dict

    def eval_once(self, sess, init, epoch):
        sess.run(init)
        total_loss = 0
        total_correct_preds = 0
        total_correct_preds_a = 0
        total_correct_preds_b = 0
        total_correct_preds_c = 0
        n_batches = 0
        time_start = time.time()
        try:
            while True:
                loss_batch, accuracy_batch, accuracy_batch_a, accuracy_batch_b, accuracy_batch_c = sess.run(
                    [self.op_loss, self.op_accuracy, self.op_accuracy_a, self.op_accuracy_b, self.op_accuracy_c],
                    feed_dict={self.is_training: False})

                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                total_correct_preds_a += accuracy_batch_a
                total_correct_preds_b += accuracy_batch_b
                total_correct_preds_c += accuracy_batch_c
                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()
        accu = total_correct_preds / self.n_samples_val
        accu_a = total_correct_preds_a / self.n_samples_val
        accu_b = total_correct_preds_b / self.n_samples_val
        accu_c = total_correct_preds_c / self.n_samples_val

        str_ = 'Epoch:{:d}, val_acc={:%} | a={:%} | b={:%} | c={:%}, val_loss={:f}, used_time:{:.2f}s'.format(
            epoch + 1, accu, accu_a, accu_b, accu_c, total_loss / n_batches, time_end - time_start)

        # 写文件
        log('\n' + str_)

        return accu, accu_a, accu_b, accu_c

    def train_one_epoch_individual(self, sess, init, epoch):
        sess.run(init)
        total_loss_a = 0
        total_loss_b = 0
        total_kl_a = 0
        total_kl_b = 0
        n_batches = 0

        time_last = time.time()

        try:
            while True:
                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    _, loss_a, kl_a = sess.run([self.op_opt_a, self.op_loss_a, self.kl_total_a],
                                               feed_dict={self.is_training: True})
                    total_kl_a += kl_a
                    _, loss_b, kl_b = sess.run([self.op_opt_b, self.op_loss_b, self.kl_total_b],
                                               feed_dict={self.is_training: True})
                    total_kl_b += kl_b

                else:
                    _, loss_a = sess.run([self.op_opt_a, self.op_loss_a], feed_dict={self.is_training: True})
                    _, loss_b = sess.run([self.op_opt_b, self.op_loss_b], feed_dict={self.is_training: True})
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
        log(str_, need_print=False)

    def train_individual(self, sess, n_epochs, lr):
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))
        for epoch in range(n_epochs):
            self.train_one_epoch_individual(sess, self.train_init, epoch)
            acc, acc_a, acc_b, acc_c = self.eval_once(sess, self.test_init, epoch)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                cr = self.get_CR(sess, self.cluster_res_list)

            if (epoch + 1) % self.cfg['train'].getint('save_step') == 0:
                if self.cfg['basic']['pruning_method'] != 'info_bottle':
                    name = '%s/tr%.2d-epo%.3d-acc%.4f-%.4f-%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc, acc_a, acc_b)
                else:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-acc%.4f-%.4f-%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, acc, acc_a, acc_b)
                self.save_weight(sess, name)

        # Count of training
        self.cnt_train += 1
        # Save into cfg
        name_train = 'train%d' % self.cnt_train
        self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'function', 'train_individual')
        self.cfg.set(name_train, 'n_epochs', str(n_epochs))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))
        self.cfg.set(name_train, 'acc_a', str(acc_a))
        self.cfg.set(name_train, 'acc_b', str(acc_b))
        self.cfg.set(name_train, 'acc_c', str(acc_c))
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.cfg.set(name_train, 'cr', str(cr))

    def train(self, sess, n_epochs, task_name, lr):
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        for epoch in range(n_epochs):
            self.train_one_epoch(sess, self.train_init, epoch, task_name)
            acc, acc_a, acc_b, acc_c = self.eval_once(sess, self.test_init, epoch)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                cr = self.get_CR(sess, self.cluster_res_list)

            if (epoch + 1) % self.cfg['train'].getint('save_step') == 0:
                if self.cfg['basic']['pruning_method'] != 'info_bottle':
                    name = '%s/tr%.2d-epo%.3d-acc%.4f-%.4f-%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc, acc_a, acc_b)
                else:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-acc%.4f-%.4f-%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, acc, acc_a, acc_b)
                self.save_weight(sess, name)

        # Count of training
        self.cnt_train += 1
        # Save into cfg
        name_train = 'train%d' % self.cnt_train
        self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'n_epochs', str(n_epochs))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))
        self.cfg.set(name_train, 'acc_a', str(acc_a))
        self.cfg.set(name_train, 'acc_b', str(acc_b))
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.cfg.set(name_train, 'cr', str(cr))

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

    def get_CR(self, sess, cluster_res_list):
        layers_name = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']

        # name_len_dict = {'conv1_1': 72, 'conv1_2': 72,
        #                  'conv2_1': 36, 'conv2_2': 36,
        #                  'conv3_1': 18, 'conv3_2': 18, 'conv3_3': 18,
        #                  'conv4_1': 9, 'conv4_2': 9, 'conv4_3': 9,
        #                  'conv5_1': 4, 'conv5_2': 4, 'conv5_3': 4}

        # 根据cluster_res_list获得剪枝前的情况
        net_origin = list()
        for cluster_layer in cluster_res_list:
            layer_set = dict()
            for key in ['A', 'B', 'C', 'ABC']:
                layer_set[key] = len(cluster_layer[key])
            net_origin.append(layer_set)

        net_pruned = list()
        for layer_name in layers_name:
            layer_masks_tf = [tf.sum(self.get_layer_id(layer_name + '/' + part_name)) for part_name in
                              ['A', 'B', 'C', 'ABC']]
            layer_masks = sess.run(layer_masks_tf)

            layer_set = dict()
            layer_set['A'], layer_set['B'], layer_set['C'], layer_set['ABC'] = [np.sum(mask) for mask in layer_masks]

            # 按照layer加入进去
            net_pruned.append(layer_set)

        def count_params(net):
            count_params = 0

            for layer_index, dict_ in enumerate(net):
                n_A, n_B, n_C, n_ABC = dict_['A'], dict_['B'], dict_['C'], dict_['ABC']
                if layer_index == 0:
                    count_params += 9 * 3 * (n_A + n_B + n_C + n_ABC)
                elif layer_index < 13:
                    count_params += 9 * (n_A_last * n_A + n_B_last * n_B + n_C_last * n_C + n_ABC_last * (
                            n_A + n_B + n_C + n_ABC))
                elif 13 <= layer_index and layer_index < 15:
                    count_params += (n_A_last * n_A + n_B_last * n_B + n_C_last * n_C + n_ABC_last * (
                            n_A + n_B + n_C + n_ABC))
                elif layer_index == 15:
                    count_params += (n_A_last * n_A + n_B_last * n_B + n_C_last * n_C)
                n_A_last, n_B_last, n_C_last, n_ABC_last = n_A, n_B, n_C, n_ABC

                # flatten
                if layer_index == 12:
                    n_A_last *= 4
                    n_B_last *= 4
                    n_C_last *= 4
                    n_ABC_last *= 4

        total_params = count_params(net_origin)
        remain_params = count_params(net_pruned)

        # 都是做mask之前的入度和出度
        # num_out_channel_dict = dict()
        # num_in_channel_dict = dict()
        #
        # for layer_index, cluster_layer in enumerate(cluster_res_list):
        #     layer_name = layers_name[layer_index]
        #     num_A = len(cluster_layer['A'])
        #     num_out_channel_dict[layer_name + '/A'] = num_A
        #
        #     num_AB = len(cluster_layer['AB'])
        #     num_out_channel_dict[layer_name + '/AB'] = num_AB
        #
        #     num_B = len(cluster_layer['B'])
        #     num_out_channel_dict[layer_name + '/B'] = num_B
        #
        #     if layer_index == 0:
        #         num_in_channel_dict[layer_name + '/A'] = 3
        #         num_in_channel_dict[layer_name + '/AB'] = 3
        #         num_in_channel_dict[layer_name + '/B'] = 3
        #     else:
        #         num_A_last = len(cluster_res_list[layer_index - 1]['A'])
        #         num_AB_last = len(cluster_res_list[layer_index - 1]['AB'])
        #         num_B_last = len(cluster_res_list[layer_index - 1]['B'])
        #
        #         num_in_channel_dict[layer_name + '/A'] = num_A_last + num_AB_last
        #         num_in_channel_dict[layer_name + '/AB'] = num_AB_last
        #         num_in_channel_dict[layer_name + '/B'] = num_AB_last + num_B_last
        #
        # # 输出被prune的数量
        # masks = list()
        # layers_type = list()
        # layers_name_list = list()
        # for layer in self.layers:
        #     if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib':
        #         # 和musks是一一对应的关系
        #         layers_name_list += [layer.layer_name]
        #         layers_type += [layer.layer_type]
        #
        #         if layer.layer_type == 'C_ib':
        #             masks += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('ib_threshold_conv'))]
        #         elif layer.layer_type == 'F_ib':
        #             masks += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('ib_threshold_fc'))]
        #
        # # 获得具体的mask
        # masks = sess.run(masks)
        #
        # # how many channels/dims are prune in each layer
        # prune_state = [np.sum(mask == 0) for mask in masks]
        # original_state = [len(mask) for mask in masks]
        #
        # # 记录一下每一层的出度被剪枝了多少
        # out_prune_dict = dict()
        # for i, layer_name in enumerate(layers_name_list):
        #     # 这一层被剪枝了多少
        #     out_prune_dict[layer_name] = prune_state[i]
        #
        # # 记录这一层被剪枝了多少个神经元
        # in_prune_dict = dict()
        # for i, layer_name in enumerate(layers_name):
        #     if i == 0:
        #         in_prune_dict[layer_name + '/A'] = 0
        #         in_prune_dict[layer_name + '/AB'] = 0
        #         in_prune_dict[layer_name + '/B'] = 0
        #         continue
        #
        #     layer_name_last = layers_name[i - 1]
        #
        #     # 输入被剪枝掉了多少!!!!
        #     in_prune_dict[layer_name + '/A'] = out_prune_dict.get(layer_name_last + '/A', 0) + out_prune_dict.get(
        #         layer_name_last + '/AB', 0)
        #     if layer_name != 'fc8':
        #         in_prune_dict[layer_name + '/AB'] = out_prune_dict.get(layer_name_last + '/AB', 0)
        #     in_prune_dict[layer_name + '/B'] = out_prune_dict.get(layer_name_last + '/AB', 0) + out_prune_dict.get(
        #         layer_name_last + '/B', 0)
        #
        # total_params, pruned_params, remain_params = 0, 0, 0
        # total_flops, remain_flops, pruned_flops = 0, 0, 0
        #
        # for index, layer_name in enumerate(layers_name_list):
        #     num_in = num_in_channel_dict[layer_name]
        #     num_out = num_out_channel_dict[layer_name]
        #
        #     num_out_prune = out_prune_dict.get(layer_name)
        #     num_in_prune = in_prune_dict.get(layer_name)
        #
        #     if layers_type[index] == 'C_ib':
        #         total_params += num_in * num_out * 9
        #         remain_params += (num_in - num_in_prune) * (num_out - num_out_prune) * 9
        #
        #         # FLOPs
        #         for key in name_len_dict.keys():
        #             if layer_name.startswith(key):
        #                 M = name_len_dict[key]
        #                 break
        #         total_flops += 2 * (9 * num_in + 1) * M * M * num_out
        #         remain_flops += 2 * (9 * (num_in - num_in_prune) + 1) * M * M * (num_out - num_out_prune)
        #
        #     elif layers_type[index] == 'F_ib':
        #         if 'fc6' in layer_name:
        #             total_params += 4 * num_in * num_out
        #             remain_params += 4 * (num_in - num_in_prune) * (num_out - num_out_prune)
        #
        #             # FLOPs
        #             total_flops += (2 * num_in * 4 - 1) * num_out
        #             remain_flops += (2 * (num_in - num_in_prune) * 4 - 1) * (num_out - num_out_prune)
        #         else:
        #             total_params += num_in * num_out
        #             remain_params += (num_in - num_in_prune) * (num_out - num_out_prune)
        #
        #             # FLOPs
        #             total_flops += (2 * num_in - 1) * num_out
        #             remain_flops += (2 * (num_in - num_in_prune) - 1) * (num_out - num_out_prune)
        #
        # n_classes = self.cfg['data'].getint('n_classes') / 3
        #
        # # output layer
        # total_params += num_in_channel_dict['fc8/A'] * n_classes
        # remain_params += (num_in_channel_dict['fc8/A'] - in_prune_dict['fc8/A']) * n_classes
        # total_params += num_in_channel_dict['fc8/B'] * n_classes
        # remain_params += (num_in_channel_dict['fc8/B'] - in_prune_dict['fc8/B']) * n_classes
        #
        # # FLOPs
        # total_flops += (2 * num_in_channel_dict['fc8/A'] - 1) * n_classes
        # remain_flops += (2 * (num_in_channel_dict['fc8/A'] - in_prune_dict['fc8/A']) - 1) * n_classes
        # total_flops += (2 * num_in_channel_dict['fc8/B'] - 1) * n_classes
        # remain_flops += (2 * (num_in_channel_dict['fc8/B'] - in_prune_dict['fc8/B']) - 1) * n_classes
        #
        # pruned_params = total_params - remain_params
        # pruned_flops = total_flops - remain_flops
        #
        # cr = np.around(float(total_params - pruned_params) / total_params, decimals=5)
        #
        # str_1 = 'Total params: {}, Pruned params: {}, Remaining params:{}, Remain/Total params:{}'.format(
        #     total_params, pruned_params, remain_params, cr)
        #
        # res_each_layer = list()
        # for i in range(len(prune_state)):
        #     res_each_layer += [str(prune_state[i]) + '/' + str(original_state[i])]
        #
        # str_2 = 'Each layer pruned: {}'.format(res_each_layer).replace("'", "")
        #
        # str_3 = 'Total FLOPs: {}, Pruned FLOPs: {}, Remaining FLOPs: {}, Remain/Total FLOPs:{}'.format(total_flops,
        #                                                                                                pruned_flops,
        #                                                                                                remain_flops,
        #                                                                                                np.around(
        #                                                                                                    float(
        #                                                                                                        total_flops - pruned_flops) / total_flops,
        #                                                                                                    decimals=5))
        #
        # log(str_1 + '\n' + str_2 + '\n' + str_3)
        return cr
