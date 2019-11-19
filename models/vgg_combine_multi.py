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


def t2c(task_index):
    if task_index == -1:
        return 'CEN'
    else:
        return chr(ord('A') + task_index)


def c2t(chr):
    if chr == 'CEN':
        return -1
    else:
        return ord(chr) - ord('A')


class VGG_Combined(BaseModel):
    def __init__(self, config, n_tasks, weight_list, cluster_res_list, signal_list):

        super(VGG_Combined, self).__init__(config)

        self.cluster_res_list = cluster_res_list

        self.is_musked = False

        self.n_tasks = n_tasks

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.kl_total_list = None
            self.kl_total = None

        self.op_logits_list = list()

        self.op_opt_list = list()

        self.op_accuracy_list = list()

        self.op_loss_list = list()

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
            self.weight_dict = self.construct_initial_weights(weight_list, cluster_res_list)
            log_t('Initialize weight matrix from weight a and b')
            self.initial_weight = True

        if self.cfg['basic'][
            'pruning_method'] == 'info_bottle' and 'conv1_1/AB/info_bottle/mu' not in self.weight_dict.keys():
            log_t('Initialize ib params')
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib(cluster_res_list))

        self.build()

    def construct_initial_weights(self, weight_list, cluster_res_list):
        def get_signal(layer_index, task_index):
            if task_index == -1:
                return self.signal_list[layer_index]['CEN']
            else:
                key = chr(ord('A') + task_index)
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
                [weight_list[task_index][layer_name + '/biases'] for task_index in range(self.n_tasks)]).astype(
                np.float32)

            # Obtain neuron list
            neuron_index_list = [cluster_res_list[layer_index][t2c(task_index)] for task_index in range(self.n_tasks)]
            CEN = cluster_res_list[layer_index]['CEN']

            def get_neurons(task_index):
                if task_index == -1:
                    return CEN
                else:
                    return neuron_index_list[task_index]

            if layer_index == 0:
                weight = np.concatenate(
                    [weight_list[task_index][layer_name + '/weights'] for task_index in range(self.n_tasks)],
                    axis=-1).astype(np.float32)

                for task_index in range(-1, self.n_tasks):
                    if get_signal(layer_index, task_index):
                        weight_dict['%s/%s/weights' % (layer_name, t2c(task_index))] = weight[:, :, :,
                                                                                       get_neurons(task_index)]
                        weight_dict['%s/%s/biases' % (layer_name, t2c(task_index))] = bias[get_neurons(task_index)]

            else:
                # Get all weights
                in_list, out_list = list(), list()
                for task_index in range(self.n_tasks):
                    n_in, n_out = np.shape(weight_list[task_index][layer_name + '/weights'])[-2:]
                    in_list.append(n_in)
                    out_list.append(n_out)

                if layer_index < 13:
                    weight = np.zeros(
                        shape=[3, 3, np.sum(in_list, dtype=np.int), np.sum(out_list, dtype=np.int)]).astype(np.float32)
                else:
                    weight = np.zeros(
                        shape=[np.sum(in_list, dtype=np.int), np.sum(out_list, dtype=np.int)]).astype(np.float32)

                # 填充权重
                for task_index in range(self.n_tasks):
                    in_start = np.sum(in_list[:task_index], dtype=np.int)
                    out_start = np.sum(out_list[:task_index], dtype=np.int)

                    weight[..., in_start:in_start + in_list[task_index], out_start:out_start + out_list[task_index]] = \
                        weight_list[task_index][layer_name + '/weights']

                # Obtain neuron list of last layer
                neuron_index_list_last = [cluster_res_list[layer_index - 1][t2c(task_index)] for task_index in
                                          range(self.n_tasks)]
                CEN_last = cluster_res_list[layer_index - 1]['CEN']

                if layer_name == 'fc6':
                    neuron_index_list_last = [get_expand(neuron_last) for neuron_last in neuron_index_list_last]
                    CEN_last = get_expand(CEN_last)

                # Weights
                # 分割权重到各个key value对里面去
                for task_index in range(self.n_tasks):
                    if get_signal(layer_index, task_index):
                        neurons = neuron_index_list[task_index]
                        neurons_last = neuron_index_list_last[task_index]

                        weight_dict['%s/%s/weights' % (layer_name, t2c(task_index))] = \
                            weight[..., CEN_last + neurons_last, :][..., neurons]
                        weight_dict['%s/%s/biases' % (layer_name, t2c(task_index))] = bias[neurons]

                weight_dict['%s/%s/weights' % (layer_name, 'CEN')] = weight[..., CEN_last, :][..., CEN]
                weight_dict['%s/%s/biases' % (layer_name, 'CEN')] = bias[CEN]

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
            # The number of neurons
            num_list = [len(cluster_res_list[layer_index][t2c(task_index)]) for task_index in range(self.n_tasks)]
            num_list.append(len(cluster_res_list[layer_index]['CEN']))

            for task_index in range(-1, self.n_tasks):
                num = num_list[task_index]
                if num != 0:
                    weight_dict['%s/%s/info_bottle/mu' % (layer_name, t2c(task_index))] = np.random.normal(loc=1,
                                                                                                           scale=0.01,
                                                                                                           size=[
                                                                                                               num]).astype(
                        np.float32)

                    weight_dict['%s/%s/info_bottle/logD' % (layer_name, t2c(task_index))] = np.random.normal(loc=-9,
                                                                                                             scale=0.01,
                                                                                                             size=[
                                                                                                                 num]).astype(
                        np.float32)
        return weight_dict

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

                if partition_name == 'CEN':
                    for i in range(self.n_tasks + 1):
                        self.kl_total_list[i] += ib_kld
                else:
                    self.kl_total_list[c2t(partition_name)] += ib_kld
            return y

        def create_conv_and_ib(input, conv_name, part_name):

            if get_signal(layer_index, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (conv_name, part_name)):
                    return get_ib(get_conv(tf.concat(input, axis=-1), regu_conv=self.regularizer_conv),
                                  'C_ib', kl_mult, part_name)
            else:
                return None

        def create_fc_and_ib(input, layer_name, part_name):
            if get_signal(layer_index, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (layer_name, part_name)):
                    fc_layer = FullConnectedLayer(tf.concat(input, axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
                    return get_ib(y, 'F_ib', self.cfg['pruning'].getfloat('gamma_fc'), part_name)
            else:
                return None

        def create_output(input, part_name):
            while None in input:
                input.remove(None)
            with tf.variable_scope('fc8/%s' % part_name):
                fc_layer = FullConnectedLayer(tf.concat(input, axis=-1), self.weight_dict,
                                              regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                return fc_layer.layer_output

        self.layers.clear()

        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total_list = [0. for _ in range(self.n_tasks + 1)]
            self.kl_total = 0.

            # 这里跟着每一层的输出结果，若这一层没有A，则y_A=None
            output_last_list = [None for _ in range(self.n_tasks + 1)]

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

                output_list = [None for _ in range(self.n_tasks + 1)]

                if layer_name.startswith('conv'):
                    if layer_index == 0:
                        for task_index in range(self.n_tasks):
                            output_list[task_index] = create_conv_and_ib([x], layer_name, t2c(task_index))
                        # CEN
                        output_list[-1] = create_conv_and_ib([x], layer_name, 'CEN')
                    else:
                        for task_index in range(self.n_tasks):
                            output_list[task_index] = create_conv_and_ib(
                                [output_last_list[task_index], output_last_list[-1]], layer_name, t2c(task_index))
                        # CEN
                        output_list[-1] = create_conv_and_ib([output_last_list[-1]], layer_name, 'CEN')
                else:
                    # Pooling layer
                    for task_index in range(-1, self.n_tasks):
                        if output_last_list[task_index] is not None:
                            output_list[task_index] = tf.nn.max_pool(output_last_list[task_index], ksize=[1, 2, 2, 1],
                                                                     strides=[1, 2, 2, 1], padding='VALID')

                # 记录上一层的结果
                output_last_list = output_list

            # Flatten
            for task_index in range(-1, self.n_tasks):
                if output_last_list[task_index] is not None:
                    output_last_list[task_index] = tf.contrib.layers.flatten(output_last_list[task_index])

            for layer_index, fc_name in zip([13, 14], ['fc6', 'fc7']):
                output_list = [None for _ in range(self.n_tasks + 1)]
                for task_index in range(self.n_tasks):
                    output_list[task_index] = create_fc_and_ib([output_last_list[task_index], output_last_list[-1]],
                                                               fc_name, t2c(task_index))
                output_list[-1] = create_fc_and_ib([output_last_list[-1]], fc_name, 'CEN')

                # 记录上一层的结果
                output_last_list = output_list

            for task_index in range(self.n_tasks):
                output_list[task_index] = create_output([output_last_list[task_index], output_last_list[-1]],
                                                        t2c(task_index))

            self.op_logits_list = [tf.nn.tanh(y) for y in output_list[:-1]]
            self.op_logits = tf.nn.tanh(tf.concat(output_list, axis=1))

    def loss(self):
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / self.n_tasks, tf.int32)
        mae_loss = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)

        self.op_loss_list = [tf.losses.mean_squared_error(
            labels=self.Y[:, dim_label_single * task_index:dim_label_single * (task_index + 1)],
            predictions=self.op_logits_list[task_index]) for task_index in range(self.n_tasks)]

        l2_loss = tf.losses.get_regularization_loss()

        # for the pruning method
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.op_loss = mae_loss + l2_loss + self.kl_factor * self.kl_total

            for task_index in range(self.n_tasks):
                self.op_loss_list[task_index] += l2_loss + self.kl_factor * self.kl_total_list[task_index]
        else:
            self.op_loss = mae_loss + l2_loss

            for task_index in range(self.n_tasks):
                self.op_loss_list[task_index] += l2_loss

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
        dim_label_single = tf.cast(tf.shape(self.Y)[1] / self.n_tasks, tf.int32)
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))

            correct_preds_list = [tf.equal(self.Y[:, dim_label_single * task_index:dim_label_single * (task_index + 1)],
                                           tf.sign(self.op_logits_list[task_index])) for task_index in
                                  range(self.n_tasks)]

            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
                                                                                           tf.float32)

            self.op_accuracy_list = [tf.reduce_sum(tf.cast(correct_preds_list[task_index], tf.float32)) / tf.cast(
                tf.shape(self.Y)[1] / self.n_tasks, tf.float32) for task_index in range(self.n_tasks)]


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

        log_t()

        return remain_params / total_params
