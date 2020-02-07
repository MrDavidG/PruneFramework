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

from models.base_model import BaseModel
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
from layers.ib_layer import InformationBottleneckLayer
from utils.logger import *
from utils.json import read_i
from utils.json import read_l
from utils.json import read_s
from utils.json import read_f

import numpy as np
import pickle
import json
import time

import os
import tensorflow as tf


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


class Model_Combined(BaseModel):
    def __init__(self, config, n_tasks, weight_list, cluster_res_dict, signal_dict):

        super(Model_Combined, self).__init__(config)

        self.is_musked = False

        self.n_tasks = n_tasks
        self.signal_dict = signal_dict
        self.cluster_res_dict = self.pro(cluster_res_dict)

        if self.pruning:
            self.kl_total_list = None
            self.kl_total = None

        self.op_logits_list = None
        self.op_opt_list = None
        self.op_acc_list = None

        self.op_loss_func_list = None
        self.op_loss_kl_list = None

        self.load_dataset()

        self.n_classes = self.Y.shape[1]

        if read_s(self.cfg, 'path', 'path_load') and os.path.exists(read_s(self.cfg, 'path', 'path_load')):
            # Directly load all weights
            log_t('Loading weights in %s' % read_s(self.cfg, 'path', 'path_load'))
            self.weight_dict = pickle.load(open(read_s(self.cfg, 'path', 'path_load'), 'rb'))
            self.initial_weight = False
        else:
            # Use pre-train weights in conv, but init weights in fc
            log_t('Initialize weight matrix from weight lists')
            self.weight_dict = self.init_weights(weight_list)
            self.initial_weight = True

        if self.pruning and self.structure[0] + '_vib/CEN/mu' not in self.weight_dict.keys():
            log_t('Initialize VIBNet params')
            self.weight_dict = dict(self.weight_dict, **self.init_weights_vib())

        self.init_labels()
        self.build()

    def pro(self, cluster_res_dict):
        """
        把输出label的序号改成对应读取的self.Y中的序号，默认按照task的顺序进行排序
        :param cluster_res_dict:
        :return:
        """
        name = self.structure[-1]
        dict_output = dict()
        index_s = 0
        for task_index in range(self.n_tasks):
            dict_output[t2c(task_index)] = np.arange(index_s,
                                                     index_s + len(cluster_res_dict[name][t2c(task_index)])).tolist()
            index_s += len(cluster_res_dict[name][t2c(task_index)])
        dict_output['CEN'] = cluster_res_dict[name]['CEN']
        cluster_res_dict[name] = dict_output
        return cluster_res_dict

    def init_labels(self):
        self.Y_list = list()
        self.n_classes_list = list()

        index_s = 0
        # 先替换成$，然后在进行分割
        str_labels = read_s(self.cfg, 'task', 'labels_task').strip().replace('],[', ']$[')
        for item in str_labels[1:-1].split('$'):
            if item.count('-') == 0:
                labels = json.loads(item)
            else:
                s, e = [int(_) for _ in item[1:-1].split('-')]
                labels = np.arange(s, e).tolist()

            # self.Y_list里面保存的是所有label里面的相对顺序位置
            # 所以要求labels里的顺序是labels_task里面去掉方括号的效果
            # labels_task仅仅是用来区分每个task的，实际只提供每个task label的长度值
            self.Y_list.append(np.arange(index_s, index_s + len(labels)).tolist())
            self.n_classes_list.append(len(labels))
            index_s = index_s + len(labels)

    def init_weights(self, weight_list):
        def get_signal(layer_name, task_index):
            return self.signal_dict[layer_name][t2c(task_index)]

        def get_expand(array, h, w, original_channel_num_a):
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
        h, w = read_l(self.cfg, 'data', 'length')

        for ind, layer_name in enumerate(self.structure):
            if layer_name == 'p':
                h, w = h // 2, w // 2
            elif layer_name == 'fla':
                neuron_index_list_last = [get_expand(neuron_last, h, w, read_i(self.cfg, 'model', 'filter_last'))
                                          for neuron_last in neuron_index_list_last]
                CEN_last = get_expand(CEN_last, h, w, read_i(self.cfg, 'model', 'filter_last'))
            else:
                # All bias
                bias = np.concatenate(
                    [weight_list[task_index][layer_name + '/b'] for task_index in range(self.n_tasks)]).astype(
                    np.float32)

                # Obtain neuron list
                neuron_index_list = [self.cluster_res_dict[layer_name][t2c(task_index)] for task_index in
                                     range(self.n_tasks)]
                CEN = self.cluster_res_dict[layer_name]['CEN']

                def get_neurons(task_index):
                    if task_index == -1:
                        return CEN
                    else:
                        return neuron_index_list[task_index]

                if ind == 0:
                    weight = np.concatenate(
                        [weight_list[task_index][layer_name + '/w'] for task_index in range(self.n_tasks)],
                        axis=-1).astype(np.float32)

                    for task_index in range(-1, self.n_tasks):
                        if get_signal(layer_name, task_index):
                            weight_dict['%s/%s/w' % (layer_name, t2c(task_index))] = weight[:, :, :,
                                                                                     get_neurons(task_index)]
                            weight_dict['%s/%s/b' % (layer_name, t2c(task_index))] = bias[get_neurons(task_index)]
                else:
                    # Get all weights
                    in_list, out_list = list(), list()
                    for task_index in range(self.n_tasks):
                        n_in, n_out = np.shape(weight_list[task_index][layer_name + '/w'])[-2:]
                        in_list.append(n_in)
                        out_list.append(n_out)

                    if layer_name.startswith('c'):
                        weight = np.zeros(
                            shape=[self.kernel_size[0], self.kernel_size[1], np.sum(in_list, dtype=np.int),
                                   np.sum(out_list, dtype=np.int)]).astype(np.float32)
                    elif layer_name.startswith('f'):
                        weight = np.zeros(shape=[np.sum(in_list, dtype=np.int), np.sum(out_list, dtype=np.int)]).astype(
                            np.float32)

                    # 填充权重
                    for task_index in range(self.n_tasks):
                        in_start = np.sum(in_list[:task_index], dtype=np.int)
                        out_start = np.sum(out_list[:task_index], dtype=np.int)

                        weight[..., in_start:in_start + in_list[task_index],
                        out_start:out_start + out_list[task_index]] = weight_list[task_index][layer_name + '/w']

                    # 分割权重到各个key value对里面去
                    for task_index in range(self.n_tasks):
                        if get_signal(layer_name, task_index):
                            neurons = neuron_index_list[task_index]
                            neurons_last = neuron_index_list_last[task_index]

                            weight_dict['%s/%s/w' % (layer_name, t2c(task_index))] = \
                                weight[..., CEN_last + neurons_last, :][..., neurons]
                            weight_dict['%s/%s/b' % (layer_name, t2c(task_index))] = bias[neurons]

                    weight_dict['%s/%s/w' % (layer_name, 'CEN')] = weight[..., CEN_last, :][..., CEN]
                    weight_dict['%s/%s/b' % (layer_name, 'CEN')] = bias[CEN]

                neuron_index_list_last = neuron_index_list
                CEN_last = CEN

        return weight_dict

    def init_weights_vib(self):
        weight_dict = dict()
        for layer_index, layer_name in enumerate([_ for _ in self.structure if _ not in ['p', 'fla']]):
            # The number of neurons
            num_list = [len(self.cluster_res_dict[layer_name][t2c(task_index)]) for task_index in range(self.n_tasks)]
            num_list.append(len(self.cluster_res_dict[layer_name]['CEN']))

            for task_index in range(-1, self.n_tasks):
                num = num_list[task_index]
                if num != 0:
                    weight_dict['%s_vib/%s/mu' % (layer_name, t2c(task_index))] = np.random.normal(loc=1,
                                                                                                   scale=0.01,
                                                                                                   size=[
                                                                                                       num]).astype(
                        np.float32)

                    weight_dict['%s_vib/%s/logD' % (layer_name, t2c(task_index))] = np.random.normal(loc=-9,
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

        def get_signal(layer_name, key):
            return self.signal_dict[layer_name][key]

        def get_ib(y, layer_type, kl_mult, partition_name):
            if self.pruning:
                if layer_type == 'C_vib':
                    name = 'conv'
                elif layer_type == 'F_vib':
                    name = 'fc'

                    # if layer_name == 'f7':
                    #     if partition_name == 'CEN':
                    #         kl_mult = kl_mult * 1.1
                    #     else:
                    #         kl_mult = kl_mult * 0.7

                mask_threshold = read_f(self.cfg, 'pruning', 'ib_threshold_' + name)
                gamma = read_f(self.cfg, 'pruning', 'gamma_' + name)

                ib_layer = InformationBottleneckLayer(y, layer_type=layer_type, weight_dict=self.weight_dict,
                                                      is_training=self.is_training, kl_mult=kl_mult * gamma,
                                                      mask_threshold=mask_threshold)

                self.layers.append(ib_layer)
                y, ib_kld = ib_layer.layer_output
                self.kl_total += ib_kld

                if partition_name == 'CEN':
                    for i in range(self.n_tasks):
                        self.kl_total_list[i] += ib_kld
                else:
                    self.kl_total_list[c2t(partition_name)] += ib_kld
            return y

        def create_conv_and_ib(input, layer_name, part_name, kl_mult):
            if get_signal(layer_name, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (layer_name, part_name)):
                    conv_layer = ConvLayer(tf.concat(input, axis=-1), self.weight_dict, is_dropout=False,
                                           is_training=self.is_training, is_musked=self.is_musked,
                                           regularizer_conv=self.regularizer_conv)
                    self.layers.append(conv_layer)
                    y = tf.nn.relu(conv_layer.layer_output)
                with tf.variable_scope('%s_vib/%s' % (layer_name, part_name)):
                    return get_ib(y, 'C_vib', kl_mult, part_name)
            else:
                return None

        def create_fc_and_ib(input, layer_name, part_name, kl_mult):
            if get_signal(layer_name, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (layer_name, part_name)):
                    fc_layer = FullConnectedLayer(tf.concat(input, axis=-1), self.weight_dict,
                                                  regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
                with tf.variable_scope('%s_vib/%s' % (layer_name, part_name)):
                    return get_ib(y, 'F_vib', kl_mult, part_name)
            else:
                return None

        def create_pool(input):
            if input is None:
                return None
            else:
                return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        def create_flatten(input):
            if input is None:
                return None
            else:
                return tf.contrib.layers.flatten(input)

        def create_output(input, layer_name, part_name):
            while None in input:
                input.remove(None)
            with tf.variable_scope('%s/%s' % (layer_name, part_name)):
                fc_layer = FullConnectedLayer(tf.concat(input, axis=-1), self.weight_dict,
                                              regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                return fc_layer.layer_output

        self.layers.clear()

        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            x = self.X

            self.kl_total_list = [0. for _ in range(self.n_tasks)]
            self.kl_total = 0.

            # 这一层没有A，则y_A=None
            output_last_list = [None for _ in range(self.n_tasks + 1)]

            for layer_index, (layer_name, kl_mult) in enumerate(
                    zip(self.structure, read_l(self.cfg, 'pruning', 'kl_mult') + [0])):
                if layer_name.startswith('c'):
                    if layer_index == 0:
                        output_list = [create_conv_and_ib([x], layer_name, t2c(task_index), kl_mult) for task_index in
                                       range(self.n_tasks)]
                        # CEN
                        output_list.append(create_conv_and_ib([x], layer_name, 'CEN', kl_mult))
                    else:
                        output_list = [
                            create_conv_and_ib([output_last_list[task_index], output_last_list[-1]], layer_name,
                                               t2c(task_index), kl_mult) for task_index in range(self.n_tasks)]
                        # CEN
                        output_list.append(create_conv_and_ib([output_last_list[-1]], layer_name, 'CEN', kl_mult))
                elif layer_name.startswith('f') and layer_name != 'fla':
                    if layer_index != len(self.structure) - 1:
                        output_list = [
                            create_fc_and_ib([output_last_list[task_index], output_last_list[-1]], layer_name,
                                             t2c(task_index), kl_mult) for task_index in range(self.n_tasks)]
                        # CEN
                        output_list.append(create_fc_and_ib([output_last_list[-1]], layer_name, 'CEN', kl_mult))
                    else:
                        output_list = [create_output([output_last_list[task_index], output_last_list[-1]], layer_name,
                                                     t2c(task_index)) for task_index in range(self.n_tasks)]
                elif layer_name == 'p':
                    output_list = [create_pool(output_last) for output_last in output_last_list]
                elif layer_name == 'fla':
                    output_list = [create_flatten(output_last) for output_last in output_last_list]

                output_last_list = output_list

            self.op_logits_list = [tf.nn.tanh(y) for y in output_list]
            self.op_logits = tf.nn.tanh(tf.concat(output_list, axis=1))

    def loss(self):
        # normal func loss
        self.op_loss_func = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
        self.op_loss_func_list = [tf.losses.mean_squared_error(labels=tf.gather(self.Y, self.Y_list[ind], axis=-1),
                                                               predictions=self.op_logits_list[ind]) for ind in
                                  range(self.n_tasks)]

        # 这里是为了scenario2做测试来用的
        # self.op_loss_func_list = [tf.losses.mean_squared_error(labels=tf.gather(self.Y, self.Y_list[ind], axis=-1),
        #                                                        predictions=self.op_logits_list[ind]) for ind in
        #                           range(self.n_tasks)]
        # self.op_loss_func = 0.05 * self.op_loss_func_list[0]+self.op_loss_func_list[1]

        # Notice: l2 loss没有分任务，建议不要使用
        self.op_loss_regu = tf.losses.get_regularization_loss()

        if self.pruning:
            self.op_loss_kl = self.kl_factor * self.kl_total
            self.op_loss_kl_list = [self.kl_factor * self.kl_total_list[ind] for ind in range(self.n_tasks)]

            self.op_loss = self.op_loss_func + self.op_loss_regu + self.op_loss_kl
            self.op_loss_list = [self.op_loss_func_list[ind] + self.op_loss_regu + self.op_loss_kl_list[ind] for ind in
                                 range(self.n_tasks)]
        else:
            self.op_loss = self.op_loss_func + self.op_loss_regu
            self.op_loss_list = [self.op_loss_func_list[ind] + self.op_loss_regu for ind in range(self.n_tasks)]

    def optimize(self, lr):
        def get_opt(type):
            if type == 'momentum':
                return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
            elif type == 'adam':
                return tf.train.AdamOptimizer(learning_rate=lr)
            elif type == 'sgd':
                return tf.train.GradientDescentOptimizer(learning_rate=lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.opt = get_opt(self.cfg['task']['optimizer'])
                self.op_opt = self.opt.minimize(self.op_loss)
                self.op_opt_list = [self.opt.minimize(op_loss) for op_loss in self.op_loss_list]

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.reduce_sum(tf.cast(tf.equal(self.Y, tf.sign(self.op_logits)), tf.float32))
            correct_preds_list = [tf.reduce_sum(
                tf.cast(tf.equal(tf.gather(self.Y, self.Y_list[ind], axis=-1), tf.sign(self.op_logits_list[ind])),
                        tf.float32)) for ind
                in range(self.n_tasks)]

            self.op_acc = correct_preds / tf.cast(tf.shape(self.Y)[1], tf.float32)
            self.op_acc_list = [correct_preds_list[ind] / len(self.Y_list[ind]) for ind in range(self.n_tasks)]

    def set_kl_factor(self, kl_factor):
        log_l('kl_factor: %f' % kl_factor)
        self.kl_factor = kl_factor
        self.loss()

    def build(self):
        self.inference()
        self.loss()
        self.evaluate()

    def train_one_epoch(self, sess, init, epoch):
        sess.run(init)

        avg_loss = 0
        avg_loss_func = 0
        avg_loss_kl = 0
        n_batches = 0

        time_last = time.time()
        try:
            while True:
                if self.pruning:
                    _, loss, loss_func, loss_kl = sess.run(
                        [self.op_opt, self.op_loss, self.op_loss_func, self.op_loss_kl],
                        feed_dict={self.is_training: True})

                    avg_loss_func += (loss_func - avg_loss_func) / (n_batches + 1.)
                    avg_loss_kl += (loss_kl - avg_loss_kl) / (n_batches + 1.)
                else:
                    _, loss = sess.run([self.op_opt, self.op_loss], feed_dict={self.is_training: True})

                avg_loss += (loss - avg_loss) / (n_batches + 1.)
                n_batches += 1

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:.4f}, curr_loss_func={:.4f}, curr_loss_kl={:.4f}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        avg_loss,
                        avg_loss_func,
                        avg_loss_kl,
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

        avg_loss = 0
        avg_loss_func = 0
        avg_loss_kl = 0

        corr_preds = 0
        corr_preds_list = [0 for _ in range(self.n_tasks)]

        n_batches = 0
        time_start = time.time()
        try:
            while True:
                if self.pruning:
                    loss, loss_func, loss_kl, acc, acc_list = sess.run(
                        [self.op_loss, self.op_loss_func, self.op_loss_kl, self.op_acc] + [self.op_acc_list],
                        {self.is_training: False})

                    avg_loss_kl += (loss_kl - avg_loss_kl) / (n_batches + 1.)
                    avg_loss_func += (loss_func - avg_loss_func) / (n_batches + 1.)
                else:
                    loss, acc, acc_list = sess.run([self.op_loss, self.op_acc] + [self.op_acc_list],
                                                   {self.is_training: False})

                avg_loss += (loss - avg_loss) / (n_batches + 1.)

                corr_preds += acc
                corr_preds_list = [corr_preds_list[ind] + acc_list[ind] for ind in range(self.n_tasks)]

                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass

        time_end = time.time()

        accu = corr_preds / self.n_samples_val
        accu_list = [corr_preds_list[ind] / self.n_samples_val for ind in range(self.n_tasks)]

        if self.pruning:
            str_ = 'Epoch:{:d}, val_acc={:.3%}|{:}, val_loss={:.4f}, val_loss_func={:.4f}, val_loss_kl={:.4f}, used_time:{:.2f}s'.format(
                epoch + 1, accu, str(['{:.3%}'.format(_) for _ in accu_list]).replace('\'', ''), avg_loss,
                avg_loss_func, avg_loss_kl,
                time_end - time_start)
        else:
            str_ = 'Epoch:{:d}, val_acc={:.4%}|{:}, val_loss={:.4f}, used_time:{:.2f}s'.format(epoch + 1, accu,
                                                                                               str(accu_list), avg_loss,
                                                                                               time_end - time_start)

        # 写文件
        log('\n' + str_)

        return accu, accu_list

    def train_one_epoch_individual(self, sess, init, epoch):
        sess.run(init)

        n_batches = 0

        time_last = time.time()
        try:
            while True:
                for task_index in range(self.n_tasks):
                    _ = sess.run([self.op_opt_list[task_index]], {self.is_training: True})

                n_batches += self.n_tasks
                str_ = 'epoch={:d}, batch={:d}/{:d}, used_time:{:.2f}s'.format(
                    epoch + 1,
                    n_batches,
                    self.total_batches_train,
                    time.time() - time_last)

                print('\r' + str_, end=' ')

                time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass

        log(str_, need_print=False)

    def train(self, sess, n_epochs, lr, type='normal', save_clean=False):
        self.optimize(lr)

        save_step = read_i(self.cfg, 'train', 'save_step')
        sess.run(tf.variables_initializer(self.opt.variables()))

        for epoch in range(n_epochs):
            if type == 'normal':
                self.train_one_epoch(sess, self.train_init, epoch)
            elif type == 'individual':
                self.train_one_epoch_individual(sess, self.train_init, epoch)

            # 为了节省时间
            if (epoch + 1) % 2 == 0:
                acc, acc_list = self.eval_once(sess, self.test_init, epoch)

                # 只有偶数的时候才计算压缩率
                if self.pruning and (epoch + 1) % 2 == 0:
                    cr, cr_flops = self.get_CR(sess, self.cluster_res_dict)

                if self.save_now((epoch + 1), n_epochs, save_step):
                    if self.pruning:
                        name = '%s/tr%.2d-epo%.3d-cr%.4f-acc%.4f' % (
                            read_s(self.cfg, 'path', 'path_save'), self.cnt_train, epoch + 1, cr, acc)
                    else:
                        name = '%s/tr%.2d-epo%.3d-acc%.4f' % (
                            read_s(self.cfg, 'path', 'path_save'), self.cnt_train, epoch + 1, acc)

                    self.save_weight(sess, name)

        if save_clean:
            self.save_weight_clean(sess, '%s-CLEAN' % name)

        # Count of training
        self.cnt_train += 1
        # Save into cfg
        name_train = 'train%d' % self.cnt_train
        if not self.cfg.has_section(name_train):
            self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'n_epochs', str(n_epochs))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))
        self.cfg.set(name_train, 'acc_list', str(acc_list))

        if self.pruning:
            self.cfg.set(name_train, 'cr', str(cr))
            self.cfg.set(name_train, 'kl_factor', str(self.kl_factor))

    def get_CR(self, sess, cluster_res_dict):
        net_origin = dict()
        for name in self.structure:
            if name.startswith('c') or name.startswith('f') and name != 'fla':
                layer = [len(cluster_res_dict[name][t2c(task_index)]) for task_index in range(self.n_tasks)]
                layer.append(len(cluster_res_dict[name][t2c(-1)]))
                net_origin[name] = layer

        # 得到每一层的每个部分还有多少神经元

        def get_n(name, task_index):
            layer = self.get_layer_by_name('%s_vib/%s' % (name, t2c(task_index)))
            if layer is None:
                return tf.constant(0, dtype=tf.int32)
            else:
                if name.startswith('c'):
                    return layer.get_remained(read_f(self.cfg, 'pruning', 'ib_threshold_conv'))
                elif name.startswith('f'):
                    return layer.get_remained(read_f(self.cfg, 'pruning', 'ib_threshold_fc'))

        net_remain = dict()
        for name in self.structure:
            if name.startswith('c') or name.startswith('f') and name != 'fla':
                if name != self.structure[-1]:
                    for task_index in range(self.n_tasks):
                        layer = [get_n(name, task_index) for task_index in range(self.n_tasks)]
                    layer.append(get_n(name, -1))
                    net_remain[name] = sess.run(layer)
                else:
                    net_remain[name] = net_origin[name]

        def count(net_dict):
            # 读入图片的大小
            h, w = read_l(self.cfg, 'data', 'length')
            params, flops = 0, 0
            prod_kernel = np.prod(self.kernel_size)

            for ind, name in enumerate(self.structure):
                if name == 'p':
                    h, w = h // 2, w // 2
                elif name.startswith('c'):
                    # 默认第一层为卷积层
                    if ind == 0:
                        # 第一层，需要特殊处理
                        channel = read_i(self.cfg, 'data', 'channels')
                        for task_index in range(-1, self.n_tasks):
                            params += prod_kernel * channel * net_dict[name][task_index]
                            flops += (2 * prod_kernel * channel - 1) * h * w * net_dict[name][task_index]
                    else:
                        n_CEN_last = n_dict_last[-1]
                        for task_index in range(self.n_tasks):
                            n_in = n_CEN_last + n_dict_last[task_index]

                            params += prod_kernel * n_in * net_dict[name][task_index]
                            flops += (2 * prod_kernel * n_in - 1) * h * w * net_dict[name][task_index]

                        # CEN单独处理
                        params += prod_kernel * n_CEN_last * net_dict[name][-1]
                        flops += (2 * prod_kernel * n_CEN_last - 1) * h * w * net_dict[name][-1]

                    n_dict_last = net_dict[name]
                elif name.startswith('f') and name != 'fla':
                    n_CEN_last = n_dict_last[-1]
                    for task_index in range(self.n_tasks):
                        n_in = n_CEN_last + n_dict_last[task_index]

                        params += n_in * net_dict[name][task_index]
                        flops += (2 * n_in - 1) * net_dict[name][task_index]

                    params += n_CEN_last * net_dict[name][-1]
                    flops += (2 * n_CEN_last - 1) * net_dict[name][-1]
                    n_dict_last = net_dict[name]
                elif name == 'fla':
                    n_dict_last = [_ * h * w for _ in n_dict_last]

            return params, flops

        total_params, total_flops = count(net_origin)
        remain_params, remain_flops = count(net_remain)

        cr = float(remain_params) / total_params
        cr_flop = float(remain_flops) / total_flops

        str_ = 'Total_params={}, Remain_params={}, cr={:.4%}, Total_flops={}, Remain_flops={}, cf_flop={:.4%}\nRemained={}'.format(
            total_params, remain_params, cr, total_flops, remain_flops, cr_flop, str(net_remain))

        log(str_)

        return cr, cr_flop

    def save_weight_clean(self, sess, path_save):
        # Obtain masks
        masks_tf, names = list(), list()
        for layer in self.layers:
            if layer.layer_type == 'C_vib':
                masks_tf.append(layer.get_mask(read_f(self.cfg, 'pruning', 'ib_threshold_conv'), dtype=tf.bool))
                names.append(layer.layer_name)
            elif layer.layer_type == 'F_vib':
                masks_tf.append(layer.get_mask(read_f(self.cfg, 'pruning', 'ib_threshold_fc'), dtype=tf.bool))
                names.append(layer.layer_name)

        masks = sess.run(masks_tf)
        masks_dict = dict(zip(names, [_.tolist() for _ in masks]))

        def rm_none(list_):
            while None in list_:
                list_.remove(None)
            return list_

        weight_dict = self.fetch_weight(sess)
        h, w = read_l(self.cfg, 'data', 'length')
        flatten = False
        name_last = None

        for ind, name in enumerate(self.structure):
            if name.startswith('c') or name.startswith('f') and name != 'fla':
                for task_index in range(-1, self.n_tasks):
                    # If the layer exists
                    if self.get_layer_by_name('%s/%s' % (name, t2c(task_index))) is None:
                        continue

                    if ind == 0:
                        mask_input = np.ones(read_i(self.cfg, 'data', 'channels'), dtype=np.bool)
                    else:
                        mask_input = np.concatenate(rm_none(
                            [masks_dict.get('%s_vib/%s' % (name_last, t2c(_)), None) for _ in set([task_index, -1])]))

                        if flatten:
                            mask_input = np.concatenate([mask_input for _ in range(h * w)], axis=0)

                    # 不是最后一层
                    if ind != len(self.structure) - 1:

                        mask_output = masks_dict.get('%s_vib/%s' % (name, t2c(task_index)))

                        # Mask
                        weight_dict['%s/%s/w' % (name, t2c(task_index))] = \
                            weight_dict['%s/%s/w' % (name, t2c(task_index))][..., mask_input, :][..., mask_output]
                        weight_dict['%s/%s/b' % (name, t2c(task_index))] = \
                            weight_dict['%s/%s/b' % (name, t2c(task_index))][
                                mask_output]
                        # 输出层没有vib层

                        weight_dict['%s_vib/%s/mu' % (name, t2c(task_index))] = \
                            weight_dict['%s_vib/%s/mu' % (name, t2c(task_index))][mask_output]
                        weight_dict['%s_vib/%s/logD' % (name, t2c(task_index))] = \
                            weight_dict['%s_vib/%s/logD' % (name, t2c(task_index))][mask_output]
                    else:
                        # 最后一层
                        weight_dict['%s/%s/w' % (name, t2c(task_index))] = weight_dict[
                                                                               '%s/%s/w' % (name, t2c(task_index))][...,
                                                                           mask_input, :]

                flatten = False
                name_last = name
            elif name == 'fla':
                flatten = True
            elif name == 'p':
                h, w = h // 2, w // 2

        pickle.dump(weight_dict, open(path_save, 'wb'))
