# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: model_combined_gate
@time: 2020/5/5 11:57 上午

Description.
"""

from models.base_model import BaseModel
from layers.conv_gate_layer import ConvLayer
from layers.fc_gate_layer import FullConnectedLayer
from layers.bn_layer import BatchNormalizeLayer
from utils.logger import *
from utils.json import read_i
from utils.json import read_l
from utils.json import read_s
from copy import deepcopy

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

        self.op_logits_list = None
        self.op_opt_list = None
        self.op_acc_list = None

        self.op_loss_func_list = None

        # Gate
        self.scores = None

        self.load_dataset()

        self.n_classes = self.Y.shape[1]

        # resnet
        self.n_block = read_l(config, 'model', 'n_block')
        self.stride = read_l(config, 'model', 'stride')

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

    def score(self):
        with tf.name_scope('score'):
            self.scores = dict()
            # 需要把最后一层所有的输出部分排除出去
            for layer in self.layers:
                if self.structure[-1] not in layer.layer_name:
                    gra = tf.gradients(self.op_loss, layer.gate)[0]
                    if gra is None:
                        self.scores[layer.layer_name] = tf.zeros_like(layer.gate, dtype=tf.float32)
                    else:
                        self.scores[layer.layer_name] = tf.abs(gra)

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

        def get_n_list(layer_name):
            in_list, out_list = list(), list()
            for task_index in range(self.n_tasks):
                n_in, n_out = np.shape(weight_list[task_index][layer_name + '/w'])[-2:]
                in_list.append(n_in)
                out_list.append(n_out)
            return in_list, out_list

        weight_dict = dict()
        h, w = read_l(self.cfg, 'data', 'length')

        for ind, layer_name in enumerate(self.structure):
            if layer_name.startswith('p'):
                h, w = h // 2, w // 2
            elif layer_name == 'fla':
                neuron_index_list_last = [get_expand(neuron_last, h, w, read_i(self.cfg, 'model', 'filter_last'))
                                          for neuron_last in neuron_index_list_last]
                CEN_last = get_expand(CEN_last, h, w, read_i(self.cfg, 'model', 'filter_last'))
            elif layer_name.startswith('c'):
                for task_index in range(self.n_tasks):
                    weight_dict['%s/%s/w' % (layer_name, t2c(task_index))] = weight_list[task_index][layer_name + '/w']
                    weight_dict['%s/%s/b' % (layer_name, t2c(task_index))] = weight_list[task_index][layer_name + '/b']
                neuron_index_list_last = [self.cluster_res_dict[layer_name][t2c(task_index)] for task_index in
                                          range(self.n_tasks)]
                CEN_last = self.cluster_res_dict[layer_name][t2c(-1)]
            elif layer_name.startswith('bn'):
                for task_index in range(self.n_tasks):
                    weight_dict['%s/%s/beta' % (layer_name, t2c(task_index))] = weight_list[task_index][
                        layer_name + '/beta']
                    weight_dict['%s/%s/gamma' % (layer_name, t2c(task_index))] = weight_list[task_index][
                        layer_name + '/gamma']
            elif layer_name.startswith('fc'):
                for task_index in range(self.n_tasks):
                    weight_dict['%s/%s/w' % (layer_name, t2c(task_index))] = weight_list[task_index][
                        '%s/w' % layer_name]
                    weight_dict['%s/%s/b' % (layer_name, t2c(task_index))] = weight_list[task_index][
                        '%s/b' % layer_name]
            elif layer_name.startswith('s'):
                n_block = self.n_block[int(layer_name[-1]) - 1]
                # block
                for ind_block in range(1, n_block + 1):
                    # c1 bn1
                    name_c1 = layer_name + '/b%d/c1' % ind_block
                    name_bn1 = layer_name + '/b%d/bn1' % ind_block
                    # Obtain neuron list
                    neuron_index_list = [self.cluster_res_dict[name_c1][t2c(task_index)] for task_index in
                                         range(self.n_tasks)]
                    CEN = self.cluster_res_dict[name_c1]['CEN']

                    in_list, out_list = get_n_list(name_c1)
                    weight = np.zeros(
                        shape=[3, 3, np.sum(in_list, dtype=np.int), np.sum(out_list, dtype=np.int)]).astype(np.float32)
                    # 填充weight
                    for task_index in range(self.n_tasks):
                        in_start = np.sum(in_list[:task_index], dtype=np.int)
                        out_start = np.sum(out_list[:task_index], dtype=np.int)

                        weight[..., in_start:in_start + in_list[task_index],
                        out_start:out_start + out_list[task_index]] = weight_list[task_index][name_c1 + '/w']

                    # bias
                    bias = np.concatenate(
                        [weight_list[task_index][name_c1 + '/b'] for task_index in range(self.n_tasks)]).astype(
                        np.float32)
                    # beta
                    beta = np.concatenate(
                        [weight_list[task_index][name_bn1 + '/beta'] for task_index in range(self.n_tasks)]).astype(
                        np.float32)
                    # gamma
                    gamma = np.concatenate(
                        [weight_list[task_index][name_bn1 + '/gamma'] for task_index in range(self.n_tasks)]).astype(
                        np.float32)

                    # 分割权重到各个key value对里面去
                    for task_index in range(self.n_tasks):
                        if get_signal(name_c1, task_index):
                            neurons = neuron_index_list[task_index]
                            neurons_last = neuron_index_list_last[task_index]

                            weight_dict['%s/%s/w' % (name_c1, t2c(task_index))] = \
                                weight[..., CEN_last + neurons_last, :][..., neurons]
                            weight_dict['%s/%s/b' % (name_c1, t2c(task_index))] = bias[neurons]
                            weight_dict['%s/%s/beta' % (name_bn1, t2c(task_index))] = beta[neurons]
                            weight_dict['%s/%s/gamma' % (name_bn1, t2c(task_index))] = gamma[neurons]

                            # center
                    weight_dict['%s/%s/w' % (name_c1, 'CEN')] = weight[..., CEN_last, :][..., CEN]
                    weight_dict['%s/%s/b' % (name_c1, 'CEN')] = bias[CEN]
                    weight_dict['%s/%s/beta' % (name_bn1, 'CEN')] = beta[CEN]
                    weight_dict['%s/%s/gamma' % (name_bn1, 'CEN')] = gamma[CEN]

                    # c2 bn2, without CEN
                    CEN_last, neuron_index_list_last = CEN, neuron_index_list
                    name_c2 = layer_name + '/b%d/c2' % ind_block
                    name_bn2 = layer_name + '/b%d/bn2' % ind_block

                    in_list, out_list = get_n_list(name_c2)
                    # c2 不进行切割
                    neuron_index_list = list()
                    n_start = 0
                    for task_index in range(self.n_tasks):
                        neuron_index_list.append([_ + n_start for _ in range(out_list[task_index])])
                        n_start += out_list[task_index]

                    weight = np.zeros(
                        shape=[3, 3, np.sum(in_list, dtype=np.int), np.sum(out_list, dtype=np.int)]).astype(np.float32)
                    # 填充weight
                    for task_index in range(self.n_tasks):
                        in_start = np.sum(in_list[:task_index], dtype=np.int)
                        out_start = np.sum(out_list[:task_index], dtype=np.int)

                        weight[..., in_start:in_start + in_list[task_index],
                        out_start:out_start + out_list[task_index]] = weight_list[task_index][name_c2 + '/w']

                    # 只需要改变输入的维度
                    for task_index in range(self.n_tasks):
                        neurons = neuron_index_list[task_index]
                        neurons_last = neuron_index_list_last[task_index]

                        weight_dict['%s/%s/w' % (name_c2, t2c(task_index))] = weight[..., CEN_last + neurons_last, :][
                            ..., neurons]
                        weight_dict['%s/%s/b' % (name_c2, t2c(task_index))] = weight_list[task_index][
                            '%s/b' % name_c2]
                        weight_dict['%s/%s/beta' % (name_bn2, t2c(task_index))] = weight_list[task_index][
                            '%s/beta' % name_bn2]
                        weight_dict['%s/%s/gamma' % (name_bn2, t2c(task_index))] = weight_list[task_index][
                            '%s/gamma' % name_bn2]

                    # down sampling
                    # 除了第一个stage以及只有每个stage的第一个block有downsample
                    if ind_block == 1 and int(layer_name[-1]) != 1:
                        name_ds = '%s/b%d/ds' % (layer_name, ind_block)
                        for task_index in range(self.n_tasks):
                            weight_dict['%s/%s/w' % (name_ds, t2c(task_index))] = weight_list[task_index][
                                '%s/w' % name_ds]
                            weight_dict['%s/%s/b' % (name_ds, t2c(task_index))] = weight_list[task_index][
                                '%s/b' % name_ds]

                    # 这里应该是改成这个block的分裂结果
                    neuron_index_list_last = [self.cluster_res_dict['%s/b%d' % (layer_name, ind_block)][t2c(task_index)]
                                              for task_index in range(self.n_tasks)]
                    CEN_last = self.cluster_res_dict['%s/b%d' % (layer_name, ind_block)]['CEN']

        return weight_dict

    def inference(self):
        """
        build the model of VGG_Combine
        :return:
        """

        def get_signal(layer_name, key):
            if 'c1' == layer_name:
                return True
            elif layer_name in self.signal_dict.keys():
                return self.signal_dict[layer_name][key]
            elif 'c2' in layer_name:
                return key != 'CEN'
            elif 'ds' in layer_name:
                return True
            else:
                raise RuntimeError

        def create_conv_bn_tf(input, name_c, name_b, part_name, stride=1, with_relu=True):
            if get_signal(name_c, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (name_c, part_name)):
                    conv_layer = ConvLayer(tf.concat(input, axis=-1), self.weight_dict, is_dropout=False,
                                           is_training=self.is_training, regularizer_conv=self.regularizer_conv,
                                           stride=stride)
                    self.layers.append(conv_layer)

                bn_layer = BatchNormalizeLayer(conv_layer.layer_output, '%s/%s' % (name_b, part_name), self.weight_dict,
                                               is_training=self.is_training)

                if with_relu:
                    y = tf.nn.relu(bn_layer.layer_output)
                else:
                    y = bn_layer.layer_output
                return y
            else:
                return None

        def create_conv_tf(input, name_c, part_name, stride=1):
            if get_signal(name_c, part_name):
                while None in input:
                    input.remove(None)
                with tf.variable_scope('%s/%s' % (name_c, part_name)):
                    conv_layer = ConvLayer(tf.concat(input, axis=-1), self.weight_dict, is_dropout=False,
                                           is_training=self.is_training, regularizer_conv=self.regularizer_conv,
                                           stride=stride)
                    self.layers.append(conv_layer)
                return conv_layer.layer_output
            else:
                return None

        def create_bn_tf(input, name_b, part_name):
            if input is None:
                return None
            else:
                bn_layer = BatchNormalizeLayer(input, name_b + '/' + part_name, self.weight_dict,
                                               is_training=self.is_training)
                return bn_layer.layer_output

        def create_pool_avg(input):
            if input is None:
                return None
            else:
                return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        def create_pool_max(input):
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

            # 这一层没有A，则y_A=None
            output_last_list = [None for _ in range(self.n_tasks + 1)]

            for layer_index, layer_name in enumerate(self.structure):
                if layer_name.startswith('c'):
                    # first layer
                    output_list = [create_conv_tf([x], layer_name, t2c(task_index), stride=self.stride[layer_index]) for
                                   task_index in range(self.n_tasks)]
                    # 因为在进入stage之后才会进行分割
                    indices_list = [self.cluster_res_dict[layer_name][t2c(task_index)] for task_index in
                                    range(self.n_tasks)]
                    indices_list.append(self.cluster_res_dict[layer_name][t2c(-1)])
                elif layer_name.startswith('bn'):
                    output_list = [create_bn_tf(output_last_list[task_index], layer_name, t2c(task_index)) for
                                   task_index in range(self.n_tasks)]
                elif layer_name == 'p_max':
                    output_list = [create_pool_max(output_last) for output_last in output_last_list]
                elif layer_name == 'p_avg':
                    output_list = [create_pool_avg(output_last) for output_last in output_last_list]
                elif layer_name == 'fla':
                    output_list = [create_flatten(output_last) for output_last in output_last_list]
                elif layer_name.startswith('f') and layer_name != 'fla':
                    output_list = [create_output([output_last_list[task_index]], layer_name, t2c(task_index)) for
                                   task_index in range(self.n_tasks)]
                elif layer_name.startswith('s'):
                    n_block = self.n_block[int(layer_name[-1]) - 1]
                    for ind_block in range(1, n_block + 1):
                        # spilt input， len(output_last_list) = n_task, len(indices_list)= n_task+1
                        if ind_block == 1 and int(layer_name[-1]) != 1:
                            name_identity = '%s/b%d/ds' % (layer_name, ind_block)
                            identity_list = [create_conv_tf([output_last], name_identity, t2c(task_index), 2) for
                                             task_index, output_last in enumerate(output_last_list)]
                        else:
                            identity_list = output_last_list
                        output_concat = tf.concat(output_last_list, axis=-1)

                        def gather(output_concat, indices):
                            if len(indices) == 0:
                                return None
                            else:
                                return tf.gather(output_concat, indices, axis=-1)

                        output_last_list = [gather(output_concat, indices) for indices in indices_list]
                        # c1
                        name_c1 = '%s/b%d/c1' % (layer_name, ind_block)
                        name_bn1 = '%s/b%d/bn1' % (layer_name, ind_block)
                        if ind_block == 1:
                            stride = self.stride[layer_index]
                        else:
                            stride = 1
                        output_c1_list = [
                            create_conv_bn_tf([output_last_list[task_index], output_last_list[-1]], name_c1, name_bn1,
                                              t2c(task_index), stride) for task_index in range(self.n_tasks)]
                        # CEN
                        output_c1_list.append(
                            create_conv_bn_tf([output_last_list[-1]], name_c1, name_bn1, 'CEN', stride))

                        # c2
                        name_c2 = '%s/b%d/c2' % (layer_name, ind_block)
                        name_bn2 = '%s/b%d/bn2' % (layer_name, ind_block)
                        output_c2_list = [
                            create_conv_bn_tf([output_c1_list[task_index], output_c1_list[-1]], name_c2, name_bn2,
                                              t2c(task_index), 1, False) for task_index in range(self.n_tasks)]

                        # plus identity
                        def plus(c2, identity):
                            if identity is None:
                                return c2
                            else:
                                return c2 + identity

                        output_block_list = [plus(c2, identity) for c2, identity in zip(output_c2_list, identity_list)]

                        # relu, without CEN
                        output_list = [tf.nn.relu(_) for _ in output_block_list]

                        output_last_list = output_list

                        # 因为在进入stage之后才会进行分割
                        name_block = '%s/b%d' % (layer_name, ind_block)
                        indices_list = [self.cluster_res_dict[name_block][t2c(task_index)] for task_index in
                                        range(self.n_tasks)]
                        indices_list.append(self.cluster_res_dict[name_block][t2c(-1)])

                output_last_list = output_list

            self.op_logits_list = [tf.nn.tanh(y) for y in output_list]
            self.op_logits = tf.nn.tanh(tf.concat(output_list, axis=1))

    def loss(self):
        # normal func loss
        self.op_loss_func = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
        self.op_loss_func_list = [tf.losses.mean_squared_error(labels=tf.gather(self.Y, self.Y_list[ind], axis=-1),
                                                               predictions=self.op_logits_list[ind]) for ind in
                                  range(self.n_tasks)]

        # Notice: l2 loss没有分任务，建议不要使用
        self.op_loss_regu = tf.losses.get_regularization_loss()
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

    def build(self):
        self.inference()
        self.loss()
        self.evaluate()
        self.loss()
        self.score()

    def train_one_epoch(self, sess, init, epoch, N=10):
        sess.run(init)

        avg_loss = 0
        n_batches = 0

        time_last = time.time()

        dict_ = [v for k, v in self.scores.items()]
        score_avg = [0 for _ in range(len(dict_))]

        try:
            while n_batches < N:
                _, loss, score = sess.run([self.op_opt, self.op_loss] + [dict_], feed_dict={self.is_training: True})

                avg_loss += (loss - avg_loss) / (n_batches + 1.)
                n_batches += 1

                for ind, s in enumerate(score):
                    score_avg[ind] += (s - score_avg[ind]) / n_batches

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:.4f}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        avg_loss,
                        time.time() - time_last)

                    print('\r' + str_, end=' ')
                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass

        log(str_, need_print=False)

        return score_avg

    def fetch_weight(self, sess):
        weight_dict = dict()
        for tensor in tf.trainable_variables():
            name = '/'.join(tensor.name.split('/')[1:])[:-2]
            weight_dict[name] = sess.run(tensor)
        return weight_dict

    def eval_once(self, sess, init, epoch):
        sess.run(init)

        avg_loss = 0

        corr_preds = 0
        corr_preds_list = [0 for _ in range(self.n_tasks)]

        n_batches = 0
        time_start = time.time()
        try:
            while True:
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

    def prune(self, sess, n_epoch, lr):
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        scores = self.train_one_epoch(sess, self.train_init, n_epoch)

        alpha = 0.2

        gates_list_tf = list()
        for layer in self.layers:
            if self.structure[-1] not in layer.layer_name:
                gates_list_tf.append(layer.gate)

        gates_list = sess.run(gates_list_tf)

        gates_dict = dict()
        for name, score, gate in zip(self.scores.keys(), scores, gates_list):
            # TODO: 需要排除之前删除掉的
            score_clean = score[gate != 0]
            if len(score_clean) == 1 and 's3' not in name:
                index = 0
            elif len(score_clean) == 1 and 's3' in name and 'CEN' not in name:
                index = 0
            elif len(score_clean) <= 4:
                index = 1
            else:
                index = int(len(score_clean) * alpha)

            threshold = np.sort(score_clean)[index]
            gate_new = np.float32(score >= threshold) * gate
            gates_dict['%s/g' % name] = gate_new

        log('')

        cr, cr_flops = self.get_CR(gates_dict, self.cluster_res_dict)

        path = "%s/epoch%d-cr%.4f-crf%.4f" % (self.cfg['path']['path_save'], n_epoch, cr, cr_flops)
        self.save_weight(sess, gates_dict, path)

        return path

    def get_cr_task(self, sess):
        gates_list_tf = list()
        for layer in self.layers:
            if self.structure[-1] not in layer.layer_name:
                gates_list_tf.append(layer.gate)
        gates_list = sess.run(gates_list_tf)

        gates_dict = dict()
        for key, gate in zip(self.scores.keys(), gates_list):
            gates_dict['%s/g' % key] = gate

        for task_index in range(self.n_tasks):
            copy_dict = deepcopy(gates_dict)
            for k, v in copy_dict.items():
                if '/%s' % t2c(task_index) in k:
                    copy_dict[k] = np.zeros_like(v)

            print('Task', t2c(task_index), 'turn into 0')
            self.get_CR(copy_dict, self.cluster_res_dict)

    def fine(self, sess, n_epoch, lr):
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        self.train_one_epoch(sess, self.train_init, n_epoch, N=9999999)

        log('')

        gates_list_tf = list()
        for layer in self.layers:
            if self.structure[-1] not in layer.layer_name:
                gates_list_tf.append(layer.gate)
        gates_list = sess.run(gates_list_tf)

        gates_dict = dict()
        for key, gate in zip(self.scores.keys(), gates_list):
            gates_dict['%s/g' % key] = gate

        path_save = "%s/fine-epoch%d" % (self.cfg['path']['path_save'], n_epoch)
        self.save_weight(sess, gates_dict, path_save)

        return path_save

    def get_CR(self, gates_dict, cluster_res_dict):
        p_f_array = np.zeros(4)

        def conv(n_in, n_out, r_in, r_out, kernelsz, length):
            n_params = n_in * n_out * kernelsz * kernelsz
            r_params = r_in * r_out * kernelsz * kernelsz

            n_flops = (2 * kernelsz * kernelsz * n_in - 1) * length * length * n_out
            r_flops = (2 * kernelsz * kernelsz * r_in - 1) * length * length * r_out
            return np.array([n_params, r_params, n_flops, r_flops])

        def fc(n_in, n_out, r_in, r_out):
            n_params = n_in * n_out
            r_params = r_in * r_out

            n_flops = (2 * n_in - 1) * n_out
            r_flops = (2 * r_in - 1) * r_out
            return np.array([n_params, r_params, n_flops, r_flops])

        def bn(n_out, r_out):
            return np.array([n_out, r_out, n_out, r_out]) * 2

        # 剪枝后网络结构
        n_dict = dict()
        for k, v in gates_dict.items():
            n_dict[k[:-2]] = np.sum(v)

        # TODO:

        h, w = read_l(self.cfg, 'data', 'length')

        n_in_dict = dict()
        for layer_name in self.structure:
            if layer_name.startswith('c'):
                for task_index in range(self.n_tasks):
                    n_in_dict['%s/%s' % (layer_name, t2c(task_index))] = 3
                n_out_last = [len(self.cluster_res_dict[layer_name][t2c(task_index)]) for task_index in
                              range(self.n_tasks)]
                n_out_last.append(len(self.cluster_res_dict[layer_name]['CEN']))
            elif layer_name.startswith('fc'):
                for task_index in range(self.n_tasks):
                    n_in_dict['%s/%s' % (layer_name, t2c(task_index))] = 512
            elif layer_name.startswith('s'):
                n_block = self.n_block[int(layer_name[-1]) - 1]
                for block_index in range(1, n_block + 1):
                    # c1
                    for task_index in range(self.n_tasks):
                        n_in_dict['%s/b%d/c1/%s' % (layer_name, block_index, t2c(task_index))] = n_out_last[
                                                                                                     task_index] + \
                                                                                                 n_out_last[-1]
                        n_in_dict['%s/b%d/c1/CEN' % (layer_name, block_index)] = n_out_last[-1]
                    # c2
                    for task_index in range(self.n_tasks):
                        n_in_dict['%s/b%d/c2/%s' % (layer_name, block_index, t2c(task_index))] = len(
                            self.cluster_res_dict['%s/b%d/c1' % (layer_name, block_index)][t2c(task_index)]) + len(
                            self.cluster_res_dict['%s/b%d/c1' % (layer_name, block_index)]['CEN'])

                    n_out_last = [len(self.cluster_res_dict['%s/b%d' % (layer_name, block_index)][t2c(task_index)]) for
                                  task_index in range(self.n_tasks)]
                    n_out_last.append(len(self.cluster_res_dict['%s/b%d' % (layer_name, block_index)]['CEN']))

        r_in_dict = dict()
        for layer_name in self.structure:
            if layer_name.startswith('c'):
                for task_index in range(self.n_tasks):
                    r_in_dict['%s/%s' % (layer_name, t2c(task_index))] = 3
                gate_con = np.concatenate(
                    [gates_dict['%s/%s/g' % (layer_name, t2c(task_index))] for task_index in range(self.n_tasks)])
                output_last_list = [np.sum(gate_con[cluster_res_dict[layer_name][t2c(task_index)]]) for task_index in
                                    range(self.n_tasks)]
                output_last_list.append(np.sum(gate_con[cluster_res_dict[layer_name]['CEN']]))
            elif layer_name.startswith('fc'):
                for task_index in range(self.n_tasks):
                    r_in_dict['%s/%s' % (layer_name, t2c(task_index))] = np.sum(
                        gates_dict['s4/b2/c2/%s/g' % t2c(task_index)])

            elif layer_name.startswith('s'):
                n_block = self.n_block[int(layer_name[-1]) - 1]
                for block_index in range(1, n_block + 1):
                    # c1
                    name_c1 = '%s/b%d/c1' % (layer_name, block_index)
                    for task_index in range(self.n_tasks):
                        r_in_dict['%s/%s' % (name_c1, t2c(task_index))] = output_last_list[task_index] + \
                                                                          output_last_list[-1]
                    r_in_dict['%s/CEN' % name_c1] = output_last_list[-1]

                    # c2
                    for task_index in range(self.n_tasks):
                        name_c2 = '%s/b%d/c2/%s' % (layer_name, block_index, t2c(task_index))
                        r_in_dict[name_c2] = np.sum(gates_dict.get('%s/%s/g' % (name_c1, t2c(task_index)), 0)) + np.sum(
                            gates_dict['%s/CEN/g' % name_c1])

                    gate_last_block = [gates_dict['%s/b%d/c2/%s/g' % (layer_name, block_index, t2c(task_index))] for
                                       task_index in range(self.n_tasks)]
                    gate_con = np.concatenate(gate_last_block)
                    output_last_list = [
                        np.sum(gate_con[cluster_res_dict['%s/b%d' % (layer_name, block_index)][t2c(task_index)]]) for
                        task_index in range(self.n_tasks)]
                    output_last_list.append(
                        np.sum(gate_con[cluster_res_dict['%s/b%d' % (layer_name, block_index)]['CEN']]))

        # 输入的维度
        len_dict = {'c1': 72, 'bn': 36, 's1': 18, 's2': 9, 's3': 5, 's4': 2, 'fc': 1 * 512}

        for layer_name, n_out in zip(self.structure, self.dimension + [20]):
            if layer_name.startswith('c'):
                for task_index in range(self.n_tasks):
                    name_g = '%s/%s/g' % (layer_name, t2c(task_index))
                    r_out = np.sum(gates_dict[name_g])
                    p_f_array += conv(3, n_out, 3, r_out, 7, h) + bn(n_out, r_out)

            elif layer_name.startswith('bn') or layer_name.startswith('p'):
                pass
            elif layer_name.startswith('fc'):
                for task_index in range(self.n_tasks):
                    n_in = n_in_dict['%s/%s' % (layer_name, t2c(task_index))]
                    r_in = r_in_dict['%s/%s' % (layer_name, t2c(task_index))]
                    r_out = 20
                p_f_array += fc(n_in, n_out, r_in, r_out)

            elif layer_name.startswith('s'):
                n_block = self.n_block[int(layer_name[-1]) - 1]
                for block_index in range(1, n_block + 1):
                    # c1
                    if block_index == 1:
                        len_ = len_dict[layer_name]
                    else:
                        len_ = len_dict[layer_name] // 2
                    for task_index in [_ for _ in range(self.n_tasks)] + [-1]:
                        if self.signal_dict['%s/b%d/c1' % (layer_name, block_index)][t2c(task_index)]:
                            name_c1 = '%s/b%d/c1/%s' % (layer_name, block_index, t2c(task_index))
                            r_in = r_in_dict[name_c1]
                            r_out = np.sum(gates_dict['%s/g' % name_c1])
                            n_in = n_in_dict[name_c1]
                            p_f_array += conv(n_in, n_out, r_in, r_out, 3, len_) + bn(n_out, r_out)
                    # c2
                    for task_index in range(self.n_tasks):
                        name_c2 = '%s/b%d/c2/%s' % (layer_name, block_index, t2c(task_index))
                        r_in = r_in_dict[name_c2]
                        r_out = np.sum(gates_dict['%s/g' % name_c2])
                        n_in = n_in_dict[name_c2]
                        p_f_array += conv(n_in, n_out, r_in, r_out, 3, len_dict[layer_name] // 2) + bn(n_out, r_out)

        total_params, remain_params, total_flops, remain_flops = p_f_array
        cr = remain_params / total_params
        cr_flop = remain_flops / total_flops
        str_ = 'Total_params={}, Remain_params={}, cr={:.4%}, Total_flops={}, Remain_flops={}, cf_flop={:.4%}'.format(
            total_params, remain_params, cr, total_flops, remain_flops, cr_flop)

        log(str_)

        return cr, cr_flop

    def save_weight(self, sess, gates_dict, path_save):
        weight_dict = self.fetch_weight(sess)
        if gates_dict is not None:
            weight_dict = dict(weight_dict, **gates_dict)
        pickle.dump(weight_dict, open(path_save, 'wb'))
