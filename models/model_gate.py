# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: model_gate
@time: 2020/4/30 8:41 下午

Description. 
"""

from models.base_model import BaseModel
from layers.conv_gate_layer import ConvLayer
from layers.fc_gate_layer import FullConnectedLayer
from utils.configer import get_cfg
from utils.json import read_i
from utils.json import read_l
from utils.logger import *

import tensorflow as tf
import numpy as np

import pickle
import time
import os


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.load_dataset()

        # 取决于task而不是data
        self.n_classes = read_i(self.cfg, 'task', 'n_labels')

        # Gate
        self.scores = None

        # 必须有pretrain weights
        if self.cfg['path']['path_load'] and os.path.exists(self.cfg['path']['path_load']):
            log_l('Loading weights in %s' % self.cfg['path']['path_load'])
            self.weight_dict = pickle.load(open(self.cfg['path']['path_load'], 'rb'))
            self.initial_weight = False
        else:
            print('There is no pretrained neural network.')
            exit()

    def inference(self):
        def activate(input, type):
            if type == 'r':
                return tf.nn.relu(input)
            elif type == 's':
                return tf.nn.softmax(input)
            elif type == 't':
                return tf.nn.tanh(input)
            elif type == 'N':
                return input

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            y = self.X

            for ind, name_layer in enumerate(self.structure):
                if name_layer == 'p':
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                elif name_layer == 'fla':
                    y = tf.contrib.layers.flatten(y)
                elif name_layer.startswith('c'):
                    with tf.variable_scope(name_layer):
                        conv_layer = ConvLayer(y, self.weight_dict, self.is_training, self.regularizer_conv)
                        self.layers.append(conv_layer)
                        y = activate(conv_layer.layer_output, self.activation[ind])
                elif name_layer.startswith('f') and name_layer != 'fla':
                    with tf.variable_scope(name_layer):
                        fc_layer = FullConnectedLayer(y, self.weight_dict, self.regularizer_fc)
                        self.layers.append(fc_layer)
                        y = activate(fc_layer.layer_output, self.activation[ind])

                        # Gate

            self.op_logits = y

    def loss(self):
        with tf.name_scope('loss'):
            self.op_loss_func = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
            self.op_loss_regu = tf.losses.get_regularization_loss()

            self.op_loss = self.op_loss_func + self.op_loss_regu

    def optimize(self, lr):
        def get_opt(type):
            if type == 'momentum':
                return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
            elif type == 'adam':
                return tf.train.AdamOptimizer(learning_rate=lr)
            elif type == 'sgd':
                return tf.train.GradientDescentOptimizer(learning_rate=lr)

        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.opt = get_opt(self.cfg['task']['optimizer'])
                self.op_opt = self.opt.minimize(self.op_loss)

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
                                                                                           tf.float32)

    def score(self):
        with tf.name_scope('score'):
            self.scores = dict()
            for layer in self.layers[:-1]:
                self.scores[layer.layer_name] = tf.abs(tf.gradients(self.op_loss, layer.gate)[0])

    def build(self, weight_dict=None):
        if weight_dict:
            self.weight_dict = weight_dict
        self.inference()
        self.evaluate()
        self.loss()
        self.score()

    def prune_one_iteration(self, sess, init, epoch, N=10):
        sess.run(init)

        n_batches = 0

        time_last = time.time()

        dict_ = [v for k, v in self.scores.items()]
        score_avg = [0 for _ in range(len(self.scores))]

        try:
            while n_batches < N:
                _, loss, loss_func, loss_regu, score = sess.run(
                    [self.op_opt, self.op_loss, self.op_loss_func, self.op_loss_regu] + [dict_],
                    feed_dict={self.is_training: True})

                n_batches += 1

                # 求梯度的平均值
                for ind, s in enumerate(score):
                    score_avg[ind] += (s - score_avg[ind]) / n_batches

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, cur_loss={:.4f}, cur_func={:.4f}, cur_regu={:.4f}, used_time:{:.2f}s'.format(
                        epoch,
                        n_batches,
                        10,
                        loss,
                        loss_func,
                        loss_regu,
                        time.time() - time_last)
                    print('\r' + str_, end=' ')
                    time_last = time.time()
        except tf.errors.OutOfRangeError:
            pass
        print()
        log(str_, need_print=False)

        return score_avg

    def fetch_weight(self, sess):
        weight_dict = dict()
        weight_list = list()
        for layer in self.layers:
            weight_list.append(layer.get_params(sess))
        for params_dict in weight_list:
            for k, v in params_dict.items():
                weight_dict[k.split(':')[0]] = v
        return weight_dict

    def save_weight_clean(self, sess, save_path, mask_dict):
        weight_dict = self.fetch_weight(sess)

        h, w = read_l(self.cfg, 'data', 'length')

        mask_in = [True for _ in range(read_i(self.cfg, 'data', 'channels'))]
        for ind, key in enumerate(self.structure[:-1]):
            if key.startswith('c') or key.startswith('f') and key != 'fla':
                mask_out = mask_dict[key]

                weight_dict['%s/w' % key] = weight_dict['%s/w' % key][..., mask_in, :][..., mask_out]
                weight_dict['%s/b' % key] = weight_dict['%s/b' % key][mask_out]

                mask_in = mask_out

                if key.startswith('c'):
                    channel_last = self.dimension[ind]
            elif key == 'p':
                h, w = h // 2, w // 2
            elif key == 'fla':
                # mask_in = get_expand(mask_in, h, w, channel_last)
                mask_in = np.concatenate([mask_in for _ in range(h * w)])

        # output
        layer_output_name = self.structure[-1]
        weight_dict['%s/w' % layer_output_name] = weight_dict['%s/w' % layer_output_name][..., mask_in, :]

        file_handler = open(save_path, 'wb')
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def eval_once(self, sess, init, epoch):
        sess.run(init)

        avg_loss = 0
        avg_loss_func = 0
        avg_loss_regu = 0

        total_correct_preds = 0

        n_batches = 1

        time_start = time.time()
        try:
            while True:
                loss_batch, loss_func, loss_regu, acc_batch = sess.run(
                    [self.op_loss, self.op_loss_func, self.op_loss_regu, self.op_accuracy],
                    feed_dict={self.is_training: False})

                avg_loss += (loss_batch - avg_loss) / n_batches
                avg_loss_func += (loss_func - avg_loss_func) / n_batches

                avg_loss_regu += (loss_regu - avg_loss_regu) / n_batches

                total_correct_preds += acc_batch

                n_batches += 1

        except tf.errors.OutOfRangeError:
            pass
        time_end = time.time()

        acc = total_correct_preds / self.n_samples_val

        str_ = 'Epoch:{:d}, val_acc={:.4%}, val_loss={:.4f}|{:.4f}-{:.4f}, used_time:{:.2f}s'.format(epoch + 1,
                                                                                                     acc,
                                                                                                     avg_loss,
                                                                                                     avg_loss_func,
                                                                                                     avg_loss_regu,
                                                                                                     time_end - time_start)

        log(str_)
        return acc

    def train(self, sess, n_epoch, lr):
        # Build optimize graph
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        scores = self.prune_one_iteration(sess, self.train_init, n_epoch)

        # 做剪枝操作
        scores_con = np.sort(np.concatenate(scores))
        threshold = scores_con[int(len(scores_con) * 0.1)]

        # 获取mask
        masks = dict()
        for k, v in zip(self.scores.keys(), scores):
            masks[k] = v > threshold

        # 获取CR
        cr, cr_flops = self.get_CR(masks)

        # 保存这一轮留下来的权重
        path = "%s/epoch%d-cr%.4f-crf%.4f" % (self.cfg['path']['path_save'], n_epoch, cr, cr_flops)
        self.save_weight_clean(sess, path, masks)

        return path

    def fine_tune(self, sess, n_epoch, lr):
        # Build optimize graph
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        scores = self.prune_one_iteration(sess, self.train_init, n_epoch, 999999)

        # 获取mask
        masks = dict()
        for k, v in zip(self.scores.keys(), scores):
            masks[k] = v > -2

        # 获取CR
        cr, cr_flops = self.get_CR(masks)

        # 保存这一轮留下来的权重
        path = "%s/tune_epoch%d-cr%.4f-crf%.4f" % (self.cfg['path']['path_save'], n_epoch, cr, cr_flops)
        self.save_weight(sess, path)

        return path

    def get_CR(self, masks):
        remain_state = dict()
        for k, v in masks.items():
            remain_state[k] = np.sum(v)

        length_fm = read_l(self.cfg, 'data', 'length')

        total_params, remain_params = 0, 0
        total_flops, remain_flops = 0, 0

        # in_neurons, in_pruned
        in_n, in_r = read_i(self.cfg, 'data', 'channels'), read_i(self.cfg, 'data', 'channels')
        prod_kernel = np.prod(self.kernel_size)
        for name_layer, out_n in zip(self.structure[:-1], self.dimension[:-1]):
            if name_layer == 'p':
                length_fm = [length_fm[0] // 2, length_fm[1] // 2]
            elif name_layer.startswith('c'):
                # param
                total_params += in_n * out_n * prod_kernel
                remain_params += in_r * remain_state[name_layer] * prod_kernel
                # flop
                total_flops += (2 * prod_kernel * in_n - 1) * np.prod(length_fm) * out_n
                remain_flops += (2 * prod_kernel * in_r - 1) * np.prod(length_fm) * remain_state[name_layer]
                # For next layer
                in_n, in_r = out_n, remain_state[name_layer]
            elif name_layer.startswith('f') and name_layer != 'fla':
                # param
                total_params += in_n * out_n
                remain_params += in_r * remain_state[name_layer]
                # flop
                total_flops += (2 * in_n - 1) * out_n
                remain_flops += (2 * in_r - 1) * remain_state[name_layer]
                # For next layer
                in_n, in_r = out_n, remain_state[name_layer]
            elif name_layer == 'fla':
                in_n = in_n * np.prod(length_fm)
                in_r = in_r * np.prod(length_fm)

            # print(name_layer, total_flops)

        # Output layer
        total_params += in_n * self.n_classes
        remain_params += in_r * self.n_classes

        total_flops += (2 * in_n - 1) * self.n_classes
        remain_flops += (2 * in_r - 1) * self.n_classes

        # print('output', total_flops)

        cr = np.around(float(remain_params) / total_params, decimals=5)
        cr_flops = np.around(float(remain_flops) / total_flops, decimals=5)

        str_1 = 'Total parameters: {}, Remaining params: {}, CR: {}'.format(total_params, remain_params, cr)
        str_2 = 'Total FLOPs: {}, Remaining FLOPs: {}, CR_FLOPs: {}'.format(total_flops, remain_flops, cr_flops)
        str_3 = 'Each layer remained: %s' % list(remain_state.values())

        log(str_1 + '\n' + str_2 + '\n' + str_3)

        return cr, cr_flops


def exp(model_name, data_name, task_name, plan_prune, plan_tune, path_model, suffix):
    # 时间戳
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # cfg
    cfg = get_cfg(task_name, model_name, data_name, time_stamp, path_model, suffix=suffix)

    # begin
    log_l('Task: %s |   Data: %s  | Model: %s' % (task_name, data_name, model_name))

    # Create model

    # Train
    log_l('')
    for n_epoch in range(plan_prune['n_epochs']):
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model = Model(cfg)
        model.build()

        sess.run(tf.global_variables_initializer())

        log_l('Iteration %d:' % n_epoch)

        # 这个测出来的是上一次的准确率
        model.eval_once(sess, model.test_init, -1)
        # 进行本次的训练，或者说剪枝
        path_model = model.train(sess=sess, n_epoch=n_epoch, lr=plan_prune['lr'])
        # 本次准确率回复的情况
        model.eval_once(sess, model.test_init, -1)
        # 保存下来上一轮剪枝结果
        cfg['path']['path_load'] = str(path_model)

        sess.close()

    log_l('Fine tuning')
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model(cfg)
    model.build()

    sess.run(tf.global_variables_initializer())
    # 最后在这里进行fine tuning
    for plan in plan_tune:
        for n_epoch in range(plan['n_epochs']):
            log_l('')
            model.fine_tune(sess, n_epoch, plan['lr'])
            model.eval_once(sess, model.test_init, n_epoch)

    model.save_cfg()
