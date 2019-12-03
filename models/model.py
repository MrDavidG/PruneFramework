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

from models.base_model import BaseModel
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
from layers.ib_layer import InformationBottleneckLayer
from utils.configer import get_cfg
from utils.json import read_f
from utils.json import read_i
from utils.json import read_l
from utils.logger import *

import tensorflow as tf
import numpy as np

import pickle
import json
import time
import os


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.load_dataset()

        # 取决于task而不是data
        self.n_classes = read_i(self.cfg, 'task', 'n_labels')

        # load model or pretrain weights
        if self.cfg['path']['path_load'] and os.path.exists(self.cfg['path']['path_load']):
            log_l('Loading weights in %s' % self.cfg['path']['path_load'])
            self.weight_dict = pickle.load(open(self.cfg['path']['path_load'], 'rb'))
            self.initial_weight = False
        else:
            log_l('Initialize weight matrix')
            model_name = ''.join([_ for _ in self.cfg['model']['name'] if _.isalpha()])
            if self.cfg.has_option('data', 'path_pretrain_%s' % model_name):
                path_pretrain = self.cfg['data']['path_pretrain_%s' % model_name]
            else:
                path_pretrain = ''
            self.weight_dict = self.init_weights(path_weights_pretrain=path_pretrain)
            self.initial_weight = True

        if self.pruning and self.structure[0] + '/vib/mu' not in self.weight_dict.keys():
            log_l('Initialize vib weights')
            self.weight_dict = dict(self.weight_dict, **self.init_weights_vib())
        else:
            log_l('VIBNet weights exists or not in pruning')

    def init_weights(self, path_weights_pretrain):
        def weights_variable(dim_input, dim_fc):
            return np.random.normal(loc=0, scale=np.sqrt(1. / dim_input), size=[dim_input, dim_fc]).astype(
                dtype=np.float32)

        def filters_variable(in_channel, dim_channel):
            return np.random.normal(loc=0, scale=np.sqrt(1. / in_channel / 9.),
                                    size=[3, 3, in_channel, dim_channel]).astype(np.float32)

        def bias_variable(dim_fc):
            return (np.zeros(shape=[dim_fc], dtype=np.float32)).astype(dtype=np.float32)

        # Weights of conv
        if os.path.exists(path_weights_pretrain):
            weight_dict = pickle.load(open(path_weights_pretrain, 'rb'))
        else:
            weight_dict = dict()
            dim_input = read_i(self.cfg, 'data', 'channels')
            for ind, name_layer in enumerate(self.structure):
                if name_layer.startswith('c'):
                    dim_c = self.dimension[ind]

                    weight_dict['%s/w' % name_layer] = filters_variable(dim_input, dim_c)
                    weight_dict['%s/b' % name_layer] = bias_variable(dim_c)

                    dim_input = dim_c

        # conv层输出的feature map的维度（粗略计算,默认经过flatten）
        n_pooling = self.structure.count('p')
        h, w = json.loads(self.cfg['data']['length'])
        dim_input = (h // 2 ** n_pooling) * (w // 2 ** n_pooling) * read_i(self.cfg, 'model', 'filter_last')

        # fc层权重初始化
        for ind, name_layer in enumerate(self.structure):
            if name_layer.startswith('f') and name_layer != 'fla':
                dim_fc = self.dimension[ind]

                weight_dict['%s/w' % name_layer] = weights_variable(dim_input, dim_fc)
                weight_dict['%s/b' % name_layer] = bias_variable(dim_fc)

                dim_input = dim_fc

        return weight_dict

    def init_weights_vib(self):
        weight_dict = dict()
        for name_layer, dim_layer in zip(self.structure[:-1], self.dimension[:-1]):
            if dim_layer != 0:
                weight_dict['%s_vib/mu' % name_layer] = np.random.normal(loc=1, scale=0.01, size=[dim_layer]).astype(
                    np.float32)
                weight_dict['%s_vib/logD' % name_layer] = np.random.normal(loc=-9, scale=0.01, size=[dim_layer]).astype(
                    np.float32)
        return weight_dict

    def set_kl_factor(self, kl_factor):
        log_t('set kl_factor as %f' % kl_factor)
        self.kl_factor = kl_factor
        self.loss()

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
            self.kl_total = 0.

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

                    # VIBNet
                    if self.pruning:
                        with tf.variable_scope(name_layer + '_vib'):
                            kl_mult = self.kl_mult[ind] * read_f(self.cfg, 'pruning', 'gamma_conv')
                            ib_layer = InformationBottleneckLayer(y, 'C_vib', self.weight_dict, self.is_training,
                                                                  kl_mult, self.threshold)
                            self.layers.append(ib_layer)
                            y, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                elif name_layer.startswith('f') and name_layer != 'fla':
                    with tf.variable_scope(name_layer):
                        fc_layer = FullConnectedLayer(y, self.weight_dict, self.regularizer_fc)
                        self.layers.append(fc_layer)
                        y = activate(fc_layer.layer_output, self.activation[ind])

                        # VIBNet and not output layer
                    if self.pruning and ind != len(self.structure) - 1:
                        with tf.variable_scope(name_layer + '_vib'):
                            kl_mult = self.kl_mult[ind] * read_f(self.cfg, 'pruning', 'gamma_fc')
                            ib_layer = InformationBottleneckLayer(y, 'F_vib', self.weight_dict, self.is_training,
                                                                  kl_mult, self.threshold)
                            self.layers.append(ib_layer)
                            y, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

            self.op_logits = y

    def loss(self):
        with tf.name_scope('loss'):
            self.op_loss_func = tf.losses.mean_squared_error(labels=self.Y, predictions=self.op_logits)
            self.op_loss_regu = tf.losses.get_regularization_loss()

            # VIBNet
            if self.pruning:
                self.op_loss_kl = self.kl_factor * self.kl_total
                self.op_loss = self.op_loss_func + self.op_loss_regu + self.op_loss_kl
            else:
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

    def build(self, weight_dict=None):
        if weight_dict:
            self.weight_dict = weight_dict
        self.inference()
        self.evaluate()
        self.loss()

    def train_one_epoch(self, sess, init, epoch):
        sess.run(init)

        n_batches = 0

        time_last = time.time()
        try:
            while True:
                if self.pruning:
                    _, loss, loss_func, loss_kl, loss_regu = sess.run(
                        [self.op_opt, self.op_loss, self.op_loss_func, self.op_loss_kl, self.op_loss_regu],
                        feed_dict={self.is_training: True})
                else:
                    _, loss, loss_func, loss_regu, w = sess.run(
                        [self.op_opt, self.op_loss, self.op_loss_func, self.op_loss_regu, self.layers[-1].layer_output],
                        feed_dict={self.is_training: True})

                n_batches += 1

                if n_batches % 5 == 0:
                    if self.pruning:
                        str_ = 'epoch={:d}, batch={:d}/{:d}, cur_loss={:.4f}, cur_func={:.4f}, cur_kl={:.4f}, cur_regu={:.4f}, used_time:{:.2f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.total_batches_train,
                            loss,
                            loss_func,
                            loss_kl,
                            loss_regu,
                            time.time() - time_last)
                    else:
                        str_ = 'epoch={:d}, batch={:d}/{:d}, cur_loss={:.4f}, cur_func={:.4f}, cur_regu={:.4f}, used_time:{:.2f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.total_batches_train,
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

    def fetch_weight(self, sess):
        weight_dict = dict()
        weight_list = list()
        for layer in self.layers:
            weight_list.append(layer.get_params(sess))
        for params_dict in weight_list:
            for k, v in params_dict.items():
                weight_dict[k.split(':')[0]] = v
        return weight_dict

    def save_weight_clean(self, sess, save_path):
        def get_expand(array, h, w, original_channel_num_a=512):
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

        weight_dict = self.fetch_weight(sess)

        # Obtain musks
        keys = [layer.layer_name.replace('_vib', '') for layer in self.layers if layer.layer_type in ['C_vib', 'F_vib']]
        values = sess.run([layer.get_mask(self.threshold, dtype=tf.bool) for layer in self.layers if
                           layer.layer_type in ['C_vib', 'F_vib']])

        mask_dict = dict(zip(keys, values))

        h, w = read_l(self.cfg, 'data', 'length')

        mask_in = [True for _ in range(read_i(self.cfg, 'data', 'channels'))]
        for ind, key in enumerate(self.structure[:-1]):
            if key.startswith('c') or key.startswith('f') and key != 'fla':
                mask_out = mask_dict[key]

                weight_dict['%s/w' % key] = weight_dict['%s/w' % key][..., mask_in, :][..., mask_out]
                weight_dict['%s/b' % key] = weight_dict['%s/b' % key][mask_out]

                weight_dict['%s_vib/mu' % key] = weight_dict['%s_vib/mu' % key][mask_out]
                weight_dict['%s_vib/logD' % key] = weight_dict['%s_vib/logD' % key][mask_out]

                mask_in = mask_out

                if key.startswith('c'):
                    channel_last = self.dimension[ind]
            elif key == 'p':
                h, w = h // 2, w // 2
            elif key == 'fla':
                mask_in = get_expand(mask_in, h, w, channel_last)

        # output
        layer_output_name = self.structure[-1]
        weight_dict['%s/w' % layer_output_name] = weight_dict['%s/w' % layer_output_name][..., mask_in, :]

        file_handler = open(save_path, 'wb')
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def eval_once(self, sess, init, epoch):
        sess.run(init)

        avg_loss = 0
        avg_loss_kl = 0
        avg_loss_func = 0
        avg_loss_regu = 0

        total_correct_preds = 0

        n_batches = 1

        time_start = time.time()
        try:
            while True:
                if self.pruning:
                    loss_batch, loss_func, loss_kl, loss_regu, acc_batch = sess.run(
                        [self.op_loss, self.op_loss_func, self.op_loss_kl, self.op_loss_regu, self.op_accuracy],
                        feed_dict={self.is_training: False})

                    avg_loss_kl += (loss_kl - avg_loss_kl) / n_batches
                else:
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

        str_ = 'Epoch:{:d}, val_acc={:.4%}, val_loss={:.4f}|{:.4f}-{:.4f}-{:.4f}, used_time:{:.2f}s'.format(epoch + 1,
                                                                                                            acc,
                                                                                                            avg_loss,
                                                                                                            avg_loss_func,
                                                                                                            avg_loss_kl,
                                                                                                            avg_loss_regu,
                                                                                                            time_end - time_start)

        log(str_)
        return acc

    def train(self, sess, n_epoch, lr, save_clean=False):
        # Build optimize graph
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        save_step = read_i(self.cfg, 'train', 'save_step')

        for epoch in range(n_epoch):
            self.train_one_epoch(sess, self.train_init, epoch)
            acc = self.eval_once(sess, self.test_init, epoch)

            if self.pruning:
                cr, cr_flops = self.get_CR(sess)

            # save
            if self.save_now(epoch + 1, n_epoch, save_step):
                if self.pruning:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-fl%.4f-acc%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, cr_flops, acc)
                else:
                    name = '%s/tr%.2d-epo%.3d-acc%.4f' % (self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc)

                self.save_weight(sess, name)

        if save_clean:
            self.save_weight_clean(sess, '%s-CLEAN' % name)

        # Count of training
        self.cnt_train += 1
        # Save into cfg
        name_train = 'train%d' % self.cnt_train
        self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'n_epochs', str(n_epoch))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))
        if self.pruning:
            self.cfg.set(name_train, 'cr', str(cr))

    def get_CR(self, sess):
        # Obtain musks
        keys = [layer.layer_name.replace('_vib', '') for layer in self.layers if layer.layer_type in ['C_vib', 'F_vib']]
        values = sess.run(
            [layer.get_remained(self.threshold) for layer in self.layers if layer.layer_type in ['C_vib', 'F_vib']])

        remain_state = dict(zip(keys, values))

        length_fm = read_l(self.cfg, 'data', 'length')

        total_params, remain_params = 0, 0
        total_flops, remain_flops = 0, 0

        # in_neurons, in_pruned
        in_n, in_r = 3, 3
        for name_layer, out_n in zip(self.structure[:-1], self.dimension[:-1]):
            if name_layer == 'p':
                length_fm = [length_fm[0] // 2, length_fm[1] // 2]
            elif name_layer.startswith('c'):
                # param
                total_params += in_n * out_n * 9
                remain_params += in_r * remain_state[name_layer] * 9
                # flop
                total_flops += 2 * (9 * in_n + 1) * np.prod(length_fm) * out_n
                remain_flops += 2 * (9 * in_r + 1) * np.prod(length_fm) * remain_state[name_layer]
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
                continue

        # Output layer
        total_params += in_n * self.n_classes
        remain_params += in_r * self.n_classes

        total_flops += (2 * in_n - 1) * self.n_classes
        remain_flops += (2 * in_r - 1) * self.n_classes

        cr = np.around(float(remain_params) / total_params, decimals=5)
        cr_flops = np.around(float(remain_flops) / total_flops, decimals=5)

        str_1 = 'Total parameters: {}, Remaining params: {}, CR: {}'.format(total_params, remain_params, cr)
        str_2 = 'Total FLOPs: {}, Remaining FLOPs: {}, CR_FLOPs: {}'.format(total_flops, remain_flops, cr_flops)
        str_3 = 'Each layer remained: %s' % list(remain_state.values())

        log(str_1 + '\n' + str_2 + '\n' + str_3)

        return cr, cr_flops


def exp(model_name, data_name, task_name, pruning, pruning_set, save_step, plan_train_normal, plan_train_vib,
        path_model=None):
    # 时间戳
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # cfg
    if pruning:
        cfg = get_cfg(task_name, model_name, data_name, time_stamp, path_model, suffix='vib')

        cfg['basic']['pruning_method'] = 'info_bottle'
        cfg.add_section('pruning')
        for key in pruning_set.keys():
            cfg.set('pruning', key, str(pruning_set[key]))
    else:
        cfg = get_cfg(task_name, model_name, data_name, time_stamp, path_model)

    # overwrite
    cfg.set('train', 'save_step', save_step)

    # begin
    log_l('Task: %s |   Data: %s  | Model: %s' % (task_name, data_name, model_name))

    # Create model
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # build graph
    model = Model(cfg)
    model.build()

    # summary
    writer = tf.summary.FileWriter('../logs', sess.graph)

    sess.run(tf.global_variables_initializer())

    # Pre test
    log_l('Pre test')
    model.eval_once(sess, model.test_init, -1)

    # Train
    if pruning:
        model.get_CR(sess)
        log_l('')
        for plan in plan_train_vib:
            model.set_kl_factor(plan['kl_factor'])

            for set_ in plan['train']:
                model.train(sess=sess, n_epoch=set_['n_epochs'], lr=set_['lr'], save_clean=set_['save_clean'])
            model.save_cfg()
    else:
        log_l('')
        for plan in plan_train_normal:
            model.train(sess=sess, n_epoch=plan['n_epochs'], lr=plan['lr'])
            model.save_cfg()
