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
from layers.bn_layer import BatchNormalizeLayer
from layers.stage_layer import StageLayer
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

        # Resnet
        self.n_block = read_l(self.cfg, 'model', 'n_block')
        self.stride = read_l(self.cfg, 'model', 'stride')

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

    def init_weights(self, path_weights_pretrain):
        def weights_variable(dim_input, dim_fc):
            return np.random.normal(loc=0, scale=np.sqrt(1. / dim_input), size=[dim_input, dim_fc]).astype(
                dtype=np.float32)

        def filters_variable(in_channel, dim_channel, kernelsz=[3, 3]):
            return np.random.normal(loc=0, scale=np.sqrt(1. / in_channel / np.prod(kernelsz)),
                                    size=[kernelsz[0], kernelsz[1], in_channel, dim_channel]).astype(np.float32)

        def ones_variable(dim):
            return np.ones(shape=[dim], dtype=np.float32)

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
                    kernelsz = self.kernel_size[ind]

                    weight_dict['%s/w' % name_layer] = filters_variable(dim_input, dim_c, [kernelsz, kernelsz])
                    weight_dict['%s/b' % name_layer] = bias_variable(dim_c)

                    dim_input = dim_c

                elif name_layer.startswith('s'):
                    dim_b = self.dimension[ind]
                    n_block = self.n_block[int(name_layer[-1]) - 1]

                    for idx_block in range(1, n_block + 1):
                        name_prefix = '%s/b%d/' % (name_layer, idx_block)
                        if idx_block == 1 and int(name_layer[-1]) != 1:
                            # Increase dimension for each first block except the first blcok of the first stage
                            weight_dict[name_prefix + 'ds/w'] = filters_variable(dim_input, dim_b, [1, 1])
                            weight_dict[name_prefix + 'ds/b'] = bias_variable(dim_b)

                        weight_dict[name_prefix + 'c1/w'] = filters_variable(dim_input, dim_b, [3, 3])
                        weight_dict[name_prefix + 'c1/b'] = bias_variable(dim_b)

                        weight_dict[name_prefix + 'bn1/beta'] = bias_variable(dim_b)
                        weight_dict[name_prefix + 'bn1/gamma'] = ones_variable(dim_b)
                        # weight_dict[name_prefix + 'bn1/mean'] = bias_variable(dim_b)
                        # weight_dict[name_prefix + 'bn1/var'] = ones_variable(dim_b)

                        weight_dict[name_prefix + 'c2/w'] = filters_variable(dim_b, dim_b, [3, 3])
                        weight_dict[name_prefix + 'c2/b'] = bias_variable(dim_b)

                        weight_dict[name_prefix + 'bn2/beta'] = bias_variable(dim_b)
                        weight_dict[name_prefix + 'bn2/gamma'] = ones_variable(dim_b)
                        # weight_dict[name_prefix + 'bn2/mean'] = bias_variable(dim_b)
                        # weight_dict[name_prefix + 'bn2/var'] = ones_variable(dim_b)

                        dim_input = dim_b

                elif name_layer == 'bn':
                    dim_bn = self.dimension[ind]

                    weight_dict['%s/beta' % name_layer] = bias_variable(dim_bn)
                    weight_dict['%s/gamma' % name_layer] = ones_variable(dim_bn)
                    # weight_dict['%s/mean' % name_layer] = bias_variable(dim_bn)
                    # weight_dict['%s/var' % name_layer] = ones_variable(dim_bn)

                    dim_input = dim_bn

        # conv层输出的feature map的维度（粗略计算,默认经过flatten）
        dim_input = 1 * 1 * read_i(self.cfg, 'model', 'filter_last')

        # fc层权重初始化
        for ind, name_layer in enumerate(self.structure):
            if name_layer.startswith('f') and name_layer != 'fla':
                dim_fc = self.dimension[ind]

                weight_dict['%s/w' % name_layer] = weights_variable(dim_input, dim_fc)
                weight_dict['%s/b' % name_layer] = bias_variable(dim_fc)

                dim_input = dim_fc

        return weight_dict

    def inference(self):
        def activate(input, type):
            if type == 'r':
                return tf.nn.relu(input)
            elif type == 's':
                return tf.nn.softmax(input)
            elif type == 't':
                return tf.nn.tanh(input)
            else:
                return input

        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            y = self.X
            self.kl_total = 0.

            for ind, name_layer in enumerate(self.structure):
                if name_layer == 'p' or name_layer == 'p_max':
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                elif name_layer == 'p_avg':
                    y = tf.nn.avg_pool(y, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='VALID')
                elif name_layer == 'fla':
                    y = tf.contrib.layers.flatten(y)
                elif name_layer.startswith('c'):
                    with tf.variable_scope(name_layer):
                        conv_layer = ConvLayer(y, self.weight_dict, self.is_training, self.regularizer_conv)
                        self.layers.append(conv_layer)
                        y = activate(conv_layer.layer_output, self.activation[ind])
                elif name_layer.startswith('s'):
                    idx_stage = int(name_layer[-1]) - 1
                    n_block, stride_init = self.n_block[idx_stage], self.stride[ind]
                    # If downsample,  than increase dimension
                    increase = stride_init == 2

                    with tf.variable_scope(name_layer):
                        stage_layer = StageLayer(y, self.weight_dict, n_block, stride_init, increase, self.is_training)
                        self.layers.append(stage_layer)
                        y = stage_layer.layer_output
                elif name_layer == 'bn':
                    # with tf.variable_scope(name_layer):
                    bn_layer = BatchNormalizeLayer(y, name_layer, self.weight_dict, self.is_training)
                    self.layers.append(bn_layer)
                    y = bn_layer.layer_output
                elif name_layer.startswith('f') and name_layer != 'fla':
                    with tf.variable_scope(name_layer):
                        fc_layer = FullConnectedLayer(y, self.weight_dict, self.regularizer_fc)
                        self.layers.append(fc_layer)
                        y = activate(fc_layer.layer_output, self.activation[ind])

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
                _, loss, loss_func, loss_regu, w = sess.run(
                    [self.op_opt, self.op_loss, self.op_loss_func, self.op_loss_regu, self.layers[-1].layer_output],
                    feed_dict={self.is_training: True})

                n_batches += 1

                if n_batches % 100 == 0:
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

    # def fetch_weight(self, sess):
    #     weight_dict = dict()
    #     weight_list = list()
    #     for layer in self.layers:
    #         weight_list.append(layer.get_params(sess))
    #     for params_dict in weight_list:
    #         for k, v in params_dict.items():
    #             weight_dict[k.split(':')[0]] = v
    #     return weight_dict

    def fetch_weight(self, sess):
        weight_dict = dict()
        for tensor in tf.trainable_variables():
            name = '/'.join(tensor.name.split('/')[1:])[:-2]
            weight_dict[name] = sess.run(tensor)
        return weight_dict

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

            # save
            if self.save_now(epoch + 1, n_epoch, save_step):
                name = '%s/tr%.2d-epo%.3d-acc%.4f' % (self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc)

                self.save_weight(sess, name)

        if n_epoch == 0:
            self.save_weight(sess, '%s/tr%.2d-epo%.3d' % (self.cfg['path']['path_save'], self.cnt_train, 0))
            acc = '-'

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


def exp(model_name, data_name, task_name, save_step, plan_train, batch_size=None, path_model=None):
    # 时间戳
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # cfg
    cfg = get_cfg(task_name, model_name, data_name, time_stamp, path_model)

    # overwrite
    if batch_size is not None:
        cfg.set('train', 'batch_size', str(batch_size))

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
    # writer = tf.summary.FileWriter('../logs', sess.graph)

    sess.run(tf.global_variables_initializer())

    # Pre test
    log_l('Pre test')
    model.eval_once(sess, model.test_init, -1)

    # Train
    log_l('')
    for plan in plan_train:
        model.train(sess=sess, n_epoch=plan['n_epochs'], lr=plan['lr'])
        model.save_cfg()
