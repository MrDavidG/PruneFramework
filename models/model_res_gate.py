# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: model_res_gate
@time: 2019-03-27 15:31

Description.
"""

from models.base_model import BaseModel
from layers.conv_gate_layer import ConvLayer
from layers.fc_gate_layer import FullConnectedLayer
from layers.bn_layer import BatchNormalizeLayer
from layers.stage_gate_layer import StageLayer
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

        # Gate
        self.scores = None

        # Resnet
        self.n_block = read_l(self.cfg, 'model', 'n_block')
        self.stride = read_l(self.cfg, 'model', 'stride')

        # load model or pretrain weights
        if self.cfg['path']['path_load'] and os.path.exists(self.cfg['path']['path_load']):
            log_l('Loading weights in %s' % self.cfg['path']['path_load'])
            self.weight_dict = pickle.load(open(self.cfg['path']['path_load'], 'rb'))
            self.initial_weight = False
        else:
            raise RuntimeError
            # log_l('Initialize weight matrix')
            # model_name = ''.join([_ for _ in self.cfg['model']['name'] if _.isalpha()])
            # if self.cfg.has_option('data', 'path_pretrain_%s' % model_name):
            #     path_pretrain = self.cfg['data']['path_pretrain_%s' % model_name]
            # else:
            #     path_pretrain = ''
            # self.weight_dict = self.init_weights(path_weights_pretrain=path_pretrain)
            # self.initial_weight = True

    # def init_weights(self, path_weights_pretrain):
    #     def weights_variable(dim_input, dim_fc):
    #         return np.random.normal(loc=0, scale=np.sqrt(1. / dim_input), size=[dim_input, dim_fc]).astype(
    #             dtype=np.float32)
    #
    #     def filters_variable(in_channel, dim_channel, kernelsz=[3, 3]):
    #         return np.random.normal(loc=0, scale=np.sqrt(1. / in_channel / np.prod(kernelsz)),
    #                                 size=[kernelsz[0], kernelsz[1], in_channel, dim_channel]).astype(np.float32)
    #
    #     def ones_variable(dim):
    #         return np.ones(shape=[dim], dtype=np.float32)
    #
    #     def bias_variable(dim_fc):
    #         return (np.zeros(shape=[dim_fc], dtype=np.float32)).astype(dtype=np.float32)
    #
    #     # Weights of conv
    #     if os.path.exists(path_weights_pretrain):
    #         weight_dict = pickle.load(open(path_weights_pretrain, 'rb'))
    #     else:
    #         weight_dict = dict()
    #         dim_input = read_i(self.cfg, 'data', 'channels')
    #
    #         for ind, name_layer in enumerate(self.structure):
    #             if name_layer.startswith('c'):
    #                 dim_c = self.dimension[ind]
    #                 kernelsz = self.kernel_size[ind]
    #
    #                 weight_dict['%s/w' % name_layer] = filters_variable(dim_input, dim_c, [kernelsz, kernelsz])
    #                 weight_dict['%s/b' % name_layer] = bias_variable(dim_c)
    #
    #                 dim_input = dim_c
    #
    #             elif name_layer.startswith('s'):
    #                 dim_b = self.dimension[ind]
    #                 n_block = self.n_block[int(name_layer[-1]) - 1]
    #
    #                 for idx_block in range(1, n_block + 1):
    #                     name_prefix = '%s/b%d/' % (name_layer, idx_block)
    #                     if idx_block == 1 and int(name_layer[-1]) != 1:
    #                         # Increase dimension for each first block except the first blcok of the first stage
    #                         weight_dict[name_prefix + 'ds/w'] = filters_variable(dim_input, dim_b, [1, 1])
    #                         weight_dict[name_prefix + 'ds/b'] = bias_variable(dim_b)
    #
    #                     weight_dict[name_prefix + 'c1/w'] = filters_variable(dim_input, dim_b, [3, 3])
    #                     weight_dict[name_prefix + 'c1/b'] = bias_variable(dim_b)
    #
    #                     weight_dict[name_prefix + 'bn1/beta'] = bias_variable(dim_b)
    #                     weight_dict[name_prefix + 'bn1/gamma'] = ones_variable(dim_b)
    #                     # weight_dict[name_prefix + 'bn1/mean'] = bias_variable(dim_b)
    #                     # weight_dict[name_prefix + 'bn1/var'] = ones_variable(dim_b)
    #
    #                     weight_dict[name_prefix + 'c2/w'] = filters_variable(dim_b, dim_b, [3, 3])
    #                     weight_dict[name_prefix + 'c2/b'] = bias_variable(dim_b)
    #
    #                     weight_dict[name_prefix + 'bn2/beta'] = bias_variable(dim_b)
    #                     weight_dict[name_prefix + 'bn2/gamma'] = ones_variable(dim_b)
    #                     # weight_dict[name_prefix + 'bn2/mean'] = bias_variable(dim_b)
    #                     # weight_dict[name_prefix + 'bn2/var'] = ones_variable(dim_b)
    #
    #                     dim_input = dim_b
    #
    #             elif name_layer == 'bn':
    #                 dim_bn = self.dimension[ind]
    #
    #                 weight_dict['%s/beta' % name_layer] = bias_variable(dim_bn)
    #                 weight_dict['%s/gamma' % name_layer] = ones_variable(dim_bn)
    #                 # weight_dict['%s/mean' % name_layer] = bias_variable(dim_bn)
    #                 # weight_dict['%s/var' % name_layer] = ones_variable(dim_bn)
    #
    #                 dim_input = dim_bn
    #
    #     # conv层输出的feature map的维度（粗略计算,默认经过flatten）
    #     dim_input = 1 * 1 * read_i(self.cfg, 'model', 'filter_last')
    #
    #     # fc层权重初始化
    #     for ind, name_layer in enumerate(self.structure):
    #         if name_layer.startswith('f') and name_layer != 'fla':
    #             dim_fc = self.dimension[ind]
    #
    #             weight_dict['%s/w' % name_layer] = weights_variable(dim_input, dim_fc)
    #             weight_dict['%s/b' % name_layer] = bias_variable(dim_fc)
    #
    #             dim_input = dim_fc
    #
    #     return weight_dict

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
        self.score()

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

    def score(self):
        self.scores = dict()

        with tf.name_scope('score'):
            for layer in self.layers[:-1]:
                if type(layer) == ConvLayer:
                    self.scores[layer.layer_name] = tf.abs(tf.gradients(self.op_loss, layer.gate)[0])
                elif type(layer) == StageLayer:
                    for block in layer.blocks:
                        for block_layer in block:
                            self.scores[block_layer.layer_name] = tf.abs(
                                tf.gradients(self.op_loss, block_layer.gate)[0])

    def fine_tune(self, sess, n_epoch, lr):
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        scores = self.prune_one_iteration(sess, self.train_init, n_epoch, 9999999)

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

    def train(self, sess, n_epoch, lr):
        # Build optimize graph
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        scores = self.prune_one_iteration(sess, self.train_init, n_epoch)

        # 做剪枝操作
        # scores_con = np.sort(np.concatenate(scores))
        # threshold = scores_con[int(len(scores_con) * 0.1)]

        # 获取mask
        # 对每一层分别进行mask
        masks = dict()
        for k, v in zip(self.scores.keys(), scores):
            threshold = np.sort(v)[int(len(v) * 0.1)]
            masks[k] = v > threshold

        # 获取CR
        cr, cr_flops = self.get_CR(masks)

        # 保存这一轮留下来的权重
        path = "%s/epoch%d-cr%.4f-crf%.4f" % (self.cfg['path']['path_save'], n_epoch, cr, cr_flops)
        self.save_weight_clean(sess, path, masks)

        return path

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

    def save_weight_clean(self, sess, save_path, mask_dict):
        weight_dict = self.fetch_weight(sess)

        mask_in = [True for _ in range(read_i(self.cfg, 'data', 'channels'))]
        for ind, key in enumerate(self.structure[:-1]):
            if key.startswith('c') or key.startswith('f') and key != 'fla':
                mask_out = mask_dict[key]

                weight_dict['%s/w' % key] = weight_dict['%s/w' % key][..., mask_in, :][..., mask_out]
                weight_dict['%s/b' % key] = weight_dict['%s/b' % key][mask_out]

                mask_in = mask_out
            elif key.startswith('bn'):
                weight_dict['bn/beta'] = weight_dict['bn/beta'][mask_in]
                weight_dict['bn/gamma'] = weight_dict['bn/gamma'][mask_in]
            elif key.startswith('s'):
                # block
                for idx_block in range(1, self.n_block[int(key[-1]) - 1] + 1):
                    mask_in_block = mask_in
                    for idx_conv in range(1, 3):
                        key_prefix = '%s/b%d/c%d' % (key, idx_block, idx_conv)
                        mask_out = mask_dict[key_prefix]

                        weight_dict['%s/w' % key_prefix] = weight_dict['%s/w' % key_prefix][..., mask_in, :][
                            ..., mask_out]
                        weight_dict['%s/b' % key_prefix] = weight_dict['%s/b' % key_prefix][mask_out]

                        key_prefix = '%s/b%d/bn%d' % (key, idx_block, idx_conv)
                        weight_dict['%s/beta' % key_prefix] = weight_dict['%s/beta' % key_prefix][mask_out]
                        weight_dict['%s/gamma' % key_prefix] = weight_dict['%s/gamma' % key_prefix][mask_out]

                        mask_in = mask_out

                    # identity
                    if self.stride[ind] == 2 and idx_block == 1:
                        key_prefix = '%s/b%d/ds' % (key, idx_block)
                        weight_dict['%s/w' % key_prefix] = weight_dict['%s/w' % key_prefix][..., mask_in_block, :][
                            ..., mask_out]
                        weight_dict['%s/b' % key_prefix] = weight_dict['%s/b' % key_prefix][mask_out]


            elif key == 'fla':
                # 最后一个展开一定是1*1
                mask_in = np.concatenate([mask_in for _ in range(1 * 1)])
            elif key.startswith('p'):
                pass

        # output
        layer_output_name = self.structure[-1]
        weight_dict['%s/w' % layer_output_name] = weight_dict['%s/w' % layer_output_name][..., mask_in, :]

        file_handler = open(save_path, 'wb')
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def get_CR(self, masks):
        def conv_pf(in_ori, out_ori, in_rem, out_rem, prod_kernel, length_fm, bn=False):
            p_ori = in_ori * out_ori * prod_kernel
            p_rem = in_rem * out_rem * prod_kernel

            f_ori = (2 * prod_kernel * in_ori - 1) * np.prod(length_fm) * out_ori
            f_rem = (2 * prod_kernel * in_rem - 1) * np.prod(length_fm) * out_rem
            if bn:
                p_ori += 2 * out_ori
                p_rem += 2 * out_rem

                f_ori += 4 * out_ori
                f_rem += 4 * out_rem
            return p_ori, p_rem, f_ori, f_rem

        remain_state = dict()
        for k, v in masks.items():
            remain_state[k] = np.sum(v)

        length_fm = read_l(self.cfg, 'data', 'length')

        total_params, remain_params = 0, 0
        total_flops, remain_flops = 0, 0

        # in_neurons, in_pruned
        in_n, in_r = read_i(self.cfg, 'data', 'channels'), read_i(self.cfg, 'data', 'channels')

        for idx, (name_layer, out_n) in enumerate(zip(self.structure[:-1], self.dimension[:-1])):
            if name_layer.startswith('c'):
                prod_kernel = np.prod([self.kernel_size[idx], self.kernel_size[idx]])
                # param
                p_ori, p_rem, f_ori, f_rem = conv_pf(in_n, out_n, in_r, remain_state[name_layer], prod_kernel,
                                                     length_fm, False)
                total_params += p_ori
                remain_params += p_rem
                # flop
                total_flops += f_ori
                remain_flops += f_rem
                # For next layer
                in_n, in_r = out_n, remain_state[name_layer]
            elif name_layer.startswith('s'):
                prod_kernel = 3 * 3
                n_block, out_n, stride, = self.n_block[int(name_layer[-1]) - 1], self.dimension[idx], self.stride[idx]

                for block in range(1, n_block + 1):
                    name_prefix = name_layer + '/b%d/' % block

                    out1_rem = remain_state[name_prefix + 'c1']
                    out2_rem = remain_state[name_prefix + 'c2']
                    # first conv and bn
                    p1_ori, p1_rem, f1_ori, f1_rem = conv_pf(in_n, out_n, in_r, out1_rem, prod_kernel, length_fm)
                    # 判断increase/downsample
                    if stride == 2:
                        length_fm = [length_fm[0] // 2, length_fm[1] // 2]
                    # second conv and bn
                    p2_ori, p2_rem, f2_ori, f2_rem = conv_pf(out_n, out_n, out1_rem, out2_rem, prod_kernel, length_fm)

                    # compute params
                    total_params += p1_ori + p2_ori
                    remain_params += p1_rem + p2_rem

                    total_flops += f1_ori + f2_ori
                    remain_flops += f1_rem + f2_rem

                    in_n, in_r = out_n, out2_rem
            elif name_layer.startswith('p'):
                length_fm = [length_fm[0] // 2, length_fm[1] // 2]
            elif name_layer == 'fla':
                in_n = in_n * np.prod(length_fm)
                in_r = in_r * np.prod(length_fm)
            elif name_layer.startswith('bn'):
                pass

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


def exp(model_name, data_name, task_name, plan_prune, plan_tune, path_model, suffix, batch_size=None):
    # 时间戳
    time_stamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # cfg
    cfg = get_cfg(task_name, model_name, data_name, time_stamp, path_model, suffix=suffix)

    if batch_size is not None:
        cfg.set('train', 'batch_size', str(batch_size))

    # begin
    log_l('Task: %s |   Data: %s  | Model: %s' % (task_name, data_name, model_name))

    log_l('')
    for n_epoch in range(plan_prune['n_epochs']):
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        log_l('Iteration %d:' % n_epoch)

        # build graph
        model = Model(cfg)
        model.build()

        sess.run(tf.global_variables_initializer())

        model.eval_once(sess, model.test_init, -1)

        path_model = model.train(sess=sess, n_epoch=n_epoch, lr=plan_prune['lr'])
        # 这里并不是剪枝后的效果，因为这里实际上还是剪枝之前的model，但是经过了训练，因为获得score的时候也进行了self.opt的训练，所以只有一开始的eval_once是上一轮的剪枝效果
        # model.eval_once(sess, model.test_init, -1)

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
