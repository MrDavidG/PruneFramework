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


class VGGNet(BaseModel):
    def __init__(self, config, task_name, musk=False, gamma=1., model_path=None):
        super(VGGNet, self).__init__(config)

        if task_name in ['celeba', 'celeba1', 'celeba2']:
            self.imgs_path = self.config.dataset_path + 'celeba/'
        else:
            self.imgs_path = self.config.dataset_path + task_name + '/'

        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name

        self.is_musked = musk

        self.gamma = gamma

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        if model_path and os.path.exists(model_path):
            # Directly load all weights
            self.weight_dict = pickle.load(open(model_path, 'rb'))
            print('[%s] Loading weight matrix in %s' % (datetime.now(), model_path))
            self.initial_weight = False
        else:
            # Use pre-train weights in conv, but init weights in fc
            self.weight_dict = self.construct_initial_weights()
            print('[%s] Initialize weight matrix' % (datetime.now()))
            self.initial_weight = True

        if self.prune_method == 'info_bottle' and 'conv1_1/info_bottle/mu' not in self.weight_dict.keys():
            print('初始ib层权重')
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib())
        else:
            print('!!!!!!!!未初始化ib权重，model_path中自带或未启动ib pruning!!!!!!!!')

    def construct_initial_weights(self, model_path='/local/home/david/Remote/datasets/vgg_imdb_pretrain_2'):
        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        # Weights of conv
        weight_dict = pickle.load(open(model_path, 'rb'))

        # fc layers
        dim_fc = np.int(self.X.shape[2] // 32) ** 2 * 512
        weight_dict['fc6/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / dim_fc), size=[dim_fc, 512]).astype(
            dtype=np.float32)
        weight_dict['fc6/biases'] = bias_variable([512])
        weight_dict['fc7/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 512), size=[512, 512]).astype(
            np.float32)
        weight_dict['fc7/biases'] = bias_variable([512])
        weight_dict['fc8/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 512),
                                                      size=[512, self.n_classes]).astype(np.float32)
        weight_dict['fc8/biases'] = bias_variable([self.n_classes])

        return weight_dict

    def construct_initial_weights_ib(self):
        weight_dict = dict()
        # parameters of the information bottleneck
        if self.prune_method == 'info_bottle':
            dim_list = np.array([64, 64,
                                 128, 128,
                                 256, 256, 256,
                                 512, 512, 512,
                                 512, 512, 512,
                                 512, 512])
            for i, name_layer in enumerate(['conv1_1', 'conv1_2',
                                            'conv2_1', 'conv2_2',
                                            'conv3_1', 'conv3_2', 'conv3_3',
                                            'conv4_1', 'conv4_2', 'conv4_3',
                                            'conv5_1', 'conv5_2', 'conv5_3',
                                            'fc6', 'fc7']):
                dim = dim_list[i]
                weight_dict[name_layer + '/info_bottle/mu'] = np.random.normal(loc=1, scale=0.01,
                                                                               size=[dim]).astype(
                    np.float32)

                weight_dict[name_layer + '/info_bottle/logD'] = np.random.normal(loc=-9,
                                                                                 scale=0.01,
                                                                                 size=[dim]).astype(
                    np.float32)

        return weight_dict

    def inference(self):
        """
        build the model of VGGNet16
        :return:
        """
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            y = self.X

            self.kl_total = 0.
            # the name of the layer and the coefficient of the kl divergence
            for set_layer in [('conv1_1', 1.0 / 32), ('conv1_2', 1.0 / 32), 'pooling',
                              ('conv2_1', 1.0 / 16), ('conv2_2', 1.0 / 16), 'pooling',
                              ('conv3_1', 1.0 / 8), ('conv3_2', 1.0 / 8), ('conv3_3', 1.0 / 8), 'pooling',
                              ('conv4_1', 1.0 / 4), ('conv4_2', 1.0 / 4), ('conv4_3', 1.0 / 4), 'pooling',
                              ('conv5_1', 1.0 / 1), ('conv5_2', 1.0 / 1), ('conv5_3', 1.0 / 1), 'pooling']:
                if set_layer != 'pooling':
                    conv_name, kl_mult = set_layer
                    with tf.variable_scope(conv_name):
                        conv = ConvLayer(y, self.weight_dict, self.config.dropout, self.is_training, self.is_musked,
                                         self.regularizer_conv, is_shared=self.is_layer_shared(conv_name),
                                         share_scope=self.share_scope, is_merge_bn=self.meta_val(
                                'is_merge_bn'))
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)

                        # Pruning of the method 'Information Bottleneck'
                        if self.prune_method == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                            self.layers.append(ib_layer)
                            y, ib_kld = ib_layer.layer_output
                            self.kl_total += ib_kld

                else:
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # fc layer
            y = tf.contrib.layers.flatten(y)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc,
                                                  is_musked=self.is_musked)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)

                    y = tf.layers.dropout(y, training=self.is_training)

                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y, layer_type='F_ib', weight_dict=self.weight_dict,
                                                              is_training=self.is_training,
                                                              kl_mult=self.gamma,
                                                              mask_threshold=self.prune_threshold)
                        self.layers.append(ib_layer)
                        y, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

            with tf.variable_scope('fc8'):
                # 最后的输出层不做剪枝
                fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc,
                                              is_musked=False)
                self.layers.append(fc_layer)
                self.op_logits = tf.nn.tanh(fc_layer.layer_output)

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

    def train_one_epoch(self, sess, init, epoch, step, time_stamp):
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
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, train_acc={:%}, train_kl={:f}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        total_loss / n_batches,
                        total_correct_preds / (n_batches * self.config.batch_size),
                        total_kl / n_batches,
                        time.time() - time_last)

                    print('\r' + str_, end=' ')

                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass

        # 写文件
        with open(
                '/local/home/david/Remote/models/model_weights/log_vgg512_' + self.task_name + '_ib_' + time_stamp,
                'a+') as f:
            f.write(str_ + '\n')

        return step

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

    def eval_once(self, sess, init, epoch, time_stamp):
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
        str_ = 'Epoch:{:d}, val_acc={:%}, val_loss={:f}, used_time:{:.2f}s'.format(epoch + 1,
                                                                                   accu,
                                                                                   total_loss / n_batches,
                                                                                   time_end - time_start)
        print('\n' + str_)

        # 写文件
        if time_stamp is not None:
            with open(
                    '/local/home/david/Remote/models/model_weights/log_vgg512_' + self.task_name + '_ib_' + time_stamp,
                    'a+') as f:
                f.write(str_ + '\n')

        return accu

    def train(self, sess, n_epochs, lr, time_stamp):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step, time_stamp)
            accu = self.eval_once(sess, self.test_init, epoch, time_stamp)

            if self.prune_method == 'info_bottle':
                cr = self.get_CR(sess, time_stamp)

            if (epoch + 1) % 10 == 0:
                if self.prune_method == 'info_bottle':
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_ib_' + self.task_name + '_' + str(
                        self.prune_threshold) + '_' + str(
                        np.around(accu, decimals=6)) + '_cr-' + str(np.around(cr, decimals=5))
                else:
                    save_path = '/local/home/david/Remote/models/model_weights/vgg512_' + self.task_name + '_' + str(
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

    def get_CR(self, sess, time_stamp=None):
        name_len_dict = [72, 72,
                         36, 36,
                         18, 18, 18,
                         9, 9, 9,
                         4, 4, 4]

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
        total_flops, pruned_flops, remain_flops = 0, 0, 0

        # for conv layers
        in_channels, in_pruned = 3, 0
        for n, n_out in enumerate([64, 64,
                                   128, 128,
                                   256, 256, 256,
                                   512, 512, 512,
                                   512, 512, 512,
                                   512, 512]):

            if n < 13:
                # Conv
                total_params += in_channels * n_out * 9
                remain_params += (in_channels - in_pruned) * (n_out - prune_state[n]) * 9

                M = name_len_dict[n]
                total_flops += 2 * (9 * in_channels + 1) * M * M * n_out
                remain_flops += 2 * (9 * (in_channels - in_pruned) + 1) * M * M * (n_out - prune_state[n])
            else:
                if n == 13:
                    # Fc
                    total_params += in_channels * 4 * n_out
                    remain_params += (in_channels - in_pruned) * 4 * (n_out - prune_state[n])

                    total_flops += (2 * in_channels * 4 - 1) * n_out
                    remain_flops += (2 * (in_channels - in_pruned) * 4 - 1) * (n_out - prune_state[n])

                else:
                    # Fc
                    total_params += in_channels * n_out
                    remain_params += (in_channels - in_pruned) * (n_out - prune_state[n])

                    total_flops += (2 * in_channels - 1) * n_out
                    remain_flops += (2 * (in_channels - in_pruned) - 1) * (n_out - prune_state[n])

            # For next layer
            in_channels = n_out
            in_pruned = prune_state[n]

        # Output layer
        total_params += in_channels * n_classes
        remain_params += (in_channels - in_pruned) * n_classes
        pruned_params = total_params - remain_params

        total_flops += (2 * in_channels - 1) * n_classes
        remain_flops += (2 * (in_channels - in_pruned) - 1) * n_classes
        pruned_flops = total_flops - remain_flops

        str_1 = 'Total parameters: {}, Pruned parameters: {}, Remaining params:{}, Remain/Total params:{}, Each layer pruned: {}'.format(
            total_params, pruned_params, remain_params, float(total_params - pruned_params) / total_params, prune_state)

        str_2 = 'Total FLOPs: {}, Pruned FLOPs: {}, Remaining FLOPs: {}, Remain/Total FLOPs:{}'.format(total_flops,
                                                                                                       pruned_flops,
                                                                                                       remain_flops,
                                                                                                       np.around(float(
                                                                                                           remain_flops) / total_flops,
                                                                                                                 decimals=5))

        print(str_1)
        print(str_2)

        if time_stamp is not None:
            with open(
                    '/local/home/david/Remote/models/model_weights/log_vgg512_' + self.task_name + '_ib_' + time_stamp,
                    'a+') as f:
                f.write(str_1 + '\n')
                f.write(str_2 + '\n')

        return float(total_params - pruned_params) / total_params


if __name__ == '__main__':
    config = process_config("../configs/ib_vgg.json")
    # config = process_config("../configs/vgg_net.json")

    # apply video memory dynamically
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['celeba']:
        print('Training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training

        session = tf.Session(config=gpu_config)

        training = tf.placeholder(dtype=tf.bool, name='training')

        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.00)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.00)

        # Train
        model = VGGNet(config, task_name, musk=False, gamma=15,
                       model_path='/local/home/david/Remote/models/model_weights/vgg512_celeba_0.89747_best')
        # model_path='/local/home/david/Remote/models/model_weights/vgg512_celeba2_0.892631_best')
        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())
        model.eval_once(session, model.test_init, -1, None)

        time_stamp = str(datetime.now())

        # 先是1e-5进行30个epoch，然后变成1e-6来进行30个epoch
        model.train(sess=session, n_epochs=30, lr=0.01, time_stamp=time_stamp)
        print('————————————————————改变为1e-6之后, 重新训练10个epoch————————————————————')
        model.kl_factor = 1e-6
        model.loss()
        model.optimize()
        model.evaluate()
        model.eval_once(session, model.test_init, -1, None)
        model.train(sess=session, n_epochs=20, lr=0.01, time_stamp=time_stamp)

        # model.train(sess=session, n_epochs=30, lr=0.0001)


def exp_celeba(task_name_list, model_path=0, regularizer=0, is_vib=False, gamma=None):
    if is_vib:
        config = process_config("../configs/ib_vgg.json")
    else:
        config = process_config("../configs/vgg_net.json")

    # apply video memory dynamically
    gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in task_name_list:
        print('Training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training

        session = tf.Session(config=gpu_config)

        training = tf.placeholder(dtype=tf.bool, name='training')

        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=regularizer)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=regularizer)

        # Train
        model = VGGNet(config, task_name, musk=False, gamma=gamma, model_path=model_path)
        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())
        model.eval_once(session, model.test_init, -1)

        model.train(sess=session, n_epochs=30, lr=0.01)

        model.train(sess=session, n_epochs=30, lr=0.001)

        model.train(sess=session, n_epochs=40, lr=0.0001)
