# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: resnet_model
@time: 2019-03-28 11:48

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/")

from models.base_model import BaseModel
from utils.config import process_config
from layers.conv_layer import ConvLayer
from layers.bn_layer import BatchNormalizeLayer
from layers.fc_layer import FullConnectedLayer
from layers.res_block import ResBlock
from layers.ib_layer import InformationBottleneckLayer
from datetime import datetime

import tensorflow as tf
import numpy as np
import pickle
import time
import os


class ResNet(BaseModel):

    def __init__(self, config, task_name, model_path=None):
        super(ResNet, self).__init__(config)

        self.init_saver()

        self.imgs_path = self.config.dataset_path + task_name + '/'
        self.meta_keys_with_default_val = {"is_merge_bn": False}

        self.task_name = task_name

        self.load_dataset()

        self.n_classes = self.Y.shape[1]

        if not model_path or not os.path.exists(model_path):
            self.initial_weight = True
            self.weight_dict = self.construct_initial_weights()
        else:
            self.weight_dict = pickle.load(open(model_path, 'rb'))
            print("loading weight matrix")
            self.initial_weight = False

    def init_saver(self):
        pass

    def construct_initial_weights(self):
        weight_dict = dict()
        weight_dict['pre_conv/weights'] = np.random.normal(loc=0., scale=np.sqrt(1 / (3 * 3 * 3)),
                                                           size=[3, 3, 3, self.config.width]).astype(np.float32)
        weight_dict['pre_conv/batch_normalization/beta'] = np.zeros(self.config.width, dtype=np.float32)
        weight_dict['pre_conv/batch_normalization/moving_mean'] = np.zeros(self.config.width, dtype=np.float32)
        weight_dict['pre_conv/batch_normalization/moving_variance'] = np.ones(self.config.width, dtype=np.float32)
        for i in range(self.config.n_group):
            for j in range(self.config.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i + 1, j + 1)
                if j == 0:
                    input_width = self.config.width * 2 ** i
                else:
                    input_width = self.config.width * 2 ** (i + 1)
                weight_dict[block_name + '/conv_1/weights'] = np.random.normal(loc=0.,
                                                                               scale=np.sqrt(1 / (3 * 3 * input_width)),
                                                                               size=[3, 3, input_width,
                                                                                     self.config.width * 2 ** (
                                                                                             i + 1)]).astype(
                    np.float32)
                weight_dict[block_name + '/conv_1/batch_normalization/beta'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[block_name + '/conv_1/batch_normalization/moving_mean'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[block_name + '/conv_1/batch_normalization/moving_variance'] = np.ones(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)

                weight_dict[block_name + '/conv_2/weights'] = np.random.normal(loc=0., scale=np.sqrt(
                    1 / (3 * 3 * self.config.width * 2 ** (i + 1))), size=[3, 3, self.config.width * 2 ** (i + 1),
                                                                           self.config.width * 2 ** (i + 1)]).astype(
                    np.float32)
                weight_dict[block_name + '/conv_2/batch_normalization/beta'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[block_name + '/conv_2/batch_normalization/moving_mean'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[block_name + '/conv_2/batch_normalization/moving_variance'] = np.ones(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)

        weight_dict['end_bn/batch_normalization/beta'] = np.zeros(self.config.width * 2 ** 3, dtype=np.float32)
        weight_dict['end_bn/batch_normalization/moving_mean'] = np.zeros(self.config.width * 2 ** 3, dtype=np.float32)
        weight_dict['end_bn/batch_normalization/moving_variance'] = np.ones(self.config.width * 2 ** 3,
                                                                            dtype=np.float32)

        weight_dict['classifier/weights'] = np.random.normal(loc=0., scale=np.sqrt(1 / 256),
                                                             size=[256, self.n_classes]).astype(np.float32)
        weight_dict['classifier/biases'] = np.zeros(self.n_classes, dtype=np.float32)
        weight_dict['is_merge_bn'] = False

        # Init information bottleneck params
        if self.prune_method == 'info_bottle':
            for i in range(self.config.n_group):
                for j in range(self.config.n_blocks_per_group):
                    block_name = 'conv{:d}_{:d}'.format(i + 1, j + 1)

        return weight_dict

    def conv_layer_names(self):
        conv_layers = list()
        conv_layers.append('pre_conv')
        for i in range(self.config.n_group):
            for j in range(self.config.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i + 1, j + 1)
                conv_layers.append(block_name + '/conv_1')
                conv_layers.append(block_name + '/conv_2')
        return conv_layers

    # merge batch norm layer to conv layer
    def merge_batch_norm_to_conv(self):
        for layer_name in self.conv_layer_names():
            weight = self.weight_dict.pop(layer_name + '/weights')
            beta = self.weight_dict.pop(layer_name + '/batch_normalization/beta')
            mean = self.weight_dict.pop(layer_name + '/batch_normalization/moving_mean')
            variance = self.weight_dict.pop(layer_name + '/batch_normalization/moving_variance')
            new_weight = weight / np.sqrt(variance)
            new_bias = beta - mean / np.sqrt(variance)
            self.weight_dict[layer_name + '/weights'] = new_weight
            self.weight_dict[layer_name + '/biases'] = new_bias

    def meta_val(self, meta_key):
        meta_key_in_weight = meta_key
        if meta_key_in_weight in self.weight_dict:
            return self.weight_dict[meta_key_in_weight]
        else:
            return self.meta_keys_with_default_val[meta_key]

    def set_meta_val(self, meta_key, meta_val):
        meta_key_in_weight = meta_key
        self.weight_dict[meta_key_in_weight] = meta_val

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

    def set_global_tensor(self, training_tensor, regu_conv, regu_fc):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
        self.regularizer_fc = regu_fc

    def is_layer_shared(self, layer_name):
        share_key = layer_name + '/is_share'
        if share_key in self.weight_dict:
            return self.weight_dict[share_key]
        return False

    def inference(self):
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            self.kl_total = 0

            with tf.variable_scope('pre_conv'):
                pre_conv_layer = ConvLayer(self.X, self.weight_dict, self.config.dropout, self.is_training,
                                           regularizer_conv=self.regularizer_conv,
                                           is_shared=self.is_layer_shared('pre_conv'),
                                           share_scope=self.share_scope, is_merge_bn=self.meta_val('is_merge_bn'))
                self.layers.append(pre_conv_layer)
                y = pre_conv_layer.layer_output

                # ib layer
                if self.prune_method == 'info_bottle':
                    ib_layer = InformationBottleneckLayer(y, self.weight_dict, is_training=self.is_training,
                                                          mask_threshold=self.prune_threshold)
                    self.layers.append(ib_layer)
                    y, ib_kld = ib_layer.layer_output
                    self.kl_total += ib_kld

            for i in range(self.config.n_group):
                for j in range(self.config.n_blocks_per_group):
                    block_name = 'conv{:d}_{:d}'.format(i + 1, j + 1)
                    scale_down = j == 0
                    with tf.variable_scope(block_name):
                        is_shared = [self.is_layer_shared(block_name + '/conv_1'),
                                     self.is_layer_shared(block_name + '/conv_2')]
                        res_block = ResBlock(y, self.weight_dict, self.config.dropout, self.is_training,
                                             self.regularizer_conv, scale_down, is_shared, share_scope=self.share_scope,
                                             is_merge_bn=self.meta_val('is_merge_bn'))
                        for layer in res_block.layers:
                            self.layers.append(layer)
                        y = res_block.layer_output

                        if self.prune_method == 'info_bottle':
                            self.kl_total += res_block.res_kld

            with tf.variable_scope("end_bn"):
                bn_layer = BatchNormalizeLayer(y, self.weight_dict, self.regularizer_conv, self.is_training)
                y = bn_layer.layer_output
                self.layers.append(bn_layer)

            y = tf.nn.relu(y)
            y = tf.reduce_mean(y, axis=[1, 2])
            with tf.variable_scope("classifier"):
                fc_layer = FullConnectedLayer(y, self.weight_dict, self.regularizer_fc)
                self.op_logits = fc_layer.layer_output
                self.layers.append(fc_layer)

    def loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.op_logits)
        l2_loss = tf.losses.get_regularization_loss()

        if self.prune_method == 'info_bottle':
            self.op_loss = tf.reduce_mean(entropy, name='loss') + l2_loss + self.kl_factor * self.kl_total
        else:
            self.op_loss = tf.reduce_mean(entropy, name='loss') + l2_loss

    def optimize(self):
        # 为了让\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=0.9,
                                                      use_nesterov=True)
                self.op_opt = self.opt.minimize(self.op_loss)

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(tf.argmax(self.op_logits, 1), tf.argmax(self.Y, 1))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self, weight_dict=None, share_scope=None, is_merge_bn=False):
        # whether merge bn and conv
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
        n_batches = 0
        time_last = time.time()
        try:
            while True:
                _, loss = sess.run([self.op_opt, self.op_loss], feed_dict={self.is_training: True})
                step += 1
                total_loss += loss
                n_batches += 1

                if n_batches % 5 == 0:
                    print('\repoch={:d},batch={:d}/{:d},curr_loss={:f},used_time:{:.2f}s'.format(epoch + 1, n_batches,
                                                                                                 self.total_batches_train,
                                                                                                 total_loss / n_batches,
                                                                                                 time.time() - time_last),
                          end=' ')
                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass
        return step

    def eval_once(self, sess, init, epoch):
        sess.run(init)
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

        print('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}'.format(epoch + 1, total_correct_preds / self.n_samples_val,
                                                                 total_loss / n_batches))
        return total_correct_preds / self.n_samples_val

    def train(self, sess, n_epochs, lr=None, save_acc_threshold=-1):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        acc_max = 0
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step)
            acc = self.eval_once(sess, self.test_init, epoch)

            if save_acc_threshold != -1 and acc > save_acc_threshold and acc > acc_max:
                self.save_weight(sess, 'model_weights/res_' + self.task_name + '_' + str(acc))

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
            '\nTesting {}, val_acc={:%}, val_loss={:f}'.format(self.task_name,
                                                               total_correct_preds / self.n_samples_val,
                                                               total_loss / n_batches))


if __name__ == '__main__':

    config = process_config("../configs/res_net.json")
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True

    # 'dtd', 'cifar100', 'daimlerpedcls', 'vgg-flowers', 'ucf101', 'aircraft', 'gtsrb', 'omniglot', 'svhn'
    for task_name in ['dtd']:
        list_accu_threshold = {'aircraft': .44, 'ucf101': .35, 'dtd': .277}
        accu_threshold = list_accu_threshold.get(task_name, -1)

        print('training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training
        with tf.device('/gpu:1'):
            training = tf.placeholder(dtype=tf.bool, name='training')
            # regularizers
            regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.001)
            regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.01)

            # Step1: Train
            resnet = ResNet(config,
                            task_name)  # , model_path='/local/home/david/Remote/models/model_weights/res_dtd_0.2776595744680851')
            resnet.set_global_tensor(training, regularizer_conv, regularizer_fc)
            resnet.build()

        session = tf.Session(config=gpu_config)
        session.run(tf.global_variables_initializer())

        resnet.train(sess=session, n_epochs=50, lr=0.01, save_acc_threshold=accu_threshold)

        # resnet.train(sess=session, n_epochs=60, lr=0.01, save_acc_threshold=accu_threshold)

        resnet.train(sess=session, n_epochs=30, lr=0.001, save_acc_threshold=accu_threshold)
        # resnet.train(sess=session, n_epochs=20, lr=0.001)

        # save the model weights
        # print('[%s] Save model weights for model res_%s' % (datetime.now(), task_name))
        # resnet.save_weight(session, 'model_weights/res_' + task_name)
