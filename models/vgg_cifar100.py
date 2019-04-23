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
from utils.time_stamp import print_with_time_stamp as print_

import tensorflow as tf
import numpy as np
import pickle
import time
import os


class VGGNet(BaseModel):
    def __init__(self, config, task_name, model_path='/local/home/david/Remote/models/model_weights/vgg_pretrain'):
        super(VGGNet, self).__init__(config)

        self.imgs_path = self.config.dataset_path + task_name + '/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        if model_path and os.path.exists(model_path):
            self.weight_dict = pickle.load(open(model_path, 'rb'), encoding='bytes')
            print_('Loading weight matrix')
            self.initial_weight = False
        else:
            self.weight_dict = self.construct_initial_weights()
            print_('Initialize weight matrix')
            self.initial_weight = True

        if self.prune_method == 'info_bottle':
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib())

    def construct_initial_weights(self):
        def bias_variable(shape):
            return (np.ones(shape=shape, dtype=np.float32) / 10).astype(dtype=np.float32)

        def weights_variable(fan_in, size):
            return np.random.normal(loc=0, scale=np.sqrt(1 / (3. * 3. * fan_in)), size=size).astype(dtype=np.float32)

        weight_dict = dict()

        # the first 2 layers
        weight_dict['conv1_1/weights'] = weights_variable(3, [3, 3, 3, 64])
        weight_dict['conv1_1/biases'] = bias_variable([64])
        weight_dict['conv1_2/weights'] = weights_variable(64, [3, 3, 64, 64])
        weight_dict['conv1_2/biases'] = bias_variable([64])
        # the second 2 layers
        weight_dict['conv2_1/weights'] = weights_variable(64, [3, 3, 64, 128])
        weight_dict['conv2_1/biases'] = bias_variable([128])
        weight_dict['conv2_2/weights'] = weights_variable(128, [3, 3, 128, 128])
        weight_dict['conv2_2/biases'] = bias_variable([128])
        # the third 3 layers
        weight_dict['conv3_1/weights'] = weights_variable(128, [3, 3, 128, 256])
        weight_dict['conv3_1/biases'] = bias_variable([256])
        weight_dict['conv3_2/weights'] = weights_variable(256, [3, 3, 256, 256])
        weight_dict['conv3_2/biases'] = bias_variable([256])
        weight_dict['conv3_3/weights'] = weights_variable(256, [3, 3, 256, 256])
        weight_dict['conv3_3/biases'] = bias_variable([256])
        # the forth 3 layers
        weight_dict['conv4_1/weights'] = weights_variable(256, [3, 3, 256, 512])
        weight_dict['conv4_1/biases'] = bias_variable([512])
        weight_dict['conv4_2/weights'] = weights_variable(512, [3, 3, 512, 512])
        weight_dict['conv4_2/biases'] = bias_variable([512])
        weight_dict['conv4_3/weights'] = weights_variable(512, [3, 3, 512, 512])
        weight_dict['conv4_3/biases'] = bias_variable([512])
        # the fifth 3 layers
        weight_dict['conv5_1/weights'] = weights_variable(512, [3, 3, 512, 512])
        weight_dict['conv5_1/biases'] = bias_variable([512])
        weight_dict['conv5_2/weights'] = weights_variable(512, [3, 3, 512, 512])
        weight_dict['conv5_2/biases'] = bias_variable([512])
        weight_dict['conv5_3/weights'] = weights_variable(512, [3, 3, 512, 512])
        weight_dict['conv5_3/biases'] = bias_variable([512])
        # fc layers
        weight_dict['fc6/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 2048), size=[2048, 4096]).astype(
            dtype=np.float32)
        weight_dict['fc6/biases'] = bias_variable([4096])
        weight_dict['fc7/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 4096), size=[4096, 4096]).astype(
            np.float32)
        weight_dict['fc7/biases'] = bias_variable([4096])
        weight_dict['fc8/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 4096),
                                                      size=[4096, self.n_classes]).astype(np.float32)
        weight_dict['fc8/biases'] = bias_variable([self.n_classes])

        return weight_dict

    def construct_initial_weights_ib(self):
        weight_dict = dict()
        # parameters of the information bottleneck
        if self.prune_method == 'info_bottle':
            dim_list = [64, 64,
                        128, 128,
                        256, 256, 256,
                        512, 512, 512,
                        512, 512, 512,
                        4096, 4096]
            for i, name_layer in enumerate(['conv1_1', 'conv1_2',
                                            'conv2_1', 'conv2_2',
                                            'conv3_1', 'conv3_2', 'conv3_3',
                                            'conv4_1', 'conv4_2', 'conv4_3',
                                            'conv5_1', 'conv5_2', 'conv5_3',
                                            'fc6', 'fc7']):
                dim = dim_list[i]
                weight_dict[name_layer + '/info_bottle/mu'] = np.random.normal(loc=9, scale=0.01,
                                                                               size=[1,
                                                                                     dim]).astype(
                    np.float32)

                weight_dict[name_layer + '/info_bottle/delta'] = np.random.normal(loc=9,
                                                                                  scale=0.01,
                                                                                  size=[1,
                                                                                        dim]).astype(
                    np.float32)

        return weight_dict

    def meta_val(self, meta_key):
        meta_key_in_weight = meta_key
        if meta_key_in_weight in self.weight_dict:
            return self.weight_dict[meta_key_in_weight]
        else:
            return self.meta_keys_with_default_val[meta_key]

    def is_layer_shared(self, layer_name):
        share_key = layer_name + '/is_share'
        if share_key in self.weight_dict:
            return self.weight_dict[share_key]
        return False

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
                              ('conv5_1', 1.0 / 2), ('conv5_2', 1.0 / 2), ('conv5_3', 1.0 / 2), 'pooling']:
                if set_layer != 'pooling':
                    conv_name, kl_mult = set_layer
                    with tf.variable_scope(conv_name):
                        conv = ConvLayer(y, self.weight_dict, self.config.dropout, self.is_training,
                                         self.regularizer_conv, is_shared=self.is_layer_shared(conv_name),
                                         share_scope=self.share_scope, is_merge_bn=self.meta_val('is_merge_bn'))
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)

                    # pruning of the method 'Information Bottleneck'
                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y, self.weight_dict, is_training=self.is_training,
                                                              kl_mult=kl_mult, mask_threshold=self.prune_threshold)
                        y, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

                else:
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # fc layer
            y = tf.contrib.layers.flatten(y)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)

                    y = tf.layers.dropout(y, training=self.is_training)

                    if self.prune_method == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y, self.weight_dict, is_training=self.is_training,
                                                              mask_threshold=self.prune_threshold)
                        y, ib_kld = ib_layer.layer_output
                        self.kl_total += ib_kld

            with tf.variable_scope('fc8'):
                fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                self.op_logits = fc_layer.layer_output

    def loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.op_logits)
        l2_loss = tf.losses.get_regularization_loss()

        # for the pruning method
        if self.prune_method == 'info_bottle':
            self.op_loss = tf.reduce_mean(entropy, name='loss') + l2_loss + self.kl_factor * self.kl_total
        else:
            self.op_loss = tf.reduce_mean(entropy, name='loss') + l2_loss

    def optimize(self):
        # 为了让bn中的\miu, \delta滑动平均
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

    def train_one_epoch(self, sess, init, epoch, step):
        sess.run(init)
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0
        time_last = time.time()
        try:
            while True:
                _, loss, accuracy_batch = sess.run([self.op_opt, self.op_loss, self.op_accuracy],
                                                   feed_dict={self.is_training: True})
                step += 1
                total_loss += loss
                total_correct_preds += accuracy_batch
                n_batches += 1

                if n_batches % 5 == 0:
                    print(
                        '\repoch={:d}, batch={:d}/{:d}, curr_loss={:f}, train_acc={:%}, used_time:{:.2f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.total_batches_train,
                            total_loss / n_batches,
                            total_correct_preds / (n_batches * self.config.batch_size),
                            time.time() - time_last),
                        end=' ')

                    time_last = time.time()

        except tf.errors.OutOfRangeError:
            pass
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

    def train(self, sess, n_epochs, lr=None):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step)
            accu = self.eval_once(sess, self.test_init, epoch)
            if epoch % 10 == 9:
                self.save_weight(sess, 'model_weights/vgg_' + self.task_name + '_' + str(accu))

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


if __name__ == '__main__':

    config = process_config("../configs/vgg_net.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto(intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['cifar100']:
        print('training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training

        session = tf.Session(config=gpu_config)

        training = tf.placeholder(dtype=tf.bool, name='training')

        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.005)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.005)

        # Step1: Train
        model = VGGNet(config, task_name, model_path=None)
        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        model.train(sess=session, n_epochs=90, lr=0.01)
        model.train(sess=session, n_epochs=40, lr=0.001)
        model.train(sess=session, n_epochs=40, lr=0.0001)
