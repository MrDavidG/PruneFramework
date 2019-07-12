# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: dense_model.py
@time: 2019-05-03 20:26

Description. 
"""

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
from utils.configer import process_config

import numpy as np
import pickle
import time

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DenseNet(BaseModel):
    def __init__(self, config, task_name, musk=False,
                 model_path='/local/home/david/Remote/models/model_weights/vgg_pretrain'):
        super(DenseNet, self).__init__(config)

        self.imgs_path = self.config.dataset_path + task_name + '/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.task_name = task_name

        self.is_musked = musk

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        if model_path and os.path.exists(model_path):
            self.weight_dict = pickle.load(open(model_path, 'rb'), encoding='bytes')
            print('Loading weight matrix')
            self.initial_weight = False
        else:
            self.weight_dict = self.construct_initial_weights()
            print('Initialize weight matrix')
            self.initial_weight = True

    def construct_initial_weights(self):
        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        weight_dict = dict()

        weight_dict['fc1/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 784), size=[784, 1024]).astype(
            dtype=np.float32)
        weight_dict['fc1/biases'] = bias_variable([1024])
        weight_dict['fc2/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 1024), size=[1024, 20]).astype(
            np.float32)
        weight_dict['fc2/biases'] = bias_variable([20])
        weight_dict['fc3/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 20), size=[20, 20]).astype(
            np.float32)
        weight_dict['fc3/biases'] = bias_variable([20])

        weight_dict['fc4/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 20), size=[20, 20]).astype(
            np.float32)
        weight_dict['fc4/biases'] = bias_variable([20])

        weight_dict['fc5/weights'] = np.random.normal(loc=0, scale=np.sqrt(1. / 20), size=[20, self.n_classes]).astype(
            np.float32)
        weight_dict['fc5/biases'] = bias_variable([self.n_classes])

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

            # fc layer
            y = tf.contrib.layers.flatten(y)

            for index in range(4):
                with tf.variable_scope('fc' + str(index + 1)):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc,
                                                  is_musked=self.is_musked)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)

            with tf.variable_scope('fc5'):
                fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc,
                                              is_musked=self.is_musked)
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
                # self.opt = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=0.9,
                #                                       use_nesterov=True)
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
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

        accu = np.around(total_correct_preds / self.n_samples_val, decimals=4)
        print('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}'.format(epoch + 1,
                                                                 accu,
                                                                 total_loss / n_batches))
        return accu

    def train(self, sess, n_epochs, lr=None):
        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step)

            epoch_tmp = epoch + 1
            if epoch_tmp < 20 or epoch_tmp < 100 and epoch_tmp % 10 == 0 or epoch_tmp < 1000 and epoch_tmp % 100 == 0 or epoch_tmp % 200 == 0:
                accu = self.eval_once(sess, self.test_init, epoch)

        model.save_weight(session,
                          '/local/home/david/Remote/models/model_weights/dense_' + str(self.task_name) + '_' + str(
                              accu))

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

    config = process_config("../config/dense_net.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto(intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['mnist']:
        print('training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training

        session = tf.Session(config=gpu_config)

        training = tf.placeholder(dtype=tf.bool, name='training')

        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0)

        # Step1: Train
        model = DenseNet(config, task_name, musk=False, model_path=None)
        # model_path='/local/home/david/Remote/models/model_weights/dense_mnist_0.9764')

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())
        # model.get_CR()

        model.train(sess=session, n_epochs=1000, lr=0.001)

