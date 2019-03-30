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
from data_loader.image_data_generator import ImageDataGenerator
from utils.config import process_config

import tensorflow as tf
import numpy as np
import pickle
import time
import os
import keras


class VGGModel(BaseModel):
    def __init__(self, config, task_name, model_path=None):
        super(VGGModel, self).__init__(config)

        self.init_saver()

        self.imgs_path = self.config.dataset_path + task_name + '/'
        # conv with biases and without bn
        self.meta_keys_with_default_val = {"is_merge_bn": True}

        self.is_training = None
        self.regularizer_conv = None
        self.regularizer_fc = None

        self.task_name = task_name

        self.op_loss = None
        self.op_accuracy = None
        self.op_logits = None
        self.op_opt = None
        self.opt = None

        self.X = None
        self.Y = None
        self.n_classes = None
        self.test_init = None
        self.train_init = None
        self.hessian_init = None
        self.total_batches_train = None
        self.n_samples_train = None
        self.n_samples_val = None
        self.share_scope = None

        self.layers = list()

        self.load_dataset()

        self.n_classes = self.Y.shape[1]

        if not model_path:
            self.initial_weight = True
            self.weight_dict = self.construct_initial_weights()
        else:
            self.weight_dict = pickle.load(open(model_path, 'rb'))
            print('loading weight matrix')
            self.initial_weight = False

    def init_saver(self):
        pass

    # set the initialized weight and shape for each layer
    def construct_initial_weights(self):
        weight_dict_pre_train = np.load('../datasets/vgg16.npy', encoding='latin1').item()
        weight_dict = dict()

        # the first 2 layers
        weight_dict[self.task_name + '/conv1_1/weights'] = weight_dict_pre_train['conv1_1'][0]
        weight_dict[self.task_name + '/conv1_1/biases'] = weight_dict_pre_train['conv1_1'][1]
        weight_dict[self.task_name + '/conv1_2/weights'] = weight_dict_pre_train['conv1_2'][0]
        weight_dict[self.task_name + '/conv1_2/biases'] = weight_dict_pre_train['conv1_2'][1]
        # the second 2 layers
        weight_dict[self.task_name + '/conv2_1/weights'] = weight_dict_pre_train['conv2_1'][0]
        weight_dict[self.task_name + '/conv2_1/biases'] = weight_dict_pre_train['conv2_1'][1]
        weight_dict[self.task_name + '/conv2_2/weights'] = weight_dict_pre_train['conv2_2'][0]
        weight_dict[self.task_name + '/conv2_2/biases'] = weight_dict_pre_train['conv2_2'][1]
        # the third 3 layers
        weight_dict[self.task_name + '/conv3_1/weights'] = weight_dict_pre_train['conv3_1'][0]
        weight_dict[self.task_name + '/conv3_1/biases'] = weight_dict_pre_train['conv3_1'][1]
        weight_dict[self.task_name + '/conv3_2/weights'] = weight_dict_pre_train['conv3_2'][0]
        weight_dict[self.task_name + '/conv3_2/biases'] = weight_dict_pre_train['conv3_2'][1]
        weight_dict[self.task_name + '/conv3_3/weights'] = weight_dict_pre_train['conv3_3'][0]
        weight_dict[self.task_name + '/conv3_3/biases'] = weight_dict_pre_train['conv3_3'][1]
        # the forth 3 layers
        weight_dict[self.task_name + '/conv4_1/weights'] = weight_dict_pre_train['conv4_1'][0]
        weight_dict[self.task_name + '/conv4_1/biases'] = weight_dict_pre_train['conv4_1'][1]
        weight_dict[self.task_name + '/conv4_2/weights'] = weight_dict_pre_train['conv4_2'][0]
        weight_dict[self.task_name + '/conv4_2/biases'] = weight_dict_pre_train['conv4_2'][1]
        weight_dict[self.task_name + '/conv4_3/weights'] = weight_dict_pre_train['conv4_3'][0]
        weight_dict[self.task_name + '/conv4_3/biases'] = weight_dict_pre_train['conv4_3'][1]
        # the fifth 3 layers
        weight_dict[self.task_name + '/conv5_1/weights'] = weight_dict_pre_train['conv5_1'][0]
        weight_dict[self.task_name + '/conv5_1/biases'] = weight_dict_pre_train['conv5_1'][1]
        weight_dict[self.task_name + '/conv5_2/weights'] = weight_dict_pre_train['conv5_2'][0]
        weight_dict[self.task_name + '/conv5_2/biases'] = weight_dict_pre_train['conv5_2'][1]
        weight_dict[self.task_name + '/conv5_3/weights'] = weight_dict_pre_train['conv5_3'][0]
        weight_dict[self.task_name + '/conv5_3/biases'] = weight_dict_pre_train['conv5_3'][1]
        # the full connected layer
        weight_dict[self.task_name + '/fc6/weights'] = weight_dict_pre_train['fc6'][0]
        weight_dict[self.task_name + '/fc6/biases'] = weight_dict_pre_train['fc6'][1]
        weight_dict[self.task_name + '/fc7/weights'] = weight_dict_pre_train['fc7'][0]
        weight_dict[self.task_name + '/fc7/biases'] = weight_dict_pre_train['fc7'][1]
        weight_dict[self.task_name + '/fc8/weights'] = weight_dict_pre_train['fc8'][0]
        weight_dict[self.task_name + '/fc8/biases'] = weight_dict_pre_train['fc8'][1]

        return weight_dict

    def meta_val(self, meta_key):
        meta_key_in_weight = self.task_name + '/' + meta_key
        if meta_key_in_weight in self.weight_dict:
            return self.weight_dict[meta_key_in_weight]
        else:
            return self.meta_keys_with_default_val[meta_key]

    def is_layer_shared(self, layer_name):
        share_key = self.task_name + '/' + layer_name + '/is_share'
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
            for conv_name in ['conv1_1', 'conv1_2', 'pooling',
                              'conv2_1', 'conv2_2', 'pooling',
                              'conv3_1', 'conv3_2', 'conv3_3', 'pooling',
                              'conv4_1', 'conv4_2', 'conv4_3', 'pooling',
                              'conv5_1', 'conv5_2', 'conv5_3', 'pooling']:
                if conv_name is not 'pooling':
                    with tf.variable_scope(conv_name):
                        conv = ConvLayer(y, self.weight_dict, self.config.dropout, self.is_training,
                                         self.regularizer_conv, is_shared=self.is_layer_shared(conv_name),
                                         share_scope=self.share_scope, is_merge_bn=self.meta_val('is_merge_bn'))
                        self.layers.append(conv)
                        y = conv.layer_output
                else:
                    y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # fc layer
            # y = keras.layers.Flatten(y)
            y = tf.contrib.layers.flatten(y)

            for fc_name in ['fc6', 'fc7']:
                with tf.variable_scope(fc_name):
                    fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                    self.layers.append(fc_layer)
                    y = tf.nn.relu(fc_layer.layer_output)
            with tf.variable_scope('fc8'):
                fc_layer = FullConnectedLayer(y, self.weight_dict, regularizer_fc=self.regularizer_fc)
                self.layers.append(fc_layer)
                self.op_logits = tf.nn.softmax(fc_layer.layer_output)

    def load_dataset(self):
        dataset_train, dataset_val, dataset_hessian, self.total_batches_train, self.n_samples_train, self.n_samples_val = ImageDataGenerator.load_dataset(
            self.config.batch_size, self.config.cpu_cores, self.task_name,
            self.imgs_path)
        self.train_init, self.test_init, self.hessian_init, self.X, self.Y = ImageDataGenerator.dataset_iterator(
            dataset_train,
            dataset_val, dataset_hessian)

    def loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.op_logits)
        l2_loss = tf.losses.get_regularization_loss()
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
            meta_key_in_weight = self.task_name + '/' + meta_key
            weight_dict[meta_key_in_weight] = self.meta_val(meta_key)
        return weight_dict

    def save_weight(self, sess, save_path):
        self.weight_dict = self.fetch_weight(sess)
        file_handler = open(save_path, 'wb')
        pickle.dump(self.weight_dict, file_handler)
        file_handler.close()

    def train(self, sess, n_epochs, lr=None):
        # writer = tf.summary.FileWriter('graphs/convnet', tf.get_default_graph())

        if lr is not None:
            self.config.learning_rate = lr
            self.optimize()

        sess.run(tf.variables_initializer(self.opt.variables()))
        step = self.global_step_tensor.eval(session=sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(sess, self.train_init, epoch, step)
            self.eval_once(sess, self.test_init, epoch)

    def prune(self):
        # TODO
        # self.layers
        #
        # for layer in self.layers:
        #     layer.output
        #     hessian =
        #     hessian_inverse =

        # caculate the optimal parameter change and the sensitivity for each parameter at layer l

        pass

    def retrain(self):
        # TODO
        pass


if __name__ == '__main__':

    config = process_config("../configs/vgg_net.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['mnist']:
        print('training on task {:s}'.format(task_name))
        tf.reset_default_graph()
        # session for training
        session = tf.Session(config=gpu_config)
        training = tf.placeholder(dtype=tf.bool, name='training')
        # regularizer of the conv layer
        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
        # regularizer of the fc layer
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)

        # Step1: Train
        resnet = VGGModel(config, task_name)
        resnet.set_global_tensor(training, regularizer_conv, regularizer_fc)
        resnet.build()
        session.run(tf.global_variables_initializer())

        resnet.train(sess=session, n_epochs=1, lr=0.1)

        resnet.train(sess=session, n_epochs=20, lr=0.01)

        resnet.train(sess=session, n_epochs=20, lr=0.001)

        # save the model weights
        if not os.path.exists('model_weights'):
            os.mkdir('model_weights')
        resnet.save_weight(session, 'model_weights/' + task_name)

    # Step2: Analyze & Step3: Prune
    resnet.prune()

    # Step4: Retrain
