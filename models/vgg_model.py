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
from layers.fc_layer import FullConnectedLayer
from layers.conv_layer import ConvLayer

import tensorflow as tf
import numpy as np
import pickle


class VGGModel(BaseModel):
    def __init__(self, config, task_name, model_path=None):
        super(VGGModel, self).__init__(config)

        self.init_saver()

        self.imgs_path = self.config.dataset_path + task_name + '/'
        self.meta_keys_with_default_val = {"is_merge_bn": False}

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

    def construct_initial_weights(self):
        weight_dict = dict()
        weight_dict[self.task_name + '/pre_conv/weights'] = np.random.normal(loc=0., scale=np.sqrt(1 / (3 * 3 * 3)),
                                                                             size=[3, 3, 3, self.config.width]).astype(
            np.float32)

    def inference(self):
        """
        build the model
        :return:
        """
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('pre_conv'):
                pre_conv_layer = ConvLayer(self.X, self.weight_dict, self.config.dropout, self.is_training,
                                           self.regularizer_conv, is_shared=self.is_layer_shared('pre_conv'),
                                           share_scope=self.share_scope, is_merge_bn=self.meta_val('is_merge_bn'))


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
