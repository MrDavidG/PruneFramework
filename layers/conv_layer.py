# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: conv_layer.py
@time: 2019-03-27 11:44

Description. 
"""

from layers.base_layer import BaseLayer

import tensorflow as tf


class ConvLayer(BaseLayer):
    def __init__(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1,
                 is_shared=False, share_scope=None, is_merge_bn=False):
        super(ConvLayer, self).__init__()
        self.layer_type = 'C'

        if is_shared:
            if is_merge_bn:
                self.create_share_merge(x, weight_dict, is_dropout, is_training, regularizer_conv, stride, share_scope)
            else:
                self.create_share(x, weight_dict, is_dropout, is_training, regularizer_conv, stride, share_scope)
        else:
            if is_merge_bn:
                self.create_merge(x, weight_dict, is_dropout, is_training, regularizer_conv, stride)
            else:
                self.create(x, weight_dict, is_dropout, is_training, regularizer_conv, stride)

    def create(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1):
        self.layer_input = x

        filt, beta, mean, variance = self.get_conv_filter_bn(weight_dict, regularizer_conv)

        conv = tf.nn.conv2d(x, filt, [1, stride, stride, 1], padding='SAME')

        if is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=is_training)

        bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training=is_training,
                                           beta_initializer=beta, scale=False, moving_mean_initializer=mean,
                                           moving_variance_initializer=variance,
                                           beta_regularizer=regularizer_conv)
        self.weight_tensors = [filt, tf.get_variable('batch_normalization/beta'),
                               tf.get_variable('batch_normalization/moving_mean'),
                               tf.get_variable('batch_normalization/moving_variance')]
        self.layer_output = bn

    def create_share(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1,
                     share_scope=None):
        self.layer_input = x
        weights_specific, beta, mean, variance, permutation, weights_shared = self.get_conv_filter_shared(weight_dict,
                                                                                                          regularizer_conv,
                                                                                                          share_scope)
        weights = tf.concat([weights_shared, weights_specific], axis=-1)

        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

        bn = tf.layers.batch_normalization(conv, momentum=0.1, epsilon=1e-05, training=is_training,
                                           beta_initializer=beta, scale=False, moving_mean_initializer=mean,
                                           moving_variance_initializer=variance,
                                           beta_regularizer=regularizer_conv)
        share_tensor = tf.constant(True, name='is_share')
        self.weight_tensors = [weights_specific, permutation, tf.get_variable('batch_normalization/beta'),
                               tf.get_variable('batch_normalization/moving_mean'),
                               tf.get_variable('batch_normalization/moving_variance'), share_tensor]

        self.layer_output = tf.gather(bn, permutation, axis=-1)

    def create_merge(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None, stride=1):
        """
        create a conv layer with biases and without bn
        :param x:
        :param weight_dict:
        :param is_dropout:
        :param is_training:
        :param regularizer_conv:
        :param stride:
        :return:
        """
        self.layer_input = x
        filt, biases = self.get_conv_filter_bn_merge(weight_dict, regularizer_conv)
        conv = tf.nn.conv2d(x, filt, [1, stride, stride, 1], padding='SAME')
        conv = conv + biases

        if is_dropout:
            conv = tf.layers.dropout(conv, noise_shape=[tf.shape(conv)[0], 1, 1, tf.shape(conv)[3]],
                                     training=is_training)
        self.weight_tensors = [filt, biases]
        self.layer_output = conv

    def create_share_merge(self, x, weight_dict=None, is_dropout=False, is_training=False, regularizer_conv=None,
                           stride=1, share_scope=None):
        self.layer_input = x
        weights_specific, biases, permutation, weights_shared = self.get_conv_filter_shared_merge(weight_dict,
                                                                                                  regularizer_conv,
                                                                                                  share_scope)
        weights = tf.concat([weights_shared, weights_specific], axis=-1)

        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
        conv = tf.gather(conv, permutation, axis=-1)
        conv = conv + biases

        share_tensor = tf.constant(True, name='is_share')
        self.weight_tensors = [weights_specific, permutation, biases, share_tensor]

        self.layer_output = conv

    @staticmethod
    def get_conv_filter_bn(weight_dict, regularizer_conv):

        scope_name = tf.get_variable_scope().name

        filt = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                               regularizer=regularizer_conv)
        beta = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/beta'])
        mean = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_mean'])
        variance = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_variance'])

        return filt, beta, mean, variance

    @staticmethod
    def get_conv_filter_shared(weight_dict, regularizer_conv, share_scope):
        scope_name = tf.get_variable_scope().name

        filt = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                               regularizer=regularizer_conv)
        beta = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/beta'])
        mean = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_mean'])
        variance = tf.constant_initializer(weight_dict[scope_name + '/batch_normalization/moving_variance'])
        permutation = tf.get_variable(name='permutation', initializer=weight_dict[scope_name + '/permutation'],
                                      trainable=False)

        with tf.variable_scope(share_scope, reuse=True):
            with tf.variable_scope('/'.join(scope_name.split('/')[1:])):
                share_weights = tf.get_variable(name="weights")

        return filt, beta, mean, variance, permutation, share_weights

    @staticmethod
    def get_conv_filter_bn_merge(weight_dict, regularizer_conv, trainable=True):

        scope_name = tf.get_variable_scope().name
        print(scope_name)
        filt = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                               regularizer=regularizer_conv, trainable=trainable)
        biases = tf.get_variable(name="biases", initializer=weight_dict[scope_name + '/biases'],
                                 regularizer=regularizer_conv, trainable=trainable)

        return filt, biases

    @staticmethod
    def get_conv_filter_shared_merge(weight_dict, regularizer_conv, share_scope):
        scope_name = tf.get_variable_scope().name

        filt = tf.get_variable(name="weights", initializer=weight_dict[scope_name + '/weights'],
                               regularizer=regularizer_conv)

        biases = tf.get_variable(name="biases", initializer=weight_dict[scope_name + '/biases'],
                                 regularizer=regularizer_conv)

        permutation = tf.get_variable(name='permutation', initializer=weight_dict[scope_name + '/permutation'],
                                      trainable=False)

        with tf.variable_scope(share_scope, reuse=True):
            with tf.variable_scope('/'.join(scope_name.split('/')[1:])):
                share_weights = tf.get_variable(name="weights")

        return filt, biases, permutation, share_weights
