# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: base_model.py
@time: 2019-03-27 10:38

The base dnn model template.
"""

import tensorflow as tf
from abc import abstractmethod
from utils.time_stamp import print_with_time_stamp as print


class BaseModel:
    def __init__(self, config):
        # config of the model
        self.config = config
        # init the global step
        self.init_global_step()

    def save(self, sess):
        print('Saving model...')
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print('Model saved')

    # load model from the checkpoint
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print('Loading model checkpoint {} ...\n'.format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print('Model loader')

    # init a tensoflow variable as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # init a tensorflow variable as epoch counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    @abstractmethod
    def init_saver(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def predicte(self, labels, logits):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def retrain(self):
        pass
