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
from data_loader.image_data_generator import ImageDataGenerator

import pickle


class BaseModel:
    def __init__(self, config):
        self.is_training = None
        self.regularizer_conv = None
        self.regularizer_fc = None

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
        self.total_batches_train = None
        self.n_samples_train = None
        self.n_samples_val = None
        self.share_scope = None

        self.layers = list()

        self.prune_method = config.prune_method
        if self.prune_method == 'info_bottle':
            self.kl_factor = config.prune_kl_factor
            self.prune_threshold = config.prune_threshold
            self.kl_total = None

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

    def load_dataset(self):
        dataset_train, dataset_val, self.total_batches_train, self.n_samples_train, self.n_samples_val = ImageDataGenerator.load_dataset(
            self.config.batch_size, self.config.cpu_cores, self.task_name, self.imgs_path)
        self.train_init, self.test_init, self.X, self.Y = ImageDataGenerator.dataset_iterator(
            dataset_train,
            dataset_val)

    def save_weight(self, sess, save_path):
        self.weight_dict = self.fetch_weight(sess)
        file_handler = open(save_path, 'wb')
        pickle.dump(self.weight_dict, file_handler)
        file_handler.close()

    def get_layer_id(self, layer_name):
        i = 0
        for layer in self.layers:
            if layer.layer_name == layer_name:
                return i
            i += 1
        print("layer not found!")
        return -1

    @abstractmethod
    def init_saver(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def predict(self, labels, logits):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def optimize(self):
        pass
