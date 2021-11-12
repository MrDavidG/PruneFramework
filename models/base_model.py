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
from data_loader.image_data_generator import ImageDataGenerator
from utils.json import read_f
from utils.json import read_l

import pickle


class BaseModel:
    def __init__(self, config):
        self.op_loss = None
        self.op_loss_func = None
        self.op_loss_regu = None
        self.op_loss_kl = None
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
        self.learning_rate = None

        self.cnt_train = 0
        self.layers = list()
        self.task_name = config['task']['name']

        # Model
        self.structure = read_l(config, 'model', 'structure')
        self.dimension = read_l(config, 'model', 'dimension')
        self.activation = read_l(config, 'model', 'activation')
        if config.has_option('model', 'kernel_size'):
            self.kernel_size = read_l(config, 'model', 'kernel_size')
        else:
            self.kernel_size = [3, 3]
        if config.has_option('model', 'stride'):
            self.stride = read_l(config, 'model', 'stride')

        # VIBNet
        self.kl_total = 0
        self.pruning = False
        if config['basic']['pruning_method'] == 'info_bottle':
            self.pruning = True
            self.kl_mult = read_l(config, 'pruning', 'kl_mult')
            self.kl_factor = 0
            self.threshold = read_f(config, 'pruning', 'pruning_threshold')

        self.is_training = tf.placeholder(dtype=tf.bool)
        self.set_global_tensor(config['train'].getfloat('regularizer_conv'), config['train'].getfloat('regularizer_fc'))

        self.cfg = config

    def load_dataset(self):
        dataset_train, dataset_val, self.total_batches_train, self.n_samples_train, self.n_samples_val = ImageDataGenerator.load_dataset(
            self.cfg)
        self.train_init, self.test_init, self.X, self.Y = ImageDataGenerator.dataset_iterator(dataset_train,
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

    def get_layer_by_name(self, layer_name):
        for layer in self.layers:
            if layer.layer_name == layer_name:
                return layer
        return None

    # def is_layer_shared(self, layer_name):
    #     share_key = layer_name + '/is_share'
    #     if share_key in self.weight_dict:
    #         return self.weight_dict[share_key]
    #     return False

    def set_global_tensor(self, regu_conv, regu_fc):
        self.regularizer_conv = tf.contrib.layers.l2_regularizer(scale=regu_conv)
        self.regularizer_fc = tf.contrib.layers.l2_regularizer(scale=regu_fc)

    def save_cfg(self):
        with open(self.cfg['path']['path_cfg'], 'w') as file:
            self.cfg.write(file)

    def save_now(self, epoch, n_epoch, save_step):
        if save_step == -2:
            return False
        else:
            return save_step == -1 and epoch == n_epoch or save_step != -1 and epoch % save_step == 0

    @abstractmethod
    def init_saver(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def optimize(self):
        pass
