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

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.base_model import BaseModel
from layers.conv_layer import ConvLayer
from layers.fc_layer import FullConnectedLayer
from layers.ib_layer import InformationBottleneckLayer
from utils.configer import get_cfg
from datetime import datetime
from utils.logger import *

import tensorflow as tf
import numpy as np

import pickle
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'


# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


class VGGNet(BaseModel):
    def __init__(self, config, musk=False):
        super(VGGNet, self).__init__(config)

        # Used for lobs
        self.is_musked = musk

        self.load_dataset()
        self.n_classes = self.Y.shape[1]

        if self.cfg['path']['path_load'] and os.path.exists(self.cfg['path']['path_load']):
            log_l('Loading weights in %s' % self.cfg['path']['path_load'])
            self.weight_dict = pickle.load(open(self.cfg['path']['path_load'], 'rb'))
            self.initial_weight = False
        else:
            log_l('Initialize weight matrix')
            self.weight_dict = self.construct_initial_weights(
                path_weights_pretrain=config['data']['path_weights_pretrain'])
            self.initial_weight = True

        if self.cfg['basic'][
            'pruning_method'] == 'info_bottle' and 'conv1_1/info_bottle/mu' not in self.weight_dict.keys():
            log_l('Initialize vib params')
            self.weight_dict = dict(self.weight_dict, **self.construct_initial_weights_ib())
        else:
            log_l('Does not initialize vib params, already having or prune_method is None')

    def construct_initial_weights(self, path_weights_pretrain):
        # '/local/home/david/Remote/dataset/vgg_imdb_pretrain_2'
        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        # Weights of conv
        weight_dict = pickle.load(open(path_weights_pretrain, 'rb'))

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

    def set_kl_factor(self, kl_factor):
        log_l('kl_factor: %f' % kl_factor)
        self.kl_factor = kl_factor
        self.loss()
        # self.optimize 在train函数中调用

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
                              ('conv5_1', 2.0 / 1), ('conv5_2', 2.0 / 1), ('conv5_3', 2.0 / 1), 'pooling']:
                if set_layer != 'pooling':
                    conv_name, kl_mult = set_layer
                    if self.cfg['basic']['pruning_method'] == 'info_bottle':
                        kl_mult *= self.cfg['pruning'].getfloat('gamma_conv')

                    with tf.variable_scope(conv_name):
                        conv = ConvLayer(y, weight_dict=self.weight_dict, is_training=self.is_training,
                                         is_musked=self.is_musked,
                                         regularizer_conv=self.regularizer_conv)
                        self.layers.append(conv)
                        y = tf.nn.relu(conv.layer_output)

                        # Pruning of the method 'Information Bottleneck'
                        if self.cfg['basic']['pruning_method'] == 'info_bottle':
                            ib_layer = InformationBottleneckLayer(y, layer_type='C_ib', weight_dict=self.weight_dict,
                                                                  is_training=self.is_training,
                                                                  kl_mult=kl_mult,
                                                                  mask_threshold=self.cfg['pruning'].getfloat(
                                                                      'pruning_threshold'))

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

                    if self.cfg['basic']['pruning_method'] == 'info_bottle':
                        ib_layer = InformationBottleneckLayer(y, layer_type='F_ib', weight_dict=self.weight_dict,
                                                              is_training=self.is_training,
                                                              kl_mult=self.cfg['pruning'].getfloat('gamma_fc'),
                                                              mask_threshold=self.cfg['pruning'].getfloat(
                                                                  'pruning_threshold'))

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
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.op_loss = mae_loss + l2_loss + self.kl_factor * self.kl_total
        else:
            self.op_loss = mae_loss + l2_loss

    def optimize(self, lr):
        # 为了让bn中的\miu, \delta滑动平均
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # self.opt = tf.train.AdamOptimizer(learning_rate=lr)
                self.opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)

                self.op_opt = self.opt.minimize(self.op_loss)

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(self.Y, tf.sign(self.op_logits))
            self.op_accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(self.Y)[1],
                                                                                           tf.float32)

    def build(self, weight_dict=None, share_scope=None):
        if weight_dict:
            self.weight_dict = weight_dict
            self.share_scope = share_scope
        self.inference()
        self.loss()
        self.evaluate()

    def train_one_epoch(self, sess, init, epoch):
        sess.run(init)
        total_loss = 0
        total_kl = 0
        total_correct_preds = 0
        n_batches = 0
        time_last = time.time()
        try:
            while True:
                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    _, loss, accuracy_batch, kl = sess.run([self.op_opt, self.op_loss, self.op_accuracy, self.kl_total],
                                                           feed_dict={self.is_training: True})
                    total_kl += kl * self.kl_factor
                else:
                    _, loss, accuracy_batch = sess.run([self.op_opt, self.op_loss, self.op_accuracy],
                                                       feed_dict={self.is_training: True})

                total_loss += loss
                total_correct_preds += accuracy_batch
                n_batches += 1

                if n_batches % 5 == 0:
                    str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, curr_kl={:f}, train_acc={:%}, used_time:{:.2f}s'.format(
                        epoch + 1,
                        n_batches,
                        self.total_batches_train,
                        total_loss / n_batches,
                        total_kl / n_batches,
                        total_correct_preds / (n_batches * self.cfg['basic'].getint('batch_size')),
                        time.time() - time_last)

                    print('\r' + str_, end=' ')
                    time_last = time.time()
        except tf.errors.OutOfRangeError:
            pass
        print('')
        # 写入log文件
        log(str_, need_print=False)

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
        return weight_dict

    def eval_once(self, sess, init, epoch):
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
        acc = total_correct_preds / self.n_samples_val
        str_ = 'Epoch:{:d}, val_acc={:%}, val_loss={:f}, used_time:{:.2f}s'.format(epoch + 1,
                                                                                   acc,
                                                                                   total_loss / n_batches,
                                                                                   time_end - time_start)

        log(str_)
        return acc

    def train(self, sess, n_epochs, lr):
        # Build optimize graph
        self.optimize(lr)

        sess.run(tf.variables_initializer(self.opt.variables()))

        for epoch in range(n_epochs):
            self.train_one_epoch(sess, self.train_init, epoch)
            acc = self.eval_once(sess, self.test_init, epoch)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                cr = self.get_CR(sess)

            if (epoch + 1) % 10 == 0:
                if self.cfg['basic']['pruning_method'] != 'info_bottle':
                    name = '%s/tr%.2d-epo%.3d-acc%.4f' % (self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc)
                else:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-acc%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, acc)
                self.save_weight(sess, name)

        # Count of training
        self.cnt_train += 1
        # Save into cfg
        name_train = 'train%d' % self.cnt_train
        self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'n_epochs', str(n_epochs))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.cfg.set(name_train, 'cr', str(cr))

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

    def get_CR(self, sess):
        name_len_dict = [72, 72,
                         36, 36,
                         18, 18, 18,
                         9, 9, 9,
                         4, 4, 4]

        # Obtain all masks
        masks = list()
        for layer in self.layers:
            if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib':
                masks += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('pruning_threshold'))]

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

        cr = np.around(float(total_params - pruned_params) / total_params, decimals=5)
        flops = np.around(float(remain_flops) / total_flops, decimals=5)

        str_1 = 'Total parameters: {}, Pruned parameters: {}, Remaining params:{}, Remain/Total params:{}, Each layer pruned: {}'.format(
            total_params, pruned_params, remain_params, cr, prune_state)

        str_2 = 'Total FLOPs: {}, Pruned FLOPs: {}, Remaining FLOPs: {}, Remain/Total FLOPs:{}'.format(total_flops,
                                                                                                       pruned_flops,
                                                                                                       remain_flops,
                                                                                                       flops)

        log(str_1 + '\n' + str_2)

        return cr


def exp(task_names, path_models, pruning, pruning_set, plan_train_normal, plan_train_vib):
    for task_index, task_name in enumerate(task_names):
        # Obtain time stamp
        time_stamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        cfg = get_cfg(task_name, time_stamp, 'net_vgg16.cfg')
        if path_models is not None:
            cfg['path']['path_load'] = str(path_models[task_index])
        else:
            cfg['path']['path_load'] = str(None)

        # Create exp dir
        if not os.path.exists(cfg['path']['path_save']):
            os.mkdir(cfg['path']['path_save'])
            log_t('Create directory %s' % cfg['path']['path_save'])

        # Add pruning cfg into files
        if pruning:
            cfg['basic']['pruning_method'] = 'info_bottle'
            cfg.add_section('pruning')
            for key in pruning_set.keys():
                cfg.set('pruning', key, str(pruning_set[key]))

        log_l('Training on task %s' % task_name)

        # Create model
        gpu_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4)
        gpu_config.gpu_options.allow_growth = True

        tf.reset_default_graph()
        sess = tf.Session(config=gpu_config)

        model = VGGNet(cfg)
        model.build()

        sess.run(tf.global_variables_initializer())

        # Pre test
        log_l('Pre test')
        model.eval_once(sess, model.test_init, -1)

        # Train
        if pruning:
            model.get_CR(sess)
            log_l('')

            for plan in plan_train_vib:
                model.set_kl_factor(plan['kl_factor'])

                for set_ in plan['train']:
                    model.train(sess=sess, n_epochs=set_['n_epochs'], lr=set_['lr'])
                model.save_cfg()
        else:
            log_l('')
            for plan in plan_train_normal:
                model.train(sess=sess, n_epochs=plan['n_epochs'], lr=plan['lr'])
                model.save_cfg()


if __name__ == '__main__':
    # 备选参数设置
    tasks_lfw = ['lfw1', 'lfw2', 'lfw']
    tasks_celeba = ['celeba1', 'celeba2', 'celeba']
    tasks_deepfashion = ['deepfashion1', 'deepfashion2', 'deepfashion']

    plan_train_vib = [
        {'kl_factor': 1e-5,
         'train': [{'n_epochs': 30, 'lr': 0.01}]},
        {'kl_factor': 1e-6,
         'train': [{'n_epochs': 10, 'lr': 0.01}]}
    ]

    plan_train_normal = [{'n_epochs': 20, 'lr': 0.01},
                         {'n_epochs': 20, 'lr': 0.001},
                         {'n_epochs': 20, 'lr': 0.0001}]

    # 执行实验代码
    exp(task_names=['lfw'],
        path_models=None,
        # [
        #     '/local/home/david/Remote/PruneFramework/exp_files/celeba2-2019-07-15 10:07:34/tr00-epo010-acc0.8892',
        #     '/local/home/david/Remote/PruneFramework/exp_files/celeba-2019-07-15 12:11:15/tr00-epo010-acc0.8981',
        # ],
        pruning=False,
        pruning_set={
            'name': 'info_bottle',
            'gamma_conv': 1.,
            'gamma_fc': 15.,
            'pruning_threshold': 0.01
        },
        plan_train_normal=plan_train_normal,
        plan_train_vib=plan_train_vib
        )
