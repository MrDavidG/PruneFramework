# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: resnet_model
@time: 2019-03-28 11:48

Description. 
"""

from models.base_model import BaseModel
from data_loader.image_data_generator import ImageDataGenerator
from utils.config import process_config
# from utils.time_stamp import print_with_time_stamp as print
from layers.conv_layer import ConvLayer
from layers.bn_layer import BatchNormalizeLayer
from layers.fc_layer import FullConnectedLayer
from layers.res_block import ResBlock
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
            print("loading weight matrix")
            self.initial_weight = False

    def init_saver(self):
        pass

    def construct_initial_weights(self):
        weight_dict = dict()
        weight_dict[self.task_name + '/pre_conv/weights'] = np.random.normal(loc=0., scale=np.sqrt(1 / (3 * 3 * 3)),
                                                                             size=[3, 3, 3, self.config.width]).astype(
            np.float32)
        weight_dict[self.task_name + '/pre_conv/batch_normalization/beta'] = np.zeros(self.config.width,
                                                                                      dtype=np.float32)
        weight_dict[self.task_name + '/pre_conv/batch_normalization/moving_mean'] = np.zeros(self.config.width,
                                                                                             dtype=np.float32)
        weight_dict[self.task_name + '/pre_conv/batch_normalization/moving_variance'] = np.ones(self.config.width,
                                                                                                dtype=np.float32)
        for i in range(self.config.n_group):
            for j in range(self.config.n_blocks_per_group):
                block_name = 'conv{:d}_{:d}'.format(i + 1, j + 1)
                if j == 0:
                    input_width = self.config.width * 2 ** i
                else:
                    input_width = self.config.width * 2 ** (i + 1)
                weight_dict[self.task_name + '/' + block_name + '/conv_1/weights'] = np.random.normal(loc=0.,
                                                                                                      scale=np.sqrt(
                                                                                                          1 / (
                                                                                                                  3 * 3 * input_width)),
                                                                                                      size=[3, 3,
                                                                                                            input_width,
                                                                                                            self.config.width * 2 ** (
                                                                                                                    i + 1)]).astype(
                    np.float32)
                weight_dict[self.task_name + '/' + block_name + '/conv_1/batch_normalization/beta'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[self.task_name + '/' + block_name + '/conv_1/batch_normalization/moving_mean'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[
                    self.task_name + '/' + block_name + '/conv_1/batch_normalization/moving_variance'] = np.ones(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)

                weight_dict[self.task_name + '/' + block_name + '/conv_2/weights'] = np.random.normal(loc=0.,
                                                                                                      scale=np.sqrt(
                                                                                                          1 / (
                                                                                                                  3 * 3 * self.config.width * 2 ** (
                                                                                                                  i + 1))),
                                                                                                      size=[3, 3,
                                                                                                            self.config.width * 2 ** (
                                                                                                                    i + 1),
                                                                                                            self.config.width * 2 ** (
                                                                                                                    i + 1)]).astype(
                    np.float32)
                weight_dict[self.task_name + '/' + block_name + '/conv_2/batch_normalization/beta'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[self.task_name + '/' + block_name + '/conv_2/batch_normalization/moving_mean'] = np.zeros(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)
                weight_dict[
                    self.task_name + '/' + block_name + '/conv_2/batch_normalization/moving_variance'] = np.ones(
                    self.config.width * 2 ** (i + 1), dtype=np.float32)

        weight_dict[self.task_name + '/end_bn/batch_normalization/beta'] = np.zeros(self.config.width * 2 ** 3,
                                                                                    dtype=np.float32)
        weight_dict[self.task_name + '/end_bn/batch_normalization/moving_mean'] = np.zeros(self.config.width * 2 ** 3,
                                                                                           dtype=np.float32)
        weight_dict[self.task_name + '/end_bn/batch_normalization/moving_variance'] = np.ones(
            self.config.width * 2 ** 3, dtype=np.float32)

        weight_dict[self.task_name + '/classifier/weights'] = np.random.normal(loc=0., scale=np.sqrt(1 / 256),
                                                                               size=[256, self.n_classes]).astype(
            np.float32)
        weight_dict[self.task_name + '/classifier/biases'] = np.zeros(self.n_classes, dtype=np.float32)
        weight_dict[self.task_name + '/is_merge_bn'] = False

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
            weight = self.weight_dict.pop(self.task_name + '/' + layer_name + '/weights')
            beta = self.weight_dict.pop(self.task_name + '/' + layer_name + '/batch_normalization/beta')
            mean = self.weight_dict.pop(self.task_name + '/' + layer_name + '/batch_normalization/moving_mean')
            variance = self.weight_dict.pop(self.task_name + '/' + layer_name + '/batch_normalization/moving_variance')
            new_weight = weight / np.sqrt(variance)
            new_bias = beta - mean / np.sqrt(variance)
            self.weight_dict[self.task_name + '/' + layer_name + '/weights'] = new_weight
            self.weight_dict[self.task_name + '/' + layer_name + '/biases'] = new_bias

    def meta_val(self, meta_key):
        meta_key_in_weight = self.task_name + '/' + meta_key
        if meta_key_in_weight in self.weight_dict:
            return self.weight_dict[meta_key_in_weight]
        else:
            return self.meta_keys_with_default_val[meta_key]

    def set_meta_val(self, meta_key, meta_val):
        meta_key_in_weight = self.task_name + '/' + meta_key
        self.weight_dict[meta_key_in_weight] = meta_val

    def fetch_weight(self, sess):
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

    def set_global_tensor(self, training_tensor, regu_conv, regu_fc):
        self.is_training = training_tensor
        self.regularizer_conv = regu_conv
        self.regularizer_fc = regu_fc

    def save_weight(self, sess, save_path):
        self.weight_dict = self.fetch_weight(sess)
        file_handler = open(save_path, 'wb')
        pickle.dump(self.weight_dict, file_handler)
        file_handler.close()

    def is_layer_shared(self, layer_name):
        share_key = self.task_name + '/' + layer_name + '/is_share'
        if share_key in self.weight_dict:
            return self.weight_dict[share_key]
        return False

    def get_layer_id(self, layer_name):
        i = 0
        for layer in self.layers:
            if layer.layer_name == layer_name:
                return i
            i += 1
        print("layer not found!")
        return -1

    def inference(self):
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('pre_conv'):
                pre_conv_layer = ConvLayer(self.X, self.weight_dict, self.config.dropout, self.is_training,
                                           self.regularizer_conv, is_shared=self.is_layer_shared('pre_conv'),
                                           share_scope=self.share_scope, is_merge_bn=self.meta_val('is_merge_bn'))
                self.layers.append(pre_conv_layer)
                y = pre_conv_layer.layer_output

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
                        self.layers.append(res_block.layers[0])
                        self.layers.append(res_block.layers[1])
                        y = res_block.layer_output
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

    # load dataset
    def load_dataset(self):
        dataset_train, dataset_val, dataset_hessian, self.total_batches_train, self.n_samples_train, self.n_samples_val = \
            ImageDataGenerator.load_dataset(self.config.batch_size, self.config.cpu_cores, self.task_name,
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

    def hessian_conv(self, sess, layer_index):
        layer = self.layers[layer_index]
        a = tf.expand_dims(layer.layer_input, axis=-1)
        b = tf.expand_dims(layer.layer_input, axis=3)
        outprod = tf.reduce_sum(tf.reduce_mean(tf.multiply(a, b), axis=[1, 2]), axis=[0])

        batch_count = 0
        print('start calculating hessian of ' + self.task_name)
        sess.run(self.hessian_init)
        hessian_sum = 0
        try:
            while True:
                if batch_count == 0:
                    hessian_sum = sess.run(outprod, feed_dict={self.is_training: False})
                else:
                    hessian_sum += sess.run(outprod, feed_dict={self.is_training: False})
                batch_count += 1
        except tf.errors.OutOfRangeError:
            pass

        hessian = hessian_sum / self.n_samples_train
        return hessian

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
        #        start_time = time.time()
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

    config = process_config("../configs/res_net.json")
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
        resnet = ResNet(config, task_name)
        resnet.set_global_tensor(training, regularizer_conv, regularizer_fc)
        resnet.build()
        session.run(tf.global_variables_initializer())
        resnet.train(sess=session, n_epochs=80, lr=0.1)

        resnet.train(sess=session, n_epochs=20, lr=0.01)

        resnet.train(sess=session, n_epochs=20, lr=0.001)

        # save the model weights
        if not os.path.exists('model_weights'):
            os.mkdir('model_weights')
        resnet.save_weight(session, 'model_weights/' + task_name)

    # Step2: Analyze & Step3: Prune
    resnet.prune()

    # Step4: Retrain
