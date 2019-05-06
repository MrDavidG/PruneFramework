# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: draw_MI_picture
@time: 2019-04-29 10:25

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/")
from pruning_algorithms.relevancy_pruning import bins, get_mutual_information
from utils.config import process_config
from models.vgg_cifar100 import VGGNet
from models.resnet_model import ResNet
from models.dense_model import DenseNet
from datetime import datetime
from utils.mutual_information import *

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_relu_max(sess, task_name, model):
    # 记录所有layer中激活后的最大值
    max_all_layers = 0
    min_all_layers = 100
    min_all_input = 100
    max_all_input = 0
    # 所有层的输出
    layers_output_tf = [layer.layer_output for layer in model.layers[:-1]]
    sess.run(model.train_init)

    n_batch = 1
    try:
        while True:
            # labels: [batch_size, dim]
            layers_output, input_x = sess.run([layers_output_tf, model.X],
                                              feed_dict={model.is_training: False})
            # 这一层中最大的
            # [batch_size, h, w, c]
            # [batch_size, dim]
            max_layer = 0
            min_layer = 100
            for layer_output in layers_output:
                max_layer = max(np.max(layer_output), max_layer)
                min_layer = min(np.min(layer_output), min_layer)
                print(max_layer, min_layer)

            if max_all_layers < max_layer:
                max_all_layers = max_layer

            if min_all_layers > min_layer:
                min_all_layers = min_layer

            print('[%s] Batch %d, Max Output: %f, Min Output: %f' % (datetime.now(), n_batch, max_layer, min_layer))

            # input_x
            max_input = np.max(input_x)
            min_input = np.min(input_x)

            print('     Max Input: %f, Min Input: %f' % (max_input, min_input))

            max_all_input = max(max_input, max_all_input)
            min_all_input = min(min_input, min_all_input)

            n_batch += 1

    except tf.errors.OutOfRangeError:
        pass

    print('[%s] The final Max Output: %f' % (datetime.now(), max_all_layers))
    print('[%s] The final Min Output: %f' % (datetime.now(), min_all_layers))
    print('[%s] The final Max Input: %f' % (datetime.now(), max_all_input))
    print('[%s] The final Min Input: %f' % (datetime.now(), min_all_input))



def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))  # same code
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mi_with_input_label(sess, task_name, model, method, binsize, num, total_batch=1):
    nats2bits = 1.0 / np.log(2)

    layer_name_list = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']

    mi_mean_list_label = list()
    mi_mean_list_input = list()

    # 所有层的输出
    layers_output_tf = [layer.layer_output for layer in model.layers[:-1]]
    sess.run(model.test_init)
    n_batch = 1
    try:
        while True:
            # labels: [batch_size, dim]
            layers_output, input_x, labels_one_hot = sess.run([layers_output_tf] + [model.X, model.Y],
                                                              feed_dict={model.is_training: False})

            print('[%s] Batch %d:' % (datetime.now(), n_batch))

            # Relu
            layers_output = [x * (x > 0) for x in layers_output]

            for i, layer_output in enumerate(layers_output):
                print('[%s] Compute MI for layer %d' % (datetime.now(), i + 1))

                # [batch_size, h, w, c] -> [batch_size, c]
                # Hidden layer vector
                if len(layer_output.shape) > 2:
                    # [batch_size, c]
                    hidden = np.average(np.average(layer_output, axis=1), axis=1)
                else:
                    hidden = layer_output

                # MI
                labels = np.argmax(labels_one_hot, axis=1)
                # 符合这个label的sample为true
                labelixs = {}
                for j in range(10):
                    labelixs[j] = labels == j

                if method == 'bin':
                    mi_with_input, mi_with_label = bin_mi(hidden, labelixs, binsize=binsize)
                elif method == 'kde':
                    labelprobs = np.mean(labels_one_hot, axis=0)
                    mi_with_input, mi_with_label = kde_mi(hidden, labelixs, labelprobs)

                mi_with_input, mi_with_label = nats2bits * mi_with_input, nats2bits * mi_with_label
                print('[%s] Average MI with input: %f' % (datetime.now(), mi_with_input))
                print('[%s] Average MI with label: %f' % (datetime.now(), mi_with_label))

                if n_batch == 1:
                    mi_mean_list_input += [mi_with_input]
                    mi_mean_list_label += [mi_with_label]
                else:
                    mi_mean_list_input[i] = mi_mean_list_input[i] + (mi_with_input - mi_mean_list_input[i]) / n_batch
                    mi_mean_list_label[i] = mi_mean_list_label[i] + (mi_with_label - mi_mean_list_label[i]) / n_batch

            n_batch += 1

            if n_batch > total_batch:
                break
    except tf.errors.OutOfRangeError:
        pass

    return mi_mean_list_input, mi_mean_list_label


def exp(exp_config):
    # Extract experiment config
    config_name = exp_config['config']
    task_name = exp_config['task_name']

    model_type = exp_config['model_type']
    model_path = exp_config['model_path']

    total_batch = exp_config['total_batch']
    binsize = exp_config['binsize']

    method = exp_config['method']

    # Obtain a pre-train model
    config = process_config("../configs/" + config_name)

    gpu_config = tf.ConfigProto(intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    session = tf.Session(config=gpu_config)
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.00)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.00)

    if model_type == 'Vgg':
        model = VGGNet(config, task_name, model_path='/local/home/david/Remote/models/model_weights/' + model_path)
    elif model_type == 'Dense':
        model = DenseNet(config, task_name, model_path='/local/home/david/Remote/models/model_weights/' + model_path)
    elif model_type == 'Res':
        model = ResNet(config, task_name, model_path='/local/home/david/Remote/models/model_weights/' + model_path)

    model.set_global_tensor(training, regularizer_conv, regularizer_fc)
    model.build()

    session.run(tf.global_variables_initializer())

    # get_relu_max(session, task_name, model)

    # 获得MI
    rele_list_input, rele_list_label = get_mi_with_input_label(session, task_name, model, method=method,
                                                               binsize=binsize, num=100, total_batch=total_batch)

    marker_list = ['.',
                   ',',
                   's',
                   '*',
                   'h',
                   '+',
                   'x',
                   '_',
                   'o',
                   'v',
                   '^',
                   '<',
                   '>',
                   'D',
                   'd',
                   'p']

    for i in range(len(rele_list_input)):
        plt.scatter(rele_list_input[i], rele_list_label[i], alpha=0.6,  # label=layer_name_list[i],
                    marker=marker_list[i])
        plt.annotate(str(i + 1), (rele_list_input[i], rele_list_label[i]))

    plt.xlabel('MI with X')
    plt.ylabel('MI with Y')
    plt.savefig('./mi_%s.jpeg' % (datetime.now()))
    print(rele_list_input)
    print(rele_list_label)


if __name__ == '__main__':
    exp_config = {
        '0': {'model_type': 'Vgg', 'task_name': 'cifar10', 'config': 'rb_vgg.json', 'method': 'bin',
              'model_path': 'vgg_sgd_cifar10_0.7418', 'total_batch': 50, 'binsize': 0.4},

        '1': {'model_type': 'Dense', 'task_name': 'mnist', 'config': 'dense_net.json', 'method': 'bin',
              'model_path': 'dense_mnist_0.9844', 'total_batch': 100, 'binsize': 0.5},

        '2': {'model_type': 'Res', 'task_name': 'gtsrb', 'config': 'rb_res.json', 'method': 'kde',
              'model_path': 'res_gtsrb_0.9994', 'total_batch': 100, 'binsize': 0.5}
    }

    exp(exp_config['0'])
