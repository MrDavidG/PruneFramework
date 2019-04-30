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
from datetime import datetime

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))  # same code
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mi_with_input_label(sess, task_name, model, bin=0.2, num=1000, total_batch=1):
    layer_name_list = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']
    bin_list = [0.01, 0.01,
                0.01, 0.01,
                0.0001, 0.0001, 0.0001,
                0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.01,
                0.01, 0.01, 0.00002]

    rele_mean_list_label = list()
    rele_mean_list_input = list()

    # 所有层的输出
    layers_output_tf = [layer.layer_output for layer in model.layers]
    sess.run(model.train_init)
    n_batch = 1
    try:
        while True:
            # labels: [batch_size, dim]
            layers_output, input_x, labels = sess.run([layers_output_tf] + [model.X, model.Y],
                                                      feed_dict={model.is_training: False})

            n_classes = np.shape(labels)[1]

            print('[%s] Batch %d:' % (datetime.now(), n_batch))

            # Mutual information between input x and label
            # input_x: [batch_size, h, w, channel_size]
            # layer_output: [batch_size, h, w, channel_size] / [batch_size, dim]
            # 获得relu之后的结果
            layers_output = [x * (x > 0) for x in layers_output[:-1]]
            # 最后一层为softmax
            layers_output += [softmax(layers_output[-1])]
            for i, layer_output in enumerate(layers_output):
                print('[%s] Compute MI for layer %s' % (datetime.now(), layer_name_list[i]))

                dim_layer, dim_label, dim_input = layer_output.shape[-1], n_classes, input_x.shape[-1]

                rele_matrix_label = np.zeros(shape=(dim_layer, dim_label))
                # the j-th neuron in i-th layer
                for j in range(dim_layer):
                    # the k-th dimension of label
                    for k in range(dim_label):
                        # layer: [batch_size, x, y] / [batch_size]
                        # label: [batch_size]
                        if len(layer_output.shape) > 2:
                            layer_neuron_output = np.average(np.average(layer_output[:, :, :, j], axis=1), axis=1)
                            rele_matrix_label[j][k] = get_mutual_information(
                                bins(layer_neuron_output, bin_list[i], num),
                                labels[:, k])
                        else:
                            rele_matrix_label[j][k] = get_mutual_information(bins(layer_output[:, j], bin_list[i], num),
                                                                             labels[:, k])

                # 在这里可以记录一下每一层的平均的relevance
                rele_mean_with_label = np.mean(rele_matrix_label)
                print('[%s] Average MI with label: %f' % (datetime.now(), rele_mean_with_label))

                # input x
                rele_matrix_input = np.zeros(shape=(dim_layer, dim_input))
                # the j-th neuron in i-th layer
                for j in range(dim_layer):
                    # the k-th dimension of input
                    for k in range(dim_input):
                        # layer: [batch_size, x, y] / [batch_size]
                        # input: [batch_size, h, w, channel_size]
                        input_neuron = np.average(np.average(input_x[:, :, :, k], axis=1), axis=1)
                        if len(layer_output.shape) > 2:
                            layer_neuron_output = np.average(np.average(layer_output[:, :, :, j], axis=1), axis=1)
                            rele_matrix_input[j][k] = get_mutual_information(bins(layer_neuron_output, bin_list[i]),
                                                                             bins(input_neuron, bin_list[i]))
                        else:
                            rele_matrix_input[j][k] = get_mutual_information(bins(layer_output[:, j], bin_list[i]),
                                                                             bins(input_neuron, bin_list[i]))

                # 在这里可以记录一下每一层的平均的relevance
                rele_mean_with_input = np.mean(rele_matrix_input)
                print('[%s] Average MI with input: %f' % (datetime.now(), rele_mean_with_input))

                if n_batch == 1:
                    rele_mean_list_input += [rele_mean_with_input]
                    rele_mean_list_label += [rele_mean_with_label]
                else:
                    rele_mean_list_input[i] = rele_mean_list_input[i] + (
                            rele_mean_with_input - rele_mean_list_input[i]) / n_batch
                    rele_mean_list_label[i] = rele_mean_list_label[i] + (
                            rele_mean_with_label - rele_mean_list_label[i]) / n_batch

            n_batch += 1

            if n_batch > total_batch:
                break
    except tf.errors.OutOfRangeError:
        pass

    return rele_mean_list_input, rele_mean_list_label


if __name__ == '__main__':
    # Obtain a pre-train model
    config = process_config("../configs/rb_vgg.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto(intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['cifar10']:
        tf.reset_default_graph()

        session = tf.Session(config=gpu_config)
        training = tf.placeholder(dtype=tf.bool, name='training')
        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.005)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.005)

        model = VGGNet(config, task_name,
                       model_path='/local/home/david/Remote/models/model_weights/vgg_cifar10_0.7919')
        # model_path='/local/home/david/Remote/models/model_weights/vgg_cifar100_0.6582')

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        # 获得MI
        rele_list_input, rele_list_label = get_mi_with_input_label(session, task_name, model, bin=0.02, num=1000,
                                                                   total_batch=1)

        # 画图
        layer_name_list = ['c1_1', 'c1_2',
                           'c2_1', 'c2_2',
                           'c3_1', 'c3_2', 'c3_3',
                           'c4_1', 'c4_2', 'c4_3',
                           'c5_1', 'c5_2', 'c5_3',
                           'f6', 'f7', 'f8']
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

        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(len(rele_list_input)):
            plt.scatter(rele_list_input[i], rele_list_label[i], alpha=0.6,  # label=layer_name_list[i],
                        marker=marker_list[i])
            plt.annotate(layer_name_list[i], (rele_list_input[i], rele_list_label[i]))
        plt.xlabel('MI with X')
        plt.ylabel('MI with Y')
        plt.savefig('./mi33.jpeg')
        print(rele_list_input, rele_list_label)
