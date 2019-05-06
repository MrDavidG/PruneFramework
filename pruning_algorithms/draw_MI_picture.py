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

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))  # same code
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mi_with_input_label(sess, task_name, model, bin, num, total_batch=1):
    layer_name_list = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']
    # bin_list = [0.01, 0.01,
    #             0.01, 0.01,
    #             0.0001, 0.0001, 0.0001,
    #             0.0001, 0.0001, 0.0001,
    #             0.01, 0.01, 0.01,
    #             0.01, 0.01, 0.00002]

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

            # 把最后一层的输出当做labels
            softmax_last_layer_output = [softmax(layers_output[-1])]

            # labels = softmax_last_layer_output[0]

            # 获得relu之后的结果
            layers_output = [x * (x > 0) for x in layers_output[:-1]] + softmax_last_layer_output

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
                                bins(layer_neuron_output, bin, num),
                                labels[:, k])
                        else:
                            rele_matrix_label[j][k] = get_mutual_information(bins(layer_output[:, j], bin, num),
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
                            rele_matrix_input[j][k] = get_mutual_information(bins(layer_neuron_output, bin, num),
                                                                             bins(input_neuron, bin, num))
                        else:
                            rele_matrix_input[j][k] = get_mutual_information(bins(layer_output[:, j], bin, num),
                                                                             bins(input_neuron, bin, num))

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


if __name__ == '__main__':
    # Obtain a pre-train model
    config = process_config("../configs/rb_vgg.json")
    # config = process_config("../configs/res_net.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto(intra_op_parallelism_threads=4)
    gpu_config.gpu_options.allow_growth = True

    for task_name in ['mnist']:
        tf.reset_default_graph()

        session = tf.Session(config=gpu_config)
        training = tf.placeholder(dtype=tf.bool, name='training')
        regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.005)
        regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.005)

        # model = ResNet(config, task_name, model_path='/local/home/david/Remote/models/model_weights/res_gtsrb_0.9994')
        # model = VGGNet(config, task_name, model_path='/local/home/david/Remote/models/model_weights/vgg_cifar10_0.7919')
        model = DenseNet(config, task_name,
                         model_path='/local/home/david/Remote/models/model_weights/dense_mnist_0.9737')

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        get_relu_max(session, task_name, model)

        # 获得MI
        rele_list_input, rele_list_label = get_mi_with_input_label(session, task_name, model, bin=0.2741, num=100,
                                                                   total_batch=1)

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
                        marker=marker_list[i % len(marker_list)])
            plt.annotate(str(i + 1), (rele_list_input[i], rele_list_label[i]))
        plt.xlabel('MI with X')
        plt.ylabel('MI with Y')
        plt.savefig('./mi_%s.jpeg' % (datetime.now()))
        print(rele_list_input, rele_list_label)
