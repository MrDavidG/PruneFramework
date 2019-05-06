# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: relevancy_pruning
@time: 2019-04-24 16:29

Description. 
"""
import sys

sys.path.append(r"/local/home/david/Remote/")
from utils.config import process_config
from models.vgg_cifar100 import VGGNet
from models.resnet_model import ResNet
from datetime import datetime

import numpy as np
import math
import pickle

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_mutual_information(x, y, log_base=2):
    """
    Calculate and return Mutual information between two random variables
    """
    x = np.array(x)
    y = np.array(y)
    # Variable to return MI
    summation = 0.0
    # Number of columns
    n_cols = np.shape(x)[0]

    # Get uniques values of random variables
    values_x = set(x)
    values_y = set(y)

    # 两个数组中的每个元素
    for value_x in values_x:
        for value_y in values_y:
            # data[x_index]==value_x
            # data[x_index]有多少个元素等于value_x的比例
            px = np.shape(np.where(x == value_x))[1] / n_cols
            py = np.shape(np.where(y == value_y))[1] / n_cols

            # where: 值等于value_x的元素在数组内的位置
            # 两者元素中value_x和value_y如果等价的话，看它们两者的位置
            pxy = len(np.where(np.in1d(np.where(x == value_x)[0], np.where(y == value_y)[0]) == True)[0]) / n_cols
            if pxy > 0.0:
                summation += pxy * math.log((pxy / (px * py)), log_base)
    return summation


def bins(array, bin=None, num=None):
    # return np.floor(array / bin) * bin
    if bin != None:
        return np.floor(array / bin) * bin
    elif num != None:
        max = np.max(array)
        min = np.min(array)
        width = (max - min) * 1. / num
        if width == 0:
            return array
        else:
            return np.floor((array - min) / width)


def neuron_relevancy_pruning(sess, task_name, model, threshold_MI, name_save='rb_model', total_batch=1, bin=0.2,
                             num=100, regular_conv=0.01, regular_fc=0.01):
    dim_list = [3,
                64, 64,
                128, 128,
                256, 256, 256,
                512, 512, 512,
                512, 512, 512,
                4096, 4096, 100]

    layer_name_list = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3',
                       'fc6', 'fc7', 'fc8']

    rele_matrix_bool_list = list()

    layers_output_tf = [layer.layer_output for layer in model.layers[:-1]]
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
            layers_output = [x * (x > 0) for x in layers_output]
            for i, layer_output in enumerate([input_x] + layers_output):
                # input
                if i == 0:
                    print('[%s] Compute MI for input' % (datetime.now()))
                else:
                    print('[%s] Compute MI for layer %s' % (datetime.now(), layer_name_list[i - 1]))

                dim_layer, dim_label = layer_output.shape[-1], n_classes

                rele_matrix = np.zeros(shape=(dim_layer, dim_label))
                # the j-th neuron in i-th layer
                for j in range(dim_layer):
                    # the k-th dimension of label
                    for k in range(dim_label):
                        # layer: [batch_size, h, w] / [batch_size]
                        # label: [batch_size]
                        if len(layer_output.shape) > 2:
                            # 对[h, w]求平均
                            layer_neuron_output = np.average(np.average(layer_output[:, :, :, j], axis=1), axis=1)
                            rele_matrix[j][k] = get_mutual_information(bins(layer_neuron_output, bin, num),
                                                                       labels[:, k])
                        else:
                            rele_matrix[j][k] = get_mutual_information(bins(layer_output[:, j], bin, num), labels[:, k])

                # 在这里可以记录一下每一层的平均的relevance
                rele_layer_mean = np.mean(rele_matrix)
                print('[%s] Average MI: %f' % (datetime.now(), rele_layer_mean))

                # MI的阈值
                rele_matrix_bool = np.float32(rele_matrix > threshold_MI)

                # 获得MI的平均值
                if n_batch == 1:
                    rele_matrix_bool_list += [rele_matrix_bool]
                else:
                    rele_matrix_bool_list[i] = rele_matrix_bool_list[i] + (
                            rele_matrix_bool - rele_matrix_bool_list[i]) / n_batch
            n_batch += 1

            if n_batch > total_batch:
                break
    except tf.errors.OutOfRangeError:
        pass

    # 得到的rele_matrix_bool_list是从input_x到最后fc7与label之间的MI关系

    # Obtain the weights
    weight_dict = model.weight_dict

    musk_dict = dict()

    # Obtain musks
    print('[%s] Musk the weights' % (datetime.now()))
    total_pruned = 0
    total_preserve = 0
    total_params = 0
    # 这个musk需要两层相乘才能得到，最后一层的fc8不参与剪枝
    for i, layer in enumerate(model.layers[:-1]):
        print('[%s] Musk the layer %s' % (datetime.now(), layer_name_list[i]))
        layer_name = layer_name_list[i]

        # 第一层不参与剪枝
        # if i == 0:
        #     continue

        musk = np.float32(np.matmul(rele_matrix_bool_list[i], np.transpose(rele_matrix_bool_list[i + 1])) > 0)
        # 连接conv和fc层需要做特殊处理
        if layer.layer_type == 'F' and model.layers[i - 1].layer_type == 'C':
            dim_flatten = np.shape(weight_dict[layer_name + '/weights'])[0]
            dim_fliter = dim_list[i - 1]
            times_repeat = dim_flatten / dim_fliter
            musk = np.repeat(musk, times_repeat, axis=0)

        # Store musk
        musk_dict[layer_name + '/musk'] = musk

        weight_dict[layer_name + '/weights'] *= musk

        shape_weight = np.shape(weight_dict[layer_name + '/weights'])

        n_params = np.prod(shape_weight)
        if len(shape_weight) == 4:
            n_preserve = np.prod(shape_weight[:2]) * np.sum(musk)
        else:
            n_preserve = np.sum(musk)
        n_pruned = n_params - n_preserve

        total_params += n_params
        total_preserve += n_preserve
        total_pruned += n_pruned

    # 得到的musk是从输入层的(batch_size, h, w, 3)到最后fc7的musk

    # cr
    cr = np.around(total_preserve * 1. / total_params, decimals=4)
    print('[%s]: Total params: %d, pruned params: %d, preserved params: %d, cr: %f' % (datetime.now(), total_params,
                                                                                       total_pruned, total_preserve,
                                                                                       cr))

    # accu
    tf.reset_default_graph()

    session = tf.Session(config=gpu_config)
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.)

    model_ = VGGNet(config, task_name)
    model_.weight_dict = weight_dict
    model_.set_global_tensor(training, regularizer_conv, regularizer_fc)
    model_.build()
    session.run(tf.global_variables_initializer())

    acc = model_.eval_once(session, model_.test_init, 0)

    # Save weights
    weight_and_musk_dict = dict(weight_dict, **musk_dict)
    path_before_retrain = './pruning_weights/' + name_save + '_BR_acc-' + str(acc) + '_cr-' + str(cr)
    pickle.dump(weight_and_musk_dict,
                open(path_before_retrain, 'wb'))

    # Retrain the model
    print('\n[%s]: Retrain the model' % (datetime.now()))

    config_retrain = process_config("../configs/vgg_net.json")

    # Retrain
    tf.reset_default_graph()
    session = tf.Session(config=gpu_config)
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=regular_conv)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=regular_fc)

    model_ = VGGNet(config_retrain, task_name, musk=True, model_path=path_before_retrain)
    model_.set_global_tensor(training, regularizer_conv, regularizer_fc)
    model_.build()
    session.run(tf.global_variables_initializer())

    model_.eval_once(session, model_.test_init, epoch=-1)

    model_.train(session, n_epochs=10, lr=0.01)
    model_.train(session, n_epochs=30, lr=0.001)
    model_.train(session, n_epochs=20, lr=0.0001)

    acc_final = model_.eval_once(session, model_.test_init, epoch=-1)

    weight_dict_after_retrain = model_.fetch_weight(session)
    weight_and_musk_dict_after_retrain = dict(weight_dict_after_retrain, **musk_dict)
    pickle.dump(weight_and_musk_dict_after_retrain,
                open('./pruning_weights/' + name_save + '_AR_' + str(acc_final) + '_' + str(cr), 'wb'))


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

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        # prune model
        neuron_relevancy_pruning(session, task_name, model, threshold_MI=0.02, name_save='rb_vgg_cifar10',
                                 total_batch=1, bin=0.01, num=100, regular_conv=0.0, regular_fc=0.0)
