# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: model_toy
@time: 2019/12/6 10:29 下午

Description. 
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    gamma = 0.1

    data = tf.placeholder(dtype=tf.float32)
    labels = tf.placeholder(dtype=tf.float32)
    y = data
    theta_list_tf = list()
    for name in ["c1_1", "c1_2",
                 "c2_1", "c2_2",
                 "c3_1", "c3_2", "c3_3",
                 "c4_1", "c4_2", "c4_3",
                 "c5_1", "c5_2", "c5_3"]:
        y = tf.nn.relu(tf.layers.conv2d(y, 512, 3, name=name))
        theta = tf.get_variable(name=name + '_theta', initializer=512, trainable=True)
        theta_list_tf.append(theta)
        y = y * 1. / (1 + tf.exp(20 * (tf.constant(512) - theta)))

    y = tf.layers.flatten(y)
    y = tf.layers.dense(y, 1000)

    # loss
    loss = tf.losses.softmax_cross_entropy(labels, y)
    loss_flops = 0
    for theta in theta_list_tf:
        loss_flops += theta
    loss = loss + gamma * loss_flops

    # opt
    op_train = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)

    # train
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    s = tf.Session()

    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        _, loss, theta_list = s.run([op_train, loss] + [theta_list_tf], {data: x_batch, labels: y_batch})
        print('%d: loss=%f, theta=%s' % (i, loss, str(theta_list)))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    # 计算准确率，tf.cast将True和False变成浮点型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 将test数据feed进行测试
    print('acc: ', s.run(accuracy, {data: mnist.test.images, labels: mnist.test.labels}))
    print('')
