# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: lobs_pruning
@time: 2019-04-08 16:28

Description. 
"""

from models.vgg_model import VGGNet
from models.resnet_model import ResNet
from utils.time_stamp import print_with_time_stamp as print_
from utils.config import process_config

from numpy.linalg import inv, pinv, LinAlgError
import tensorflow as tf
import numpy as np
from datetime import datetime
import os


# Construct hessian computing graph for res layer (conv layer without bias)
def create_res_hessian_computing_tf_graph(input_shape, layer_kernel, layer_stride):
    """
    This function create the TensorFlow graph for computing hessian matrix for res layer.
    Step 1: It first extract image patches using tf.extract_image_patches.
    Step 2: Then calculate the hessian matrix by outer product.

    Args:
        input_shape: the dimension of input
        layer_kernel: kernel size of the layer
        layer_stride: stride of the layer
    Output:
        input_holder: TensorFlow placeholder for layer input
        get_hessian_op: A TensorFlow operator to calculate hessian matrix

    """
    input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
    patches = tf.extract_image_patches(images=input_holder,
                                       ksizes=[1, layer_kernel, layer_kernel, 1],
                                       strides=[1, layer_stride, layer_stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='SAME')

    a = tf.expand_dims(patches, axis=-1)
    b = tf.expand_dims(patches, axis=3)
    outprod = tf.multiply(a, b)

    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])

    return input_holder, get_hessian_op


def create_fc_hessian_computing_tf_graph(layer_input):
    """
    This function create the TensorFlow graph for computing hessian matrix for fully-connected layer.
    Compared with create_res_hessian_computing_tf_graph, it does not need to extract patches.
    :param input_shape:
    :return:
    """
    a = tf.expand_dims(layer_input, axis=-1)
    # Appending extra one for bias term
    vect_w_b = tf.concat([a, tf.ones([tf.shape(a)[0], 1, 1])], axis=1)
    outprod = tf.matmul(vect_w_b, vect_w_b, transpose_b=True)
    get_hessian_op = tf.reduce_mean(outprod, axis=0)
    return get_hessian_op


def create_conv_hessian_computing_tf_graph(layer_input, kernel_size=3, layer_stride=1):
    """
    This function create the TensorFlow graph for computing hessian matrix for convolutional layer.
    Compared with create_res_hessian_computing_tf_graph, it append extract one for bias term.
    :param input_shape:
    :param kernel_size:
    :param layer_stride:
    :return:
    """
    patches = tf.extract_image_patches(images=layer_input,
                                       ksizes=[1, kernel_size, kernel_size, 1],
                                       strides=[1, layer_stride, layer_stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='SAME')
    vect_w_b = tf.concat([patches, tf.ones([tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], 1])],
                         axis=3)
    a = tf.expand_dims(vect_w_b, axis=-1)
    b = tf.expand_dims(vect_w_b, axis=3)
    outprod = tf.multiply(a, b)
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    return get_hessian_op


def unfold_kernel(filt):
    """
    原函数的输入是[o,i,h,w]，但是在tensorflow里面参数的形式是[h,w,i,o],需要转化后才能用这个函数

    In pytorch format, kernel is stored as [out_channel, in_channel, height, width]
    Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
    :param kernel: numpy ndarray
    :return:
    """
    kernel = np.transpose(filt, (3, 2, 0, 1))
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])
    return weight


def generate_hessian_inv(sess, model, layer, layer_type):
    freq_moniter = 50

    if layer_type == 'F':
        generate_hessian_op = create_fc_hessian_computing_tf_graph(layer.layer_input)
    elif layer_type == 'C':
        generate_hessian_op = create_conv_hessian_computing_tf_graph(layer.layer_input)
    # 新的一轮
    sess.run(model.train_init)
    n_batches = 0
    layer_hessian = 0
    try:
        while True:
            # 一个batch的hessian结果
            # 好奇这个过程中模型的参数改变了没有
            this_hessian = sess.run(generate_hessian_op)
            layer_hessian += this_hessian
            n_batches += 1

            if n_batches % freq_moniter == 0:
                print('[%s] Now finish layer image No. %d batch' % (datetime.now(), n_batches))

    except tf.errors.OutOfRangeError:
        pass

    # 得到了这个layer的hessian
    layer_hessian = (1.0 / n_batches) * layer_hessian

    try:
        hessian_inv = inv(layer_hessian)
    except LinAlgError:
        hessian_inv = pinv(layer_hessian)
    return hessian_inv


def prune_weights(sess, model):
    print_('Preparing data')
    # 计算所有层的hessian_inv并且保存起来

    for layer_index, layer in enumerate(model.layers):
        layer_type = layer.layer_type

        # obtain the hessian inverse matrix
        hessian_inv = generate_hessian_inv(sess, model, layer, layer_type)

        # obtain the weights and biases matrix
        if layer_type == 'C':
            filt, biases = sess.run(layer.weight_tensors)
            filt_shape = tf.shape(filt)
            # 变成了(h * w * in, out)
            # 后面需要变回来
            weights = unfold_kernel(filt)
            wb = np.concatenate([weights, biases.reshape(1, -1)], axis=0)
        elif layer_type == 'F':
            weights, biases = sess.run(layer.weight_tensors)
            wb = np.hstack([weights, biases.reshape(-1, 1)]).transpose()

        l1, l2 = wb.shape

        # compute the sensitivity
        L = np.zeros([l1 * l2])
        for row_idx in range(l1):
            for col_idx in range(l2):
                L[row_idx * l2 + col_idx] = np.power(wb[row_idx, col_idx], 2) / (hessian_inv[row_idx, row_idx] + 10e-6)

        # rank the sensitivity
        sen_rank = np.argsort(L)

        # prune the weights
        n_prune = l1 * l2
        save_interval = n_prune / 20
        mask = np.ones(wb.shape)
        this_layer_CR_dict = dict()

        for i in range(n_prune):
            prune_idx = sen_rank[i]  # sen最低的元素的序号
            prune_row_idx = np.int(prune_idx / l2)  # 换算成原来的wb里面的index
            prune_col_idx = prune_idx % l2
            try:
                delta_W = - wb[prune_row_idx, prune_col_idx] / (
                        hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
            except Warning:
                print('Nan found, please change another Hessian inverse calculation method')
                break
            wb[:, prune_col_idx] += delta_W
            mask[prune_row_idx, prune_col_idx] = 0

            if i % save_interval == 0 and i / save_interval >= 4:
                wb = np.multiply(wb, mask)
                CR = 100 - (i / save_interval) * 5
                if layer_type == 'F':
                    weights = wb[0:-1, :].transpose()
                    biases = wb[-1, :].transpose()

                elif layer_type == 'C':
                    weights = wb[0:-1, :]
                    biases = wb[-1, :]
                elif layer_type == 'R':
                    weights = wb

                this_layer_CR_dict[str(CR)] = dict()

    # 完成所有的layer对所有的压缩率的情况之后
    # save pruned weights

    # if not os.path.exists('%s/%s/CR_%s' % (save_root, task_name, CR))


# 这一层的剪枝完成了，再把weights, biases变回来


# if i % save_interval == 0 and i / save_interval >= 4:
#     wb = np.multiply(wb, mask)
#     print('Construct element-wise multiplication between weight and mask matrix graph')


# Save pruned weights
# if not os.path.exists('%s/CR_%s' % (save_root, CR)):
#     os.makedirs('%s/CR_%s' % (save_root, CR))

# save file
# if layer_type == 'F':
#     np.save('%s/CR_%s/%s.weight' % (save_root, CR, layer_name), wb[0: -1, :].transpose())
#     np.save('%s/CR_%s/%s.bias' % (save_root, CR, layer_name), wb[-1, :].transpose())
# elif layer_type == 'C':
#     kernel = fold_weights(wb[0:-1, :], kernel_shape)
#     bias = wb[-1, :]
#     np.save('%s/CR_%s/%s.weight' % (save_root, CR, layer_name), kernel)
#     np.save('%s/CR_%s/%s.bias' % (save_root, CR, layer_name), bias)
# elif layer_type == 'R':
#     kernel = fold_weights(wb, kernel_shape)
#     np.save('%s/CR_%s/%s.weight' % (save_root, CR, layer_name), kernel)
# if CR == 5:
#     break


def prune_weights(model):
    # step 1: 对模型的层进行循环

    for layer in model.layers:
        mask = np.ones()


def Lobs():
    # tensorflow gpu config
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    session = tf.Session(config=gpu_config)

    # Step 1: get layer input from a well-trained deep network
    pruning_config = process_config("../configs/pruning.json")
    task_name = 'mnist'
    if pruning_config.model_type == 'VGG':
        config = process_config("../configs/pruning.json")
        model = VGGNet(config, task_name, pruning_config.model_path)
    elif pruning_config.model_type == 'RES':
        config = process_config("../configs/res_net.json")
        model = ResNet(config, task_name, pruning_config.model_path)

    # 运行一下，以获得输入
    # 其余剪枝的时候用到的量
    training = tf.placeholder(dtype=tf.bool, name='training')
    # regularizer不用?
    # regularizer of the conv layer
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
    # regularizer of the fc layer
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)
    model.set_global_tensor(training, regularizer_conv, regularizer_fc)

    # 只要把数据顺下去就可以，不建立完整的图模式
    model.inference()
    # init
    session.run(tf.global_variables_initializer())

    # Step 2: get hessian_inv
    prune_weights(session, model)


if __name__ == '__main__':
    Lobs()
