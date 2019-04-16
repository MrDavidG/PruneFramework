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
from utils.config import process_config

from numpy.linalg import inv, pinv, LinAlgError
import tensorflow as tf
import numpy as np
from datetime import datetime
import os

import pickle


# Construct hessian computing graph for res layer (conv layer without bias)
def create_res_hessian_computing_tf_graph(layer_input, kernel_size=3, layer_stride=1, stride_factor=3):
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
    patches = tf.extract_image_patches(images=layer_input,
                                       ksizes=[1, kernel_size, kernel_size, 1],
                                       strides=[1, layer_stride * stride_factor, layer_stride * stride_factor, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='SAME')

    a = tf.expand_dims(patches, axis=-1)
    b = tf.expand_dims(patches, axis=3)
    outprod = tf.multiply(a, b)
    get_hessian_op = tf.reduce_mean(outprod, axis=[0, 1, 2])
    return get_hessian_op


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


def create_conv_hessian_computing_tf_graph(layer_input, kernel_size=3, layer_stride=1, stride_factor=3):
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
                                       strides=[1, layer_stride * stride_factor, layer_stride * stride_factor, 1],
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


def fold_weights(weights, kernel_shape):
    """
    	In pytorch format, kernel is stored as [out_channel, in_channel, width, height]
    	Fold weights into a 4-dimensional tensor as [out_channel, in_channel, width, height]
    	:param weights:
    	:param kernel_shape:
    	:return:
    	"""
    kernel = np.zeros(shape=kernel_shape)
    for i in range(kernel_shape[3]):
        kernel[:, :, :, i] = weights[:, i].reshape([kernel_shape[0], kernel_shape[1], kernel_shape[2]])

    return kernel


def generate_hessian_inv(sess, model, layer, layer_type, batch_size, n_batch_used=100):
    freq_moniter = n_batch_used * batch_size

    if layer_type == 'F':
        generate_hessian_op = create_fc_hessian_computing_tf_graph(layer.layer_input)
    elif layer_type == 'C':
        generate_hessian_op = create_conv_hessian_computing_tf_graph(layer.layer_input)
    elif layer_type == 'R':
        generate_hessian_op = create_res_hessian_computing_tf_graph(layer.layer_input)
    # 新的一轮
    sess.run(model.train_init)
    n_batches = 0
    layer_hessian = 0
    try:
        while True:
            # 一个batch的hessian结果
            # 好奇这个过程中模型的参数改变了没有
            this_hessian = sess.run(generate_hessian_op, feed_dict={model.is_training: False})
            layer_hessian += this_hessian
            n_batches += 1

            if n_batches % freq_moniter == 0:
                print('\r[%s] Now finish image No. %d / %d' % (
                    datetime.now(), n_batches * batch_size, n_batch_used * batch_size), end=' ')

            if n_batches == n_batch_used:
                break
    except tf.errors.OutOfRangeError:
        pass

    # 得到了这个layer的hessian
    layer_hessian = (1.0 / n_batches) * layer_hessian

    try:
        hessian_inv = inv(layer_hessian)
    except LinAlgError:
        hessian_inv = pinv(layer_hessian)
    return hessian_inv


# Construct hessian inverse computing graph for Woodbury
def create_Woodbury_hessian_inv_graph(input_shape, dataset_size):
    """
    This function create the hessian inverse calculation graph using Woodbury method.
    """

    hessian_inv_holder = tf.placeholder(dtype=tf.float32, shape=[input_shape, input_shape])
    input_holder = tf.placeholder(dtype=tf.float32, shape=[1, input_shape])

    denominator = dataset_size + tf.matmul(a=tf.matmul(a=input_holder, b=hessian_inv_holder), b=input_holder,
                                           transpose_b=True)

    numerator = tf.matmul(a=tf.matmul(a=hessian_inv_holder, b=input_holder, transpose_b=True),
                          b=tf.matmul(a=input_holder, b=hessian_inv_holder))
    hessian_inv_op = hessian_inv_holder - numerator * (1.00 / denominator)

    return hessian_inv_holder, input_holder, hessian_inv_op


def generate_hessian_inv_Woodbury(sess, model, layer, layer_type, batch_size, n_batch_used=100,
                                  stride_factor=3, kernel_size=3, layer_stride=1):
    """
    This function calculated Hessian inverse matrix by Woodbury matrix identity.
    Args:
        Please find the same parameters explanations above.
    """
    hessian_inverse = None
    freq_moniter = (n_batch_used * batch_size) / 50

    # Begin process
    sess.run(model.train_init)
    n_batches = 0
    try:
        while True:
            # obtain the input in type of numpy array
            layer_input = sess.run(layer.layer_input, feed_dict={model.is_training: False})

            # construct tf graph
            if n_batches == 0:
                if layer_type == 'C' or layer_type == 'R':
                    layer_input_holder = tf.placeholder(dtype=tf.float32, shape=layer_input.shape)

                    get_patches_op = tf.extract_image_patches(images=layer_input,
                                                              ksizes=[1, kernel_size, kernel_size, 1],
                                                              strides=[1, layer_stride * stride_factor,
                                                                       layer_stride * stride_factor, 1],
                                                              rates=[1, 1, 1, 1],
                                                              padding='SAME')
                    dataset_size = n_batch_used * int(get_patches_op.get_shape()[0]) * int(
                        get_patches_op.get_shape()[1]) * int(get_patches_op.get_shape()[2])
                    input_dimension = get_patches_op.get_shape()[3]
                    if layer_type == 'C':
                        hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = create_Woodbury_hessian_inv_graph(
                            input_dimension + 1, dataset_size)
                    else:
                        hessian_inverse = 1000000 * np.eye(input_dimension)
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = create_Woodbury_hessian_inv_graph(
                            input_dimension, dataset_size)
                else:
                    layer_input_np = layer_input.cpu().numpy()
                    input_dimension = layer_input_np.shape[1]
                    dataset_size = n_batch_used * batch_size
                    hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                    hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = create_Woodbury_hessian_inv_graph(
                        input_dimension + 1, dataset_size)

            # Begin progress
            # sess.run(tf.global_variables_initializer())
            if layer_type == 'F':
                for i in range(layer_input.shape[0]):
                    this_input = layer_input[i]
                    wb = np.concatenate([this_input.reshape(1, -1), np.array([1.0]).reshape(1, -1)], axis=1)
                    hessian_inverse = sess.run(Woodbury_hessian_inv_op,
                                               feed_dict={hessian_inv_holder: hessian_inverse, input_holder: wb})

            elif layer_type == 'C' or layer_type == 'R':
                this_patch = sess.run(get_patches_op, feed_dict={layer_input_holder: layer_input})

                for i in range(this_patch.shape[0]):
                    for j in range(this_patch.shape[1]):
                        for m in range(this_patch.shape[2]):
                            this_input = this_patch[i][j][m]
                            if layer_type == 'C':
                                wb = np.concatenate([this_input.reshape(1, -1), np.array([1.0]).reshape(1, -1)],
                                                    axis=1)
                            else:
                                wb = this_input.reshape(1, -1)
                            hessian_inverse = sess.run(Woodbury_hessian_inv_op,
                                                       feed_dict={hessian_inv_holder: hessian_inverse,
                                                                  input_holder: wb})

            n_batches += 1
            if n_batches % freq_moniter == 0:
                print('\r[%s] Now finish image No. %d / %d' % (
                    datetime.now(), n_batches * batch_size, n_batch_used * batch_size), end=' ')

            if n_batches == n_batch_used:
                break
    except tf.errors.OutOfRangeError:
        pass

    return hessian_inverse


def prune_weights(sess, model, task_name, model_type, batch_size, use_Woodbury_list=list(),
                  save_root='./pruning_weights/',
                  n_batch_used=100):
    print('[%s] Preparing data' % (datetime.now()))
    # 计算所有层的hessian_inv并且保存起来

    for layer_index, layer in enumerate(model.layers):
        start_time = datetime.now()
        layer_type = layer.layer_type
        layer_name = layer.layer_name

        # for bn layer in resnet
        if layer_type == 'None':
            continue

        # obtain the hessian inverse matrix
        if layer_name in use_Woodbury_list:
            print('[%s] Using Woodbury for layer %s' % (datetime.now(), layer_name))
            hessian_inv = generate_hessian_inv_Woodbury(sess, model, layer, layer_type, batch_size,
                                                        n_batch_used=n_batch_used)
        else:
            hessian_inv = generate_hessian_inv(sess, model, layer, layer_type, batch_size, n_batch_used=n_batch_used)

        end_time = datetime.now()
        print('[%s] Use %d seconds' % (datetime.now(), (end_time - start_time).seconds))

        # obtain the weights and biases matrix
        if layer_type == 'C':
            filt, biases = sess.run(layer.weight_tensors)
            # (h * w * in, out)
            filt_shape = np.shape(filt)
            weights = unfold_kernel(filt)
            wb = np.concatenate([weights, biases.reshape(1, -1)], axis=0)
        elif layer_type == 'F':
            weights, biases = sess.run(layer.weight_tensors)
            wb = np.hstack([weights, biases.reshape(-1, 1)]).transpose()
        elif layer_type == 'R':
            # just the kernels, without biases
            filt = sess.run(layer.weight_tensors[0])
            filt_shape = np.shape(filt)
            wb = unfold_kernel(filt)

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
        save_interval = int(n_prune / 20)
        mask = np.ones(wb.shape)

        for i in range(n_prune):
            prune_idx = sen_rank[i]  # sen最低的元素的序号
            prune_row_idx = int(prune_idx / l2)  # 换算成原来的wb里面的index
            prune_col_idx = prune_idx % l2
            try:
                delta_W = - wb[prune_row_idx, prune_col_idx] / (
                        hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
            except Warning:
                print('[%s] Nan found, please change another Hessian inverse calculation method' % (datetime.now()))
                break
            wb[:, prune_col_idx] += delta_W
            mask[prune_row_idx, prune_col_idx] = 0

            # save weights for each CR and each layer
            CR_list = list()
            if i % save_interval == 0 and i / save_interval >= 4:
                # wb = np.multiply(wb, mask)
                CR = 100 - (i / save_interval) * 5
                CR_list += [CR]
                weights_dict = dict()

                if layer_type == 'F':
                    weights = wb[0:-1, :].transpose()
                    biases = wb[-1, :].transpose()

                    weights_dict[layer_name + '/biases'] = biases
                elif layer_type == 'C':
                    weights_ = fold_weights(wb[0:-1, :], filt_shape)
                    biases_ = wb[-1, :]

                    weights_dict[layer_name + '/biases'] = biases
                elif layer_type == 'R':
                    weights = fold_weights(wb, filt_shape)
                weights_dict[layer_name + '/weights'] = weights

                # save pruning weights
                if not os.path.exists('%s/%s/' % (save_root, task_name)):
                    os.makedirs('%s/%s/' % (save_root, task_name))

                # 读取之前的weights
                save_path = '/'.join([save_root, model_type.lower() + '_' + task_name, 'CR_' + str(CR)])
                # 合并之前的weights

                if os.path.exists(save_path):
                    file_handler = open(save_path, 'rb')
                    weights_CR = pickle.load(file_handler)
                    file_handler.close()
                    weights_CR = dict(weights_CR, **weights_dict)
                    # update weights
                    file_handler = open(save_path, 'wb')
                    pickle.dump(weights_CR, file_handler)
                    file_handler.close()
                else:
                    # save weights
                    file_handler = open(save_path, 'wb')
                    pickle.dump(weights_dict, file_handler)
                    file_handler.close()

                if CR == 10:
                    break

        print('[%s] Finish computation for layer %s' % (datetime.now(), layer_name))

    # fetch the original weights
    weights_dict_model = model.fetch_weight(sess)
    # update the weights under different CRs
    for CR in [CR_list]:
        save_path = '/'.join([save_root, model_type.lower() + '_' + task_name, 'CR_' + str(CR)])
        # obtain pruning weights
        file_handler = open(save_path, 'rb')
        weights_CR = pickle.load(pickle.load(file_handler))
        file_handler.close()
        # combine the dict
        weights_dict_model = dict(weights_dict_model, **weights_CR)
        # save complete model weights
        file_handler = open(save_path, 'wb')
        pickle.dump(weights_CR, file_handler)
        file_handler.close()


def retrain(model_type, task_name, save_root='./pruning_weights/'):
    training = tf.placeholder(dtype=tf.bool, name='training')
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    for file_name in os.listdir(save_root + model_type.lower() + '_' + task_name):
        model_path = save_root + model_type.lower() + '_' + task_name + '/' + file_name

        tf.reset_default_graph()
        session = tf.Session(config=gpu_config)

        if model_type == 'VGG':
            config = process_config("../configs/vgg_net.json")
            model = VGGNet(config, task_name, model_path)
        elif model_type == 'RES':
            config = process_config("../configs/res_net.json")
            model = ResNet(config, task_name, model_path)

        model.set_global_tensor(training, regularizer_conv, regularizer_fc)
        model.build()

        session.run(tf.global_variables_initializer())

        model.train(sess=session, n_epochs=20, lr=0.001)

        # re-save weights
        weights_dict_model = model.fetch_weight(session)
        file_handler = open(model_path, 'wb')
        pickle.dump(weights_dict_model, file_handler)
        file_handler.close()

        session.close()


def Lobs():
    # tensorflow gpu config
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    session = tf.Session(config=gpu_config)

    # Step 1: get a well-trained deep network
    config_pruning = process_config("../configs/lobs_res.json")
    task_name = 'mnist'
    if config_pruning.model_type == 'VGG':
        model = VGGNet(config_pruning, task_name, config_pruning.model_path)
    elif config_pruning.model_type == 'RES':
        model = ResNet(config_pruning, task_name, config_pruning.model_path)

    # init the model
    training = tf.placeholder(dtype=tf.bool, name='training')
    # regularizers
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0001)
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0005)
    model.set_global_tensor(training, regularizer_conv, regularizer_fc)

    # only build the tf graph of inference
    model.inference()
    # init params
    session.run(tf.global_variables_initializer())

    # Step 2: get hessian_inv and prune
    use_Woodbury_list = config_pruning.use_Woodbury_list
    model_type = config_pruning.model_type
    batch_size = config_pruning.batch_size
    prune_weights(session, model, task_name, model_type, batch_size, use_Woodbury_list)

    # Step 3: retrain
    session.close()
    retrain(config_pruning.model_type, task_name)


if __name__ == '__main__':
    Lobs()
