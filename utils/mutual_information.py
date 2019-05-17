# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: mutual_information
@time: 2019-05-04 14:48

Description. 
"""

import numpy as np
import keras.backend as K

from scipy.special import logsumexp


def kde_input(hidden, delta=0.1):
    batch_size = hidden.shape[0]

    sum_i = 0
    for hi in hidden:
        sum_j = 0
        for hj in hidden:
            sum_j += np.exp(-0.5 / delta * np.linalg.norm(hi - hj))
        sum_i += np.log(sum_j * 1. / batch_size)
    return -1. / batch_size * sum_i


def kde_mi_cus(hidden, labels_unique, labels_count, labels_inverse, delta=0.1):
    batch_size = hidden.shape[0]

    H_T = kde_input(hidden, delta=0.1)

    sum_l = 0
    for index, label in enumerate(labels_unique):
        p_l = labels_count[index]

        sum_i = 0
        for index_hi, hi in enumerate(hidden[labels_inverse == index, ...]):
            sum_j = 0
            for index_hj, hj in enumerate(hidden[labels_inverse == index, ...]):
                sum_j += np.exp(-0.5 / delta * np.linalg.norm(hi - hj))
            sum_i += np.log(1. / p_l * sum_j)
        sum_l += sum_i

    H_T_Y = -1. / batch_size * sum_l

    return H_T, H_T - H_T_Y


def get_shape(x):
    dims = np.shape(x)[1] * 1.
    N = np.shape(x)[0] * 1.
    return dims, N


def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims / 2.0) * (np.log(2 * np.pi * var) + 1)


def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + np.transpose(x2) - 2 * np.dot(X, np.transpose(X))
    return dists


def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2 * var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -np.mean(lprobs)
    return dims / 2 + h


# iclr18 on the information bottleneck theory of deep learning
def get_unique_probs(x):
    # 这里是为了找到unique的一行，所以首先把[batch_size, dim]压缩到[batch_size, 1]，然后再找到unique的count
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    # 每一个单独的行在batch_size中占的比例
    return np.asarray(unique_counts / float(sum(unique_counts)))


def kde_mi(hidden, labelixs, labelprobs):
    """

    :param hidden: 隐藏层的输出[batch_size, dim/channel_size]
    :param labelixs: 每一个label在哪些instance中出现过，即[batch_size, 1]
    :param labelprobs: 每一个label出现的概率，[dim_label]
    :return:
    """
    # Added Gaussian noise variance
    noise_variance = 1e-1

    h_upper = entropy_estimator_kl(hidden, noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    for j in range(len(labelixs)):
        if sum(labelixs[j]) != 0:
            hcond_upper = entropy_estimator_kl(hidden[labelixs[j], :], noise_variance)
            hM_given_Y_upper += labelprobs[j] * hcond_upper

    # hM_given_X = kde_condentropy(hidden, noise_variance)

    # MI_XM_upper = h_upper - hM_given_X
    MI_YM_upper = h_upper - hM_given_Y_upper

    # return MI_XM_upper, MI_YM_upper
    return 0, MI_YM_upper



def kde_mi_independent(hidden, labels):
    """

    :param hidden: 隐藏层的输出[batch_size, dim/channel_size]
    :param labelixs: 每一个label在哪些instance中出现过，即[batch_size, 1]
    :param labelprobs: 每一个label出现的概率，[dim_label]
    :return:
    """
    # Added Gaussian noise variance
    noise_variance = 1e-1
    h_upper = entropy_estimator_kl(hidden, noise_variance)

    # 对第label_index个独立的label，看做二分类问题
    hM_given_Y_upper_sum = 0
    for label_index in range(labels.shape[1]):
        labelixs = {}
        labelixs[0] = labels[:, label_index] == -1
        labelixs[1] = labels[:, label_index] == 1

        prob_label = np.mean((labels[:, label_index] == 1).astype(np.float32), axis=0)
        labelprobs = np.array([1 - prob_label, prob_label])

        # Compute conditional entropies of layer activity given output
        hM_given_Y_upper = 0.
        for j in range(len(labelixs)):
            if sum(labelixs[j]) != 0:
                hcond_upper = entropy_estimator_kl(hidden[labelixs[j], :], noise_variance)
                hM_given_Y_upper += labelprobs[j] * hcond_upper
        hM_given_Y_upper_sum += hM_given_Y_upper

    MI_YM_upper = h_upper * labels.shape[1] - hM_given_Y_upper_sum

    return MI_YM_upper


def kde_mi_unique(hidden, labels):
    """

    :param hidden: 隐藏层的输出[batch_size, dim/channel_size]
    :param labelixs: 每一个label在哪些instance中出现过，即[batch_size, 1]
    :param labelprobs: 每一个label出现的概率，[dim_label]
    :return:
    """
    # Added Gaussian noise variance
    noise_variance = 1e-1
    h_upper = entropy_estimator_kl(hidden, noise_variance)

    # 这里是为了找到unique的一行，所以首先把[batch_size, dim]压缩到[batch_size, 1]，然后再找到unique的count
    uniqueids = np.ascontiguousarray(labels).view(np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
    unique_value, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True,
                                                            return_counts=True)
    # 每一个独特的行（label）在整体中出现的概率，相当于labelprobs
    labelprobs = np.asarray(unique_counts / float(sum(unique_counts)))
    # 每一个独特的行（label）在整体中出现的位置
    labelixs = {}
    for label_index, label_value in enumerate(unique_value):
        labelixs[label_index] = unique_inverse == label_index

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    for j in range(len(labelixs)):
        if sum(labelixs[j]) != 0:
            hcond_upper = entropy_estimator_kl(hidden[labelixs[j], :], noise_variance)
            hM_given_Y_upper += labelprobs[j] * hcond_upper

    MI_YM_upper = h_upper - hM_given_Y_upper

    return MI_YM_upper


def bin_mi(hidden, labelixs, binsize=0.5):
    """

    :param hidden:
    :param labelixs: 每一个label在哪些instance中出现过，即[batch_size, 1]
    :param binsize:
    :return:
    """

    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        # 得到的是[unique]
        p_ts = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(hidden)
    H_LAYER_GIVEN_OUTPUT = 0

    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(hidden[ixs, :])

    # TODO: 以下为test
    # if 1.0 / np.log(2) * H_LAYER - H_LAYER_GIVEN_OUTPUT < 0:
    #     return 1.0 / np.log(2) * H_LAYER, 0
    # else:
    #     return 1.0 / np.log(2) * H_LAYER, 1.0 / np.log(2) * H_LAYER - H_LAYER_GIVEN_OUTPUT
    return 1.0 / np.log(2) * H_LAYER, 1.0 / np.log(2) * H_LAYER - H_LAYER_GIVEN_OUTPUT
