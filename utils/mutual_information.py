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
from scipy.special import logsumexp


def kde_input(inputs, hidden, delta=0.1):
    batch_size = inputs.shape[0]

    sum_i = 0
    for hi in hidden:
        sum_j = 0
        for hj in hidden:
            sum_j += np.exp(-0.5 / delta * np.linalg.norm(hi - hj))
        sum_i += np.log(sum_j * 1. / batch_size)
    return -1. / batch_size * sum_i


def kde_mi_cus(inputs, labels, hidden, delta=0.1):
    batch_size = labels.shape[0]

    H_T = kde_input(inputs, hidden, delta=0.1)

    labels_unique, labels_count = np.unique(labels, return_counts=True)

    sum_l = 0
    for index, label in enumerate(labels_unique):
        p_l = labels_count[index]

        sum_i = 0
        for index_hi, hi in enumerate(hidden):
            if labels[index_hi] == label:
                sum_j = 0
                for index_hj, hj in enumerate(hidden):
                    if labels[index_hj] == label:
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
    # Added Gaussian noise variance
    noise_variance = 1e-1

    h_upper = entropy_estimator_kl(hidden, noise_variance)

    # Compute conditional entropies of layer activity given output
    hM_given_Y_upper = 0.
    for j in range(10):
        if sum(labelixs[j]) != 0:
            hcond_upper = entropy_estimator_kl(hidden[labelixs[j], :], noise_variance)
            hM_given_Y_upper += labelprobs[j] * hcond_upper
        # hcond_upper = entropy_estimator_kl(hidden[labelixs[j], :], noise_variance)
        # hM_given_Y_upper += labelprobs[j] * hcond_upper

    hM_given_X = kde_condentropy(hidden, noise_variance)

    MI_XM_upper = h_upper - hM_given_X
    MI_YM_upper = h_upper - hM_given_Y_upper

    return MI_XM_upper, MI_YM_upper


def bin_mi(hidden, labelixs, binsize=0.5):
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        # 得到的是[unique]
        p_ts = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(hidden)
    H_LAYER_GIVEN_OUTPUT = 0

    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(hidden[ixs, :])

    return 1.0 / np.log(2) * H_LAYER, 1.0 / np.log(2) * H_LAYER - H_LAYER_GIVEN_OUTPUT
