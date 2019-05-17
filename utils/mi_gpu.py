# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: mi_gpu
@time: 2019-05-16 20:32

Description. 
"""

import keras.backend as K
import numpy as np


def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2 * K.dot(X, K.transpose(X))
    return dists


def get_shape(x):
    dims = K.cast(K.shape(x)[1], K.floatx())
    N = K.cast(K.shape(x)[0], K.floatx())
    return dims, N


def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2 * var)
    normconst = (dims / 2.0) * K.log(2 * np.pi * var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims / 2 + h


def get_K_function():
    Klayer_activity = K.placeholder(ndim=2)
    noise_variance = 1e-1
    entropy_func_upper = K.function([Klayer_activity, ], [entropy_estimator_kl(Klayer_activity, noise_variance), ])
    return entropy_func_upper


def kde_gpu(hidden, labelixs, labelprobs, entropy_func_upper):
    h_upper = entropy_func_upper([hidden, ])[0]

    hM_given_Y_upper = 0.
    for j in range(len(labelixs)):
        hcond_upper = entropy_func_upper([hidden[labelixs[j], :], ])[0]
        hM_given_Y_upper += labelprobs[j] * hcond_upper

    MI_YM_upper = h_upper - hM_given_Y_upper

    return 0, MI_YM_upper


def kde_gpu_without_hupper(h_upper, hidden, labelixs, labelprobs, entropy_func_upper):
    hM_given_Y_upper = 0.
    for j in range(len(labelixs)):
        if labelixs[j].max():
            hcond_upper = entropy_func_upper([hidden[labelixs[j], :], ])[0]
        else:
            hcond_upper = 0
        hM_given_Y_upper += labelprobs[j] * hcond_upper

    MI_YM_upper = h_upper - hM_given_Y_upper

    return 0, MI_YM_upper


def kde_in_gpu(hidden, labelixs, labelprobs, entropy_func_upper):
    h_upper = entropy_func_upper([hidden, ])[0]
    sum = 0
    for j in range(len(labelixs)):
        _, mi = kde_gpu_without_hupper(h_upper, hidden, labelixs[j], labelprobs[j], entropy_func_upper)
        sum += mi
    return 0, sum
