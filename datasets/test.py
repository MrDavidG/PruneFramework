import tensorflow as tf

import numpy as np
import torch


def unfold_kernel(kernel):
    """
    In pytorch format, kernel is stored as [height, width, in_channel, out_channel]
    Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
    :param kernel: numpy ndarray
    :return:
    """
    k_shape = kernel.shape
    weight = np.zeros([k_shape[0] * k_shape[1] * k_shape[2], k_shape[3]])
    for i in range(k_shape[3]):
        weight[:, i] = np.reshape(kernel[:, :, :, i], [-1])
    return weight


def his_unfold_kernel(kernel):
    """
    In pytorch format, kernel is stored as [out_channel, in_channel, height, width]
    Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
    :param kernel: numpy ndarray
    :return:
    """
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

    return weight


if __name__ == '__main__':
    a = np.array([[[[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]],
                   [[0.7, 0.8],
                    [0.9, 1.0],
                    [1.1, 1.2]],
                   [[1.3, 1.4],
                    [1.5, 1.6],
                    [1.7, 1.8]],
                   [[1.9, 2.],
                    [2.1, 2.2],
                    [2.3, 2.4]]],

                  [[[2.5, 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]]],
                  [[[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]]],
                  [[[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]]],
                  [[[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]],
                   [[0., 0.],
                    [0., 0.],
                    [0., 0.]]]])
    # 3,2,4,5
    b = np.array([[[[0.1, 0., 0., 0., 0.],
                    [0.7, 0., 0., 0., 0.],
                    [1.3, 0., 0., 0., 0.],
                    [1.9, 0., 0., 0., 0.]],

                   [[0.2, 0., 0., 0., 0.],
                    [0.8, 0., 0., 0., 0.],
                    [1.4, 0., 0., 0., 0.],
                    [2., 0., 0., 0., 0.]]],

                  [[[0.3, 0., 0., 0., 0.],
                    [0.9, 0., 0., 0., 0.],
                    [1.5, 0., 0., 0., 0.],
                    [2.1, 0., 0., 0., 0.]],

                   [[0.4, 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1.6, 0., 0., 0., 0.],
                    [2.2, 0., 0., 0., 0.]]],

                  [[[0.5, 0., 0., 0., 0.],
                    [1.1, 0., 0., 0., 0.],
                    [1.7, 0., 0., 0., 0.],
                    [2.3, 0., 0., 0., 0.]],

                   [[0.6, 0., 0., 0., 0.],
                    [1.2, 0., 0., 0., 0.],
                    [1.8, 0., 0., 0., 0.],
                    [2.4, 0., 0., 0., 0.]]]])
    print(his_unfold_kernel(a))
    print('!!!!!!!!!1')
    print(unfold_kernel(b))
    h = 2
    w = 1
    i = 0
    o = 1
    # print(a[o][i][h][w])
    # print(b[h][w][i][o])

    print(np.transpose(b, (3, 2, 0, 1)))  # 跟A一样了应该
    print(np.shape(b))
