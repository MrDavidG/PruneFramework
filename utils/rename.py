# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: rename
@time: 2019-11-21 21:08

Description. 
"""

import pickle


if __name__ == '__main__':
    path_weights = '/local/home/david/Remote/PruneFramework/dataset/vgg_pretrain_deepfashion'

    w = pickle.load(open(path_weights, 'rb'))

    a = ['conv1_1/weights', 'conv1_2/weights', 'conv2_1/weights', 'conv2_2/weights', 'conv3_1/weights', 'conv3_2/weights', 'conv3_3/weights', 'conv4_1/weights', 'conv4_2/weights', 'conv4_3/weights', 'conv5_1/weights', 'conv5_2/weights', 'conv5_3/weights',
         'conv1_1/biases', 'conv1_2/biases', 'conv2_1/biases', 'conv2_2/biases', 'conv3_1/biases', 'conv3_2/biases', 'conv3_3/biases', 'conv4_1/biases', 'conv4_2/biases', 'conv4_3/biases', 'conv5_1/biases', 'conv5_2/biases', 'conv5_3/biases']
    # b = ['c1_1/w', 'c1_2/w', 'c2_1/w', 'c2_2/w', 'c3_1/w', 'c3_2/w', 'c3_3/w', 'c4_1/w', 'c4_2/w', 'c4_3/w', 'c5_1/w', 'c5_2/w', 'c5_3/w', 'f6/w', 'f7/w', 'f8/w', 'c1_1/b', 'c1_2/b', 'c2_1/b', 'c2_2/b', 'c3_1/b', 'c3_2/b', 'c3_3/b', 'c4_1/b', 'c4_2/b', 'c4_3/b', 'c5_1/b', 'c5_2/b', 'c5_3/b', 'f6/b', 'f7/b', 'f8/b']

    b = ['c1_1/w', 'c1_2/w', 'c2_1/w', 'c2_2/w', 'c3_1/w', 'c3_2/w', 'c3_3/w', 'c4_1/w', 'c4_2/w', 'c4_3/w', 'c5_1/w',
         'c5_2/w', 'c5_3/w', 'c1_1/b', 'c1_2/b', 'c2_1/b', 'c2_2/b', 'c3_1/b', 'c3_2/b',
         'c3_3/b', 'c4_1/b', 'c4_2/b', 'c4_3/b', 'c5_1/b', 'c5_2/b', 'c5_3/b']

    w_dict = dict()

    for old, new in zip(a, b):
        w_dict[new] = w[old]


    pickle.dump(w_dict, open('/local/home/david/Remote/PruneFramework/dataset/vgg16_pretrain_conv_imgnet', 'wb'))
