# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: cal_mean_std
@time: 2019-11-27 11:14

Description. 
"""

import tensorflow as tf
import numpy as np


def run():
    path_data = '/local/home/david/Datasets/fashionmnist/img/'

    s = tf.Session()

    avg = 0

    for i in range(10000):
        file_name = path_data + str(i) + '.png'

        img_string = tf.read_file(file_name)
        img = tf.cast(tf.image.decode_jpeg(img_string, channels=1), dtype=tf.float32)
        img = tf.image.resize_image_with_crop_or_pad(img, 124, 496)

        a = s.run(img)
        new = np.mean(a)
        avg += (new - avg) / (i + 1)
        # print(avg)
        print(i + 1, avg)

    print(avg)


if __name__ == '__main__':
    run()
