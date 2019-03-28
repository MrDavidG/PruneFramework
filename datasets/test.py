# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: test
@time: 2019-03-27 21:43

Description. 
"""

import os
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
import numpy as np
from PIL import Image

def toImage(x, y, save_dir = "./mnist/train/"):
    # 声明图片宽高
    rows = 28
    cols = 28

    # 获取图片总数
    images_count = np.shape(x)[0]
    pixels_per_image = 28 * 28

    images_to_extract = np.shape(x)[0]

    # 获取标签总数
    labels_count = np.shape(y)[0]

    # 检查数据集是否符合预期格式
    if images_count == labels_count:
        print("数据集总共包含 %s 张图片，和 %s 个标签" % (images_count, labels_count))
        print("每张图片包含 %s 个像素" % (pixels_per_image))
        print("数据类型：%s" % (x[0].dtype))

         # 创建数字图片的保存目录
        for i in range(10):
            dir = "%s/%s/" % (save_dir, i)
            if not os.path.exists(dir):
                print("目录 ""%s"" 不存在！自动创建该目录..." % dir)
                os.makedirs(dir)

        # 通过python图片处理库，生成图片
        indices = [0 for i in range(0, 10)]
        for i in range(0, images_to_extract):
            img = Image.new("L", (cols, rows))
            for m in range(rows):
                for n in range(cols):
                    img.putpixel((n, m), 255-int(x[i][m][n]))
                    # 根据图片所代表的数字label生成对应的保存路径
            digit = y[i]
            path = "%s/%s/%s.jpg" % (save_dir, y[i], indices[digit])
            indices[digit] += 1
            img.save(path)
            # 由于数据集图片数量庞大，保存过程可能要花不少时间，有必要打印保存进度
            if ((i + 1) % 50) == 0:
                print("图片保存进度：已保存 %s 张，共需保存 %s 张" % (i + 1, images_to_extract))

    else:
        print("图片数量和标签数量不一致！")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

toImage(x_train, y_train)
toImage(x_test, y_test, save_dir = "./mnist/test/")
