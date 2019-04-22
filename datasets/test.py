# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: test
@time: 2019-04-16 10:22

Description.
"""
import pickle

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import os


def getx():
    for img in os.listdir('/local/home/david/Datasets/imagenet12/val/100/'):
        img_path = '/local/home/david/Datasets/imagenet12/val/0/ILSVRC2012_val_00025527.JPEG'
        # '/local/home/david/Datasets/imagenet12/val/100/' + img
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='caffee')

        return x


def keras_vgg16():
    model = VGG16(weights='imagenet')

    count = 0.
    for label in range(1000):
        for img in os.listdir('/local/home/david/Datasets/imagenet12/train/' + str(label)):
            img_path = '/local/home/david/Datasets/imagenet12/train/' + str(label) + '/' + img
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x, mode='caffee')

            result = model.predict(x)
            if np.argmax(result) == label:
                count += 1.

    print(count/10000)

def cus():
    weights = VGG16(weights='imagenet').get_weights()

    x = getx()
    # 1

    # 开始构建网络

    c1 = tf.nn.conv2d(x, weights[0], [1, 1, 1, 1], padding='SAME')
    c1 = tf.nn.bias_add(c1, weights[1])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(c1)
    print(np.shape(output))
    print(np.argmax(output))


if __name__ == '__main__':
    keras_vgg16()
    # print(getx())

def savedir():
    w = {'ucf101std': [0.10898684, 0.10810246, 0.10679939],
         'omniglotstd': [0.27046935, 0.27046935, 0.27046935],
         'caltech256mean': [0.55177064, 0.53382816, 0.50567107],
         'vgg-petsstd': [0.25922821, 0.25420126, 0.26146911],
         'sketchesmean': [0.97960958, 0.97960958, 0.97960958],
         'dtdstd': [0.25226877, 0.2409176, 0.2499409],
         'vgg-petsmean': [0.47812268, 0.44583364, 0.39578528],
         'svhnmean': [0.4376821, 0.4437697, 0.47280442],
         'daimlerpedclsmean': [0.48203529, 0.48203529, 0.48203529],
         'mnistmean': [0.18228742, 0.18228742, 0.18228742],
         'planktonstd': [0.16591533, 0.16591533, 0.16591533],
         'gtsrbstd': [0.27560347, 0.26576119, 0.27089863],
         'planktonmean': [0.94033073, 0.94033073, 0.94033073],
         'imagenet12_largemean': [0.485, 0.456, 0.406],
         'vgg-flowersstd': [0.28509808, 0.23842338, 0.2639633],
         'gtsrbmean': [0.33921263, 0.3117836, 0.32047045],
         'ucf101mean': [0.49953101, 0.49880386, 0.49981996],
         'svhnstd': [0.19803012, 0.20101562, 0.19703614], 'dtdmean': [0.52696497, 0.47025164, 0.42396662],
         'mit-indoorstd': [0.25213846, 0.24675795, 0.24937516],
         'aircraftstd': [0.21070221, 0.20508901, 0.23729657],
         'cifar100mean': [0.50705882, 0.48666667, 0.44078431],
         'mit-indoormean': [0.48822609, 0.43138942, 0.37296835],
         'omniglotmean': [0.08099839, 0.08099839, 0.08099839],
         'aircraftmean': [0.47983041, 0.51074066, 0.53437998],
         'caltech256std': [0.31026209, 0.30732538, 0.32094492],
         'sketchesstd': [0.06532731, 0.06532731, 0.06532731],
         'daimlerpedclsstd': [0.23612616, 0.23612616, 0.23612616],
         'imagenet12_largestd': [0.229, 0.224, 0.225],
         'imagenet12mean': [103.939, 116.779, 123.68], 'imagenet12std': [1., 1., 1.],
         'vgg-flowersmean': [0.43414682, 0.38309883, 0.29714763],
         'cifar100std': [0.26745098, 0.25647059, 0.27607843],
         'mniststd': [0.38076293, 0.38076293, 0.38076293]}
    import pickle

    pickle.dump(w, open('/local/home/david/Remote/datasets/datasets_mean_std.pickle', 'wb'))
