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

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
from models.vgg_model import VGGNet
from models.resnet_model import ResNet
from utils.config import process_config

def keras_vgg16():
    model = VGG16(weights='imagenet')

    img_path = '/local/home/david/Downloads/ILSVRC2012_val_00007368.JPEG'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)

    print(result)

if __name__ == '__main__':

    keras_vgg16()

    config = process_config("../configs/vgg_net.json")
    # apply video memory dynamically
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    session = tf.Session(config=gpu_config)
    # session = tf.InteractiveSession()
    # 标志位
    training = tf.placeholder(dtype=tf.bool, name='training')
    # regularizer of the conv layer
    regularizer_conv = tf.contrib.layers.l2_regularizer(scale=0.0000)
    # regularizer of the fc layer
    regularizer_fc = tf.contrib.layers.l2_regularizer(scale=0.0000)

    # Step1: Train
    model = VGGNet(config, 'imagenet12',model_path='../models/model_weights/vgg_pretrain.npy')
    model.set_global_tensor(training, regularizer_conv, regularizer_fc)

    # only build the tf graph of inference

    img_path = '/local/home/david/Downloads/ILSVRC2012_val_00007368.JPEG'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model.X = x

    model.inference()
    # init params
    session.run(tf.global_variables_initializer())

    log = session.run([model.layers[0].layer_output], feed_dict={model.is_training: False})
    print(log)