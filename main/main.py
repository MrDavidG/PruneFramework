# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: main
@time: 2019-03-27 16:16

Description. 
"""

from utils.config import process_config
from utils.logger import Logger
from train.vgg_train import VGGTrain
from models.vgg_model import VGGModel

import tensorflow as tf

if __name__ == '__main__':
    """
    
    example of VGGNet
    
    """
    # obtain config
    config = process_config('...')
    # create sess
    sess = tf.Session()
    # create the model
    model = VGGModel(config, task_name='', model_path='')
    # TODO: create data generator
    # data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)

    # train
    trainer = VGGTrain(sess, model, data, config, logger)
