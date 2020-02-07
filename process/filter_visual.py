# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: process
@time: 2019/12/13 4:38 下午

Description.
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from pruning_algorithms.rdnet_multi import init_tf
from utils.configer import load_cfg
from models.model import Model
from PIL import Image
from utils.logger import logger

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import os

# gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'


# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def get_layers_output(path_model, name_layers, batch_size=1):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini')

    cfg.set('train', 'batch_size', str(batch_size))

    tf.reset_default_graph()
    sess = init_tf()

    model = Model(cfg)
    model.build()

    sess.run(tf.global_variables_initializer())

    sess.run(model.test_init)
    layers_output_tf = [tf.nn.relu(model.get_layer_by_name(name).layer_output) for name in name_layers]
    layers_output = sess.run(layers_output_tf, feed_dict={model.is_training: False})

    return layers_output


def draw_fmap(path_model, name_layers, inds, suffix=None):
    logger.record_log = False

    # [batch_size, h, w, channel_size]
    fmaps = get_layers_output(path_model, name_layers)

    # Normalize
    for ind, fmap_ in enumerate(fmaps):
        for ind_filter in inds:
            fmap = fmap_[0, :, :, ind_filter]
            max, min = np.max(fmap), np.min(fmap)

            fmap = (fmap - min) / (max - min) * 255

            img = Image.fromarray(fmap)
            img = img.convert("L")

            if suffix is None:
                img.save('./fmap-%s-ind%d.png' % (name_layers[ind], ind_filter))
            else:
                img.save('./fmap-%s-%s-ind%d.png' % (name_layers[ind], suffix, ind_filter))


if __name__ == '__main__':
    draw_fmap(
        '../exp_files/celeba_a-vgg128-2019-12-14_08-58-16/tr02-epo020-acc0.9014',
        ['c5_3'],
        [46, 100, 463, 506, 45, 237, 460],
        'TASK-A'
    )

    # draw_fmap(
    #     '../exp_files/celeba_b-vgg128-2019-12-14_11-36-08/tr02-epo020-acc0.8839',
    #     ['c5_3'],
    #     [85, 143, 401, 70, 227, 416, 500],
    #     'TASK-B'
    # )
