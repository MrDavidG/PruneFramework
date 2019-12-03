# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: train_model
@time: 2019-12-02 15:32

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.model import exp

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

if __name__ == '__main__':
    # 实验分为三个部分，model，data，task(labels,也就是标签要哪些)
    # model:提供网络结构
    # data:提供数据集train/test分的方式，或者是mean，std值
    # task:提供的是label的序号，也就是相当于指定了任务，同时也把任务名指出来，从此以后找data不再依靠task_name

    tasks = [
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist'
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_a',
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_b',
        },
        {
            'model_name': 'vgg512',
            'data_name': 'deepfashion',
            'task_name': 'deepfashion',
        },

        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_0',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_8',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_11',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_16',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_17',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_18',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_20',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_27',

        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_32',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_42',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_46',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_58',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_66',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_68',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw_70',
        },
    ]

    # 'kl_mult': [1. / 32, 1. / 32, 0,
    #                         1. / 16, 1. / 16, 0,
    #                         1. / 8, 1. / 8, 1. / 8, 0,
    #                         1. / 4, 1. / 4, 1. / 4, 0,
    #                         2., 2., 2., 0,
    #                         0,
    #                         1., 1.]

    tasks_vib = [
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_a',
            'path_model': '../exp_files/fashionmnist_a-lenet5-2019-11-27_09-27-20/tr02-epo015-acc0.9761',

            'pruning': True,
            'gamma_conv': 0.1,
            'gamma_fc': 30.,
            'kl_mult': [1. / 8, 0,
                        1. / 4, 0,
                        1. / 2, 0,
                        0,
                        10.],
            'plan_train_vib': [
                {'kl_factor': 5e-6, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': True}]},
                {'kl_factor': 1e-7, 'train': [{'n_epochs': 10, 'lr': 0.01, 'save_clean': True}]}
            ]
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_b',
            'path_model': '../exp_files/fashionmnist_b-lenet5-2019-11-27_09-31-06/tr02-epo015-acc0.9800',

            'pruning': True,
            'gamma_conv': 0.1,
            'gamma_fc': 30.,
            'kl_mult': [1. / 8, 0,
                        1. / 4, 0,
                        1. / 2, 0,
                        0,
                        10.],
            'plan_train_vib': [
                {'kl_factor': 5e-6, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': True}]},
                {'kl_factor': 1e-7, 'train': [{'n_epochs': 10, 'lr': 0.01, 'save_clean': True}]}
            ]
        }
    ]

    for task in tasks[4:]:
        exp(
            model_name=task['model_name'],
            data_name=task['data_name'],
            task_name=task['task_name'],

            save_step='-1',
            plan_train_normal=task.get('plan_train_normal',
                                       [{'n_epochs': 20, 'lr': 0.01},
                                        {'n_epochs': 20, 'lr': 0.001},
                                        {'n_epochs': 20, 'lr': 0.0001}]),

            pruning=task.get('pruning', False),
            pruning_set={
                'name': 'info_bottle',
                'gamma_conv': task.get('gamma_conv', None),
                'gamma_fc': task.get('gamma_fc', None),
                'kl_mult': task.get('kl_mult', None),
                'pruning_threshold': 0.01
            },
            plan_train_vib=task.get('plan_train_vib', None),
            path_model=None
        )
