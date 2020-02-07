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

    tasks_celeba_list = [
        # {
        #     'model_name': 'vgg512',
        #     'data_name': 'celeba',
        #     'task_name': 'celeba_a',
        # },
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_b',

            'plan_train_normal': [{'n_epochs': 10, 'lr': 0.01},
                                  {'n_epochs': 5, 'lr': 0.001},
                                  {'n_epochs': 5, 'lr': 0.0001}]
        }
    ]

    tasks_celeba_vib_list = [
        # {
        #     'model_name': 'vgg512',
        #     'data_name': 'celeba',
        #     'task_name': 'celeba_a',
        #     'path_model': '../exp_files/celeba_a-vgg512-2020-01-30_12-01-08/tr02-epo020-acc0.9028',
        #
        #     'pruning': True,
        #     'gamma_conv': 1.,
        #     'gamma_fc': 30.,
        #     'kl_mult': [1.15 / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 4, 1. / 4, 1. / 4, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_train_vib': [
        #         {'kl_factor': 1e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': False}]}
        #     ]
        # },
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_b',
            'path_model': '../exp_files/celeba_b-vgg512-2020-01-30_16-58-08/tr00-epo010-acc0.8903',

            'pruning': True,
            'gamma_conv': 1.,
            'gamma_fc': 30.,
            'kl_mult': [1. / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 4, 1. / 4, 1. / 4, 0,
                        2., 2., 2., 0,
                        0,
                        1., 1.],
            'plan_train_vib': [
                        {'kl_factor': 1e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
                        {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': False}]}
            ]
        }
    ]

    tasks_lenet5_list = [
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_a',
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_b',
        }
    ]

    tasks_lenet5_vib_list = [
        # {
        #     'model_name': 'lenet5',
        #     'data_name': 'fashionmnist',
        #     'task_name': 'fashionmnist_a',
        #     'path_model': '../exp_files/fashionmnist_a-lenet5-2019-12-19_14-46-21/tr02-epo020-acc0.9605',
        #
        #     'pruning': True,
        #     'gamma_conv': 0.1,
        #     'gamma_fc': 30.,
        #     'kl_mult': [1.3 / 8, 0,
        #                 1.6 / 4, 0,
        #                 0,
        #                 9.,
        #                 6.],
        #     'plan_train_vib': [
        #         {'kl_factor': 5e-6, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': False}]},
        #         {'kl_factor': 1e-7, 'train': [{'n_epochs': 5, 'lr': 0.0001, 'save_clean': False}]}
        #     ]
        # },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_b',
            'path_model': '../exp_files/fashionmnist_b-lenet5-2019-12-19_14-51-13/tr02-epo020-acc0.9637',

            'pruning': True,
            'gamma_conv': 0.1,
            'gamma_fc': 30.,
            'kl_mult': [0.125, 0,
                        0.7, 0,
                        0,
                        9.,
                        6.],
            'plan_train_vib': [
                {'kl_factor': 5e-6, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
                {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': False}]},
                {'kl_factor': 1e-7, 'train': [{'n_epochs': 5, 'lr': 0.0001, 'save_clean': False}]}
            ]
        }
    ]

    tasks_lfw15_list = [
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_0',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_1',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_2',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_3',
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_4',
        }
    ]

    tasks_lfw_ab = [
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_0b',
        }
    ]

    tasks_lfw15_vib_list = [
        # {
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #     'task_name': 'lfw15_0',
        #     'path_model': '../exp_files/lfw15_0-vgg128-2019-12-05_12-36-04/tr02-epo020-acc0.9023',
        #
        #     'pruning': True,
        #     'gamma_conv': 1.,
        #     'gamma_fc': 50.,
        #     'kl_mult': [1. / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 4, 1. / 4, 1. / 4, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_train_vib': [
        #         {'kl_factor': 3.2e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': True}]}
        #     ]
        # },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_1',
            'path_model': '../exp_files/lfw15_1-vgg128-2019-12-05_12-46-30/tr02-epo020-acc0.8415',

            'pruning': True,
            'gamma_conv': 1.,
            'gamma_fc': 50.,
            'kl_mult': [1.15 / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 4, 1. / 4, 1. / 4, 0,
                        2., 2., 2., 0,
                        0,
                        1., 1.],
            'plan_train_vib': [
                {'kl_factor': 3.45e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
                {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': True}]},
                {'kl_factor': 1e-7, 'train': [{'n_epochs': 5, 'lr': 0.001, 'save_clean': True}]}
            ]
        },
        # {
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #     'task_name': 'lfw15_2',
        #     'path_model': '../exp_files/lfw15_2-vgg128-2019-12-05_12-56-54/tr02-epo020-acc0.8503',
        #
        #     'pruning': True,
        #     'gamma_conv': 1.,
        #     'gamma_fc': 50.,
        #     'kl_mult': [1. / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 4, 1. / 4, 1. / 4, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_train_vib': [
        #         {'kl_factor': 3.47e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': True}]}
        #     ]
        # },
        # {
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #     'task_name': 'lfw15_3',
        #     'path_model': '../exp_files/lfw15_3-vgg128-2019-12-05_13-07-07/tr02-epo020-acc0.8662',
        #
        #     'pruning': True,
        #     'gamma_conv': 1.,
        #     'gamma_fc': 50.,
        #     'kl_mult': [1. / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 4, 1. / 4, 1. / 4, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_train_vib': [
        #         {'kl_factor': 3.2e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': True}]}
        #     ]
        # },
        # {
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #     'task_name': 'lfw15_4',
        #     'path_model': '../exp_files/lfw15_4-vgg128-2019-12-06_03-25-14/tr02-epo020-acc0.8744',
        #
        #     'pruning': True,
        #     'gamma_conv': 1,
        #     'gamma_fc': 50.,
        #     'kl_mult': [1. / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 4, 1. / 4, 1. / 4, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_train_vib': [
        #         {'kl_factor': 3.5e-5, 'train': [{'n_epochs': 30, 'lr': 0.01, 'save_clean': False}]},
        #         {'kl_factor': 1e-6, 'train': [{'n_epochs': 10, 'lr': 0.001, 'save_clean': True}]}
        #     ]
        # }
    ]

    tasks_lfw10_list = [
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_0'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_1'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_2'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_3'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_4'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_5'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_6'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_7'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_8'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_9'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw10_10'
        }
    ]

    tasks_scenario2 = [
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_scenario2_a',

            'plan_train_normal': [{'n_epochs': 0, 'lr': 0}]
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_scenario2_b',

            'plan_train_normal': [{'n_epochs': 0, 'lr': 0}]
        }
    ]

    for task in tasks_celeba_vib_list:
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
            path_model=task.get('path_model', None)
        )
