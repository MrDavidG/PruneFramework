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

if __name__ == '__main__':
    # An experiment is consisted of model, data and task
    # model: choose from ```config/model.cfg```
    # data: choose from ```config/data.cfg```
    # task: choose from ```config/task.cfg```


    # An example of training VGG512 on task CelebA_a and CelebA_b (without pruning)
    tasks_celeba_list = [
        # train vgg512 on task celebA_a
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_a',
        },
        # train vgg512 on task celebA_b with specific training plan
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_b',

            'plan_train_normal': [{'n_epochs': 10, 'lr': 0.01},
                                  {'n_epochs': 5, 'lr': 0.001},
                                  {'n_epochs': 5, 'lr': 0.0001}]
        }
    ]


    # An example of pruning trained model with VIBNet
    tasks_celeba_vib_list = [
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_a',
            # load model, without training
            'path_model': '../exp_files/celeba_a-vgg512-2020-01-30_12-01-08/tr02-epo020-acc0.9028',
            
            # hyperparameters for VIBNet
            'pruning': True,
            'gamma_conv': 1.,
            'gamma_fc': 30.,
            'kl_mult': [1.15 / 32, 1. / 32, 0,
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

    

    for task in tasks_celeba_list:
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
