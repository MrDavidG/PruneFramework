# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: run_rdnet
@time: 2019-12-02 15:31

Description
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from pruning_algorithms.rdnet_multi import pruning

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    task_lists = [
        {
            'task_name': 'rdnet_lfw_1',
            'model_name': 'vgg128',
            'data_name': 'lfw',

            # hyperparameters for PMT
            'path_cluster_res': None,
            'cluster_conv': 0.01,
            'cluster_fc': 0.05,
            'cluster_layer_range': [_ for _ in range(15)],

            'path_model': None,

            # hyperparameters for VIBNet
            'gamma_conv': 1.,
            'gamma_fc': 30.,
            'kl_mult': [1. / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 4, 1. / 4, 1. / 4, 0,
                        2., 2., 2., 0,
                        0,
                        1., 1.],
            'plan_retrain': [
                {'kl_factor': 1e-5,
                 'train': [{'n_epochs': 80, 'lr': 0.01, 'type': 'normal', 'save_clean': True}]},
                {'kl_factor': 1e-6,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': True}]}
            ]
        }
    ]


    for task in task_lists:
        print(str(task))

        pruning(
            task_name=task['task_name'],
            model_name=task['model_name'],
            data_name=task['data_name'],
            cluster_set={
                'path_cluster_res': task.get('path_cluster_res', None),
                'batch_size': task.get('batch_size', 128),
                'cluster_threshold_dict': {"conv": task.get('cluster_conv', 0), "fc": task.get('cluster_fc', 0)},
                'cluster_layer_range': task.get('cluster_layer_range', list())
            },
            pruning_set={
                'name': task.get('name', 'info_bottle'),
                'gamma_conv': task.get('gamma_conv', 1.),
                'gamma_fc': task.get('gamma_fc', 30.),
                'kl_mult': task.get('kl_mult', str([])),
                'ib_threshold_conv': task.get('ib_threshold_conv', 0.01),
                'ib_threshold_fc': task.get('ib_threshold_fc', 0.01),
                'pruning_threshold': -1  
            },
            plan_retrain=task['plan_retrain'],
            path_model=task.get('path_model', None),
            retrain=task.get('retrain', True),

            # suffix 
            suffix=task.get('suffix', None)
        )
