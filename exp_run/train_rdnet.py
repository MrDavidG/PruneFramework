# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: run_rdnet
@time: 2019-12-02 15:31

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/")

from pruning_algorithms.rdnet_multi import pruning

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

if __name__ == '__main__':
    task_lists = [
        {
            'task_name': 'rdnet_lfw_1',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.001,
            'cluster_fc': 0.001,
            'cluster_layer_range': np.arange(9, 15).tolist(),

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
                 'type': 'normal',
                 'train': [{'n_epochs': 80, 'lr': 0.01, 'save_clean': True}]},
                {'kl_factor': 1e-6,
                 'type': 'individual',
                 'train': [{'n_epochs': 10, 'lr': 0.001}]}
            ]
        }
    ]

    # [1. / 8, 0, 1. / 4, 0, 1. / 2, 0, 0, 10]

    for task in task_lists:
        print(str(task))

        pruning(
            task_name=task['task_name'],
            model_name=task['model_name'],
            data_name=task['data_name'],
            cluster_set={
                'path_cluster_res': task.get('path_cluster_res', None),
                'batch_size': task.get('batch_size', 128),
                'cluster_threshold_dict': {"conv": task['cluster_conv'], "fc": task['cluster_fc']},
                'cluster_layer_range': task['cluster_layer_range']
            },
            pruning_set={
                'name': 'info_bottle',
                'gamma_conv': task.get('gamma_conv', 1.),
                'gamma_fc': task.get('gamma_fc', 30.),
                'kl_mult': task['kl_mult'],
                'ib_threshold_conv': task.get('ib_threshold_conv', 0.01),
                'ib_threshold_fc': task.get('ib_threshold_fc', 0.01),
                'pruning_threshold': -1  # 没用，完全为了占位置
            },
            plan_retrain=task['plan_retrain'],
            path_model=None
        )
