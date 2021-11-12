# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: cvpr_multi.py
@time: 2020/5/10 12:38 下午

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from pruning_algorithms.cvpr_multi import prune

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    # An example to train rdnet_lenet5 on task rdnet_fashionmnist, refer to ```config/rdnet_fashionmnet``` for their meaning
    task_lenet5 = [
        {
            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',

            'path_cluster_res': None,

            'path_model': None,

            'plan_prune': [
                {'n_epochs': 30, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 10, 'lr': 0.01},
                {'n_epochs': 10, 'lr': 0.005},
                {'n_epochs': 10, 'lr': 0.001}
            ]
        }
    ]

    for task in task_lenet5:
        prune(
            task_name=task['task_name'],
            model_name=task['model_name'],
            data_name=task['data_name'],

            path_model=task.get('path_model', None),

            cluster_set={
                'path_cluster_res': task['path_cluster_res']
            },

            plan_prune=task['plan_prune'],
            plan_fine=task['plan_fine'],
            suffix=task.get('suffix', None)
        )
