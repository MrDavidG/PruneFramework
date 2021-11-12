# encoding: utf-8

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from pruning_algorithms.rdnet_multi_res import pruning

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    # An example to prune resnet18 by VIBNet
    tasks_resnet18 = [
        {
            'task_name': 'rdnet_celeba_resnet18',
            'model_name': 'rdnet_resnet18',
            'data_name': 'celeba',

            'path_cluster_res': None,
            
            'path_model': None,

            'plan_prune': [
                {'n_epochs': 40, 'lr': 0.01}
            ],

            'plan_fine': [
                {'n_epochs': 10, 'lr': 0.01},
                {'n_epochs': 10, 'lr': 0.005},
                {'n_epochs': 10, 'lr': 0.001},
                {'n_epochs': 10, 'lr': 0.0005},
                {'n_epochs': 10, 'lr': 0.0001}
            ],
            'suffix': 'cvpr'
        }
    ]

    for task in tasks_resnet18:
        print(str(task))

        pruning(
            task_name=task['task_name'],
            model_name=task['model_name'],
            data_name=task['data_name'],

            path_model=task.get('path_model', None),

            cluster_set={
                'path_cluster_res': task.get('path_cluster_res', None),
                'batch_size': task.get('batch_size', 128),
                'cluster_threshold_dict': {'c1': 0, 's1': 5, 's2': 10, 's3': 10, 's4': 10},
                'cluster_layer_range': task.get('cluster_layer_range', None)
            },

            plan_prune=task['plan_prune'],
            plan_fine=task['plan_fine'],
            batch_size=task.get('batch_size', 32),
            suffix=task.get('suffix', None)
        )
