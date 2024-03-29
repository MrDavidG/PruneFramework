# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: prune_cvpr
@time: 2020/5/4 10:14 上午

Description. 
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.model_gate import exp

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    task_celeba = [
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_a',
            'plan_prune': {'n_epochs': 21, 'lr': 0.01},
            'path_model': '../exp_files/celeba_a-vgg512-2020-01-30_12-01-08/tr02-epo020-acc0.9028',
        }
    ]

    

    for task in task_celeba:
        exp(
            model_name=task['model_name'],
            data_name=task['data_name'],
            task_name=task['task_name'],

            plan_prune=task.get('plan_prune', {'n_epochs': 0, 'lr': 0.01}),
            plan_tune=task.get('plan_tune', [{'n_epochs': 20, 'lr': 0.01},
                                             {'n_epochs': 15, 'lr': 0.005},
                                             {'n_epochs': 15, 'lr': 0.001}]),
            path_model=task['path_model'],
            suffix=task.get('suffix', 'cvpr')
        )
