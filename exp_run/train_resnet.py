"""
 Train resnet for celeba_a and celeba_b
"""
import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.model_res import exp

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # an example to train resnet18 on CelebA
    tasks_celeba = [
        {
            'model_name': 'resnet18',
            'data_name': 'celeba',
            'task_name': 'celeba_a'
        },
        {
            'model_name': 'resnet18',
            'data_name': 'celeba',
            'task_name': 'celeba_b'
        },

        {
            'model_name': 'resnet18',
            'data_name': 'celeba',
            'task_name': 'celeba'
        }
    ]

    for task in tasks_celeba:
        exp(
            model_name=task['model_name'],
            data_name=task['data_name'],
            task_name=task['task_name'],

            save_step='-1',
            plan_train=task.get('plan_train_normal',
                                [
                                    {'n_epochs': 30, 'lr': 0.01},
                                ]),
            path_model=task.get('path_model', None),
            batch_size=task.get('batch_size', None)
        )
