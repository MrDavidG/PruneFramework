import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

from models.model_res_gate import exp

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    # An example to train and prune resnet18 on task celeba_a
    tasks_celeba_resnet18 = [
        {
            'model_name': 'resnet18',
            'data_name': 'celeba',
            'task_name': 'celeba_a',

            'plan_prune': {'n_epochs': 21, 'lr': 0.01},
            'path_model': None
        }
    ]

    

    for task in tasks_celeba_resnet18:
        exp(
            model_name=task['model_name'],
            data_name=task['data_name'],
            task_name=task['task_name'],

            plan_prune=task.get('plan_prune', {'n_epochs': 0, 'lr': 0.01}),
            plan_tune=task.get('plan_tune', [{'n_epochs': 15, 'lr': 0.01},
                                             {'n_epochs': 15, 'lr': 0.005},
                                             {'n_epochs': 15, 'lr': 0.001}]),
            path_model=task['path_model'],
            suffix=task.get('suffix', 'cvpr'),
            batch_size=task.get('batch_size', 32)
        )
