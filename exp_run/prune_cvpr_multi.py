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

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

if __name__ == '__main__':
    task_lenet5 = [
        # baseline2
        {
            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',

            'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-19_17-45-43/cluster_results/cluster_result_threshold-{\'conv\': 0.1, \'fc\': 0.1}',

            'path_model': None,

            'plan_prune': [
                {'n_epochs': 23, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 10, 'lr': 0.01},
                {'n_epochs': 10, 'lr': 0.005},
                {'n_epochs': 10, 'lr': 0.001}
            ],
            'suffix': 'cvpr-lenet5-baseline2'
        },
        # cvpr
        {
            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',

            'path_cluster_res': '../exp_files/cluster_res_lenet5',

            # 'path_model': None,

            'path_model': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2020-06-01_10-05-26-cvpr-lenet5/epoch14-cr0.0457-crf0.1572',

            'plan_prune': [
                # {'n_epochs': 15, 'lr': 0.01}
                {'n_epochs': 0, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 60, 'lr': 0.01},
                {'n_epochs': 15, 'lr': 0.008},
                {'n_epochs': 10, 'lr': 0.006},
                {'n_epochs': 10, 'lr': 0.004},
                {'n_epochs': 10, 'lr': 0.002},
                {'n_epochs': 10, 'lr': 0.001},
                {'n_epochs': 10, 'lr': 0.0001}
            ],
            'suffix': 'cvpr-lenet5'
        }
    ]

    task_celeba = [
        # baseline2
        {
            'task_name': 'rdnet_celeba',
            'model_name': 'vgg512',
            'data_name': 'celeba',

            'path_cluster_res': '/local/home/david/Remote/PruneFramework/exp_files/rdnet_celeba-vgg512-rdnet-2020-02-04_10-16-18-baseline2/cluster_results/cluster_result_threshold-{conv: 0, fc: 0}',

            'path_model': None,

            'plan_prune': [
                {'n_epochs': 26, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 10, 'lr': 0.01},
                {'n_epochs': 10, 'lr': 0.005},
                {'n_epochs': 10, 'lr': 0.001}
            ],
            'suffix': 'cvpr-baseline2'
        },
        # cvpr
        {
            'task_name': 'rdnet_celeba',
            'model_name': 'vgg512',
            'data_name': 'celeba',

            # 'path_cluster_res': '../exp_files/rdnet_celeba-vgg512-rdnet-2020-02-01_03-51-47/cluster_results/cluster_result_threshold-{conv: 0.1, fc: 1.9}',

            'path_cluster_res': '../exp_files/cluster_res_celeba_vgg512',

            'path_model': None,

            'plan_prune': [
                {'n_epochs': 25, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 10, 'lr': 0.01},
                {'n_epochs': 10, 'lr': 0.005},
                {'n_epochs': 10, 'lr': 0.001}
            ],
            'suffix': 'cvpr'
        }
    ]

    task_lfw15 = [
        # fine tuning
        # {
        #     'task_name': 'rdnet_lfw15',
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #
        #     'path_cluster_res': '../exp_files/rdnet_lfw15-vgg128-rdnet-2019-12-12_13-01-01/cluster_results/cluster_result_threshold-{\'conv\': 0.1, \'fc\': 0.1}',
        #
        #     'path_model': '/local/home/david/Remote/PruneFramework/exp_files/rdnet_lfw15-vgg128-rdnet-2020-05-18_05-20-49-cvpr-baseline2/fine-epoch0-cr0.0001-crf0.0006',
        #
        #     'plan_prune': [
        #         # {'n_epochs': 40, 'lr': 0.01}
        #     ],
        #     'plan_fine': [
        #         {'n_epochs': 10, 'lr': 0.05},
        #         {'n_epochs': 10, 'lr': 0.01},
        #         {'n_epochs': 10, 'lr': 0.005},
        #         {'n_epochs': 10, 'lr': 0.001},
        #         {'n_epochs': 10, 'lr': 0.0005},
        #         {'n_epochs': 10, 'lr': 0.0001}
        #     ],
        #     'suffix': 'cvpr-baseline2'
        # },
        # baseline2
        {
            'task_name': 'rdnet_lfw15',
            'model_name': 'vgg128',
            'data_name': 'lfw',

            'path_cluster_res': '../exp_files/rdnet_lfw15-vgg128-rdnet-2019-12-12_13-01-01/cluster_results/cluster_result_threshold-{\'conv\': 0.1, \'fc\': 0.1}',

            # 'path_model': None,

            'path_model': '../exp_files/rdnet_lfw15-vgg128-rdnet-2020-06-02_07-49-32-cvpr-baseline2/epoch1-cr0.0000-crf0.0003',

            'plan_prune': [
                # {'n_epochs': 2, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 30, 'lr': 0.01},
                {'n_epochs': 20, 'lr': 0.075},
                {'n_epochs': 20, 'lr': 0.005},
                {'n_epochs': 20, 'lr': 0.0025},
                {'n_epochs': 20, 'lr': 0.001},
                {'n_epochs': 20, 'lr': 0.0075},
                {'n_epochs': 20, 'lr': 0.0005},
                {'n_epochs': 20, 'lr': 0.0025},
                {'n_epochs': 20, 'lr': 0.0001}
            ],
            'suffix': 'cvpr-baseline2'
        },
        # cvpr
        {
            'task_name': 'rdnet_lfw15',
            'model_name': 'vgg128',
            'data_name': 'lfw',

            # 'path_cluster_res': '../exp_files/rdnet_lfw15-vgg128-rdnet-2019-12-06_03-53-59/cluster_results/cluster_result_threshold-{\'conv\': 0.05, \'fc\': 0.05}',
            'path_cluster_res': '../exp_files/cluster_res_lfw15_vgg128',

            # 'path_model': '../exp_files/rdnet_lfw15-vgg128-rdnet-2020-05-18_13-40-47-cvpr/epoch19-cr0.0001-crf0.0003',
            'path_model': '../exp_files/rdnet_lfw15-vgg128-rdnet-2020-06-01_19-13-10-cvpr/epoch17-cr0.0001-crf0.0003',

            'plan_prune': [
                {'n_epochs': 1, 'lr': 0.01}
            ],
            'plan_fine': [
                {'n_epochs': 20, 'lr': 0.01},
                {'n_epochs': 15, 'lr': 0.005},
                {'n_epochs': 15, 'lr': 0.001},
                {'n_epochs': 15, 'lr': 0.0005},
                {'n_epochs': 15, 'lr': 0.0001}
            ],
            'suffix': 'cvpr'
        }
    ]

    for task in [task_lenet5[1]]:
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
