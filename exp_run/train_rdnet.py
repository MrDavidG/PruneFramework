# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: run_rdnet
@time: 2019-12-02 15:31

Description.d
"""

import sys

sys.path.append(r"/local/home/david/Remote/PruneFramework")

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
            'cluster_conv': 0.01,
            'cluster_fc': 0.05,
            'cluster_layer_range': np.arange(14, 15).tolist(),

            'path_model': None,

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

    task_lenet5 = [
        # baseline
        # {
        #     'task_name': 'rdnet_fashionmnist',
        #     'model_name': 'rdnet_lenet5',
        #     'data_name': 'fashionmnist',
        #     'path_cluster_res': None,
        #     'cluster_conv': 0.1,
        #     'cluster_fc': 0.1,
        #     'cluster_layer_range': list(),
        #
        #     'path_model': None,
        #
        #     'gamma_conv': 1.,
        #     'gamma_fc': 45.,
        #     'kl_mult': [0.04, 0, 0.07, 0, 0, 8., 6.],
        #     'plan_retrain': [
        #         {'kl_factor': 3e-6,
        #          'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        #         {'kl_factor': 1e-7,
        #          'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]},
        #         {'kl_factor': 1e-8,
        #          'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': False}]
        #          }
        #     ]
        # },
        {
            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',
            'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-20_07-38-02/cluster_results/cluster_result_threshold-{\'conv\': 0.2, \'fc\': 0.15}',
            'cluster_conv': 0.2,
            'cluster_fc': 0.15,
            'cluster_layer_range': np.arange(0, 4).tolist(),

            'path_model': None,

            'gamma_conv': 1.,
            'gamma_fc': 45.,
            'kl_mult': [0.0625, 0, 0.4, 0, 0, 8., 5.5],
            'plan_retrain': [
                {'kl_factor': 3e-6,
                 'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 10, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 5, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-8,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': False}]
                 }
            ]
        }
    ]

    task_lfw15 = [
        # baseline
        # {
        #     'task_name': 'rdnet_lfw15',
        #     'model_name': 'vgg128',
        #     'data_name': 'lfw',
        #     'path_cluster_res': None,
        #     'cluster_conv': 0.1,
        #     'cluster_fc': 0.1,
        #     'cluster_layer_range': list(),
        #
        #     'path_model': None,
        #
        #     'gamma_conv': 1.,
        #     'gamma_fc': 30.,
        #     'kl_mult': [1. / 32, 1. / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 2, 1. / 2, 1. / 2, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1.5, 1.],
        #     'plan_retrain': [
        #         {'kl_factor': 1e-5,
        #          'train': [{'n_epochs': 60, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        #         {'kl_factor': 1e-6,
        #          'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': False}]}
        #     ]
        # },
        # rdnet
        {
            'task_name': 'rdnet_lfw15',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': '../exp_files/rdnet_lfw15-vgg128-rdnet-2019-12-06_03-53-59/cluster_results/cluster_result_threshold-{\'conv\': 0.05, \'fc\': 0.05}',

            'cluster_conv': 0.8,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(13, 15).tolist(),

            'path_model': '/local/home/david/Remote/PruneFramework/exp_files/rdnet_lfw15-vgg128-rdnet-2020-01-23_11-30-03/tr02-epo010-cr0.0000-acc0.8544',
            # None,

            'gamma_conv': 1.,
            'gamma_fc': 30.,
            'kl_mult': [1.02 / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 2, 1. / 2, 1. / 2, 0,
                        2., 2., 2., 0,
                        0,
                        1.5, 1.],
            'plan_retrain': [
                # {'kl_factor': 1.2e-5,
                #  'train': [{'n_epochs': 70, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                # {'kl_factor': 1e-6,
                #  'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]},
                # {'kl_factor': 1e-7,
                #  'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 10, 'lr': 0.0001, 'type': 'individual', 'save_clean': False}]},
            ]
        }
    ]

    task_lfw15_ab = [
        {
            'task_name': 'rdnet_lfw15_ab',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.05,
            'cluster_fc': 0.05,
            'cluster_layer_range': np.arange(7, 15).tolist(),

            'path_model': None,

            'gamma_conv': 1.,
            'gamma_fc': 30.,
            'kl_mult': [1. / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 2, 1. / 2, 1. / 2, 0,
                        2., 2., 2., 0,
                        0,
                        1., 1.],
            'plan_retrain': [
                {'kl_factor': 1e-5,
                 'train': [{'n_epochs': 0, 'lr': 0.01, 'type': 'normal', 'save_clean': True}]},
                {'kl_factor': 1e-6,
                 'train': [{'n_epochs': 0, 'lr': 0.001, 'type': 'individual', 'save_clean': True}]}
            ]
        }
    ]

    task_celeba_vision = [
        # celeba实验，用来进行可视化的vision
        {
            'task_name': 'rdnet_celeba',
            'model_name': 'vgg128',
            'data_name': 'celeba',
            'path_cluster_res': None,
            'cluster_conv': 0.1,
            'cluster_fc': 0.1,
            'cluster_layer_range': np.arange(3, 15).tolist(),

            'kl_mult': [1. / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 2, 1. / 2, 1. / 2, 0,
                        2., 2., 2., 0,
                        0,
                        1., 1.],
            'plan_retrain': [
                {'kl_factor': 1e-5,
                 'train': [{'n_epochs': 0, 'lr': 0.01, 'type': 'normal', 'save_clean': True}]},
                {'kl_factor': 1e-6,
                 'train': [{'n_epochs': 0, 'lr': 0.001, 'type': 'individual', 'save_clean': True}]}
            ]
        }
    ]

    task_lfw10 = [
        {
            'task_name': 'rdnet_lfw10_01',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_02',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_03',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_04',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_05',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_06',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_07',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_08',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_09',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        },
        {
            'task_name': 'rdnet_lfw10_10',
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'path_cluster_res': None,
            'cluster_conv': 0.6,
            'cluster_fc': 0.8,
            'cluster_layer_range': np.arange(11, 15).tolist(),

            'plan_retrain': list(),
            'retrain': False
        }
    ]

    task_scenario_1 = [
        # 验证情景1是否正确，用lenet来进行测试，剪枝不用那么用力，但是最终要保留clean的模型
        {
            'suffix': 'scenario1',

            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',
            'path_cluster_res': None,
            'cluster_conv': 0.1,
            'cluster_fc': 0.1,
            'cluster_layer_range': list(),

            'path_model': None,

            'gamma_conv': 1.,
            'gamma_fc': 45.,
            'kl_mult': [0.04, 0, 0.07, 0, 0, 8., 6.],
            'plan_retrain': [
                {'kl_factor': 3e-6,
                 'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-8,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'individual', 'save_clean': True}]}
            ]
        }
    ]

    task_scenario_2 = [
        # {
        #     'suffix': 'scenario2',
        #
        #     'task_name': 'rdnet_fashionmnist_scenario2',
        #     'model_name': 'rdnet_lenet5',
        #     'data_name': 'fashionmnist',
        #     'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-23_09-04-14-scenario2/cluster_results/cluster_result',
        #
        #     'name': 'original',
        #     'gamma_conv': 0.,
        #     'gamma_fc': 0.,
        #     'kl_mult': [0, 0, 0, 0, 0, 0, 0],
        #     'plan_retrain': [
        #         {'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        #         {'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
        #     ]
        # },

        # {
        #     'suffix': 'scenario2_same_task',
        #
        #     'task_name': 'rdnet_fashionmnist_scenario2_',
        #     'model_name': 'rdnet_lenet5',
        #     'data_name': 'fashionmnist',
        #
        #     'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-26_04-28-12-scenario2/cluster_results/cluster_result',
        #     # 这个是f4倒序
        #     # 'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-26_04-37-27-scenario2/cluster_results/cluster_result', # f4顺序
        #     # 'path_cluster_res': '../exp_files/rdnet_fashionmnist-rdnet_lenet5-rdnet-2019-12-26_04-43-08-scenario2/cluster_results/cluster_result', # 两个模型拼合
        #
        #     'name': 'original',
        #     'gamma_conv': 0.,
        #     'gamma_fc': 0.,
        #     'kl_mult': [0, 0, 0, 0, 0, 0, 0],
        #     'plan_retrain': [
        #         {'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        #         {'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
        #     ]
        # },
        # 将task a的loss缩小，仅仅对task b进行剪枝
        {
            'suffix': 'scenario2_prune_task_b',

            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',

            'task_name': 'rdnet_fashionmnist',
            'model_name': 'rdnet_lenet5',
            'data_name': 'fashionmnist',

            'cluster_layer_range': list(),

            'gamma_conv': 1.,
            'gamma_fc': 45.,
            'kl_mult': [0.04, 0, 0.07, 0, 0, 8., 6.],
            'plan_retrain': [
                {'kl_factor': 3e-6,
                 'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-7,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
            ]
        }
    ]

    task_celeba = [
        # baseline2
        {
            'task_name': 'rdnet_celeba',
            'model_name': 'vgg512',
            'data_name': 'celeba',

            'gamma_conv': 1.,
            'gamma_fc': 1.,
            'kl_mult': [1. / 32, 1. / 32, 0,
                        1. / 16, 1. / 16, 0,
                        1. / 8, 1. / 8, 1. / 8, 0,
                        1. / 2, 1. / 2, 1. / 2, 0,
                        1., 1., 1., 0,
                        0,
                        1., 1.], 
            'plan_retrain': [
                {'kl_factor': 1e-5,
                 'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
                {'kl_factor': 1e-6,
                 'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
            ],
            'suffix': 'baseline2'
        },
        # rdnet celeba
        # {
        #     'task_name': 'rdnet_celeba',
        #     'model_name': 'vgg512',
        #     'data_name': 'celeba',
        #
        #     'cluster_conv': 0.1,
        #     'cluster_fc': 1.9,
        #     'cluster_layer_range': np.arange(13, 15).tolist(),
        #
        #     'gamma_conv': 1.,
        #     'gamma_fc': 9,
        #     'kl_mult': [0.8 / 32, 0.8 / 32, 0,
        #                 1. / 16, 1. / 16, 0,
        #                 1. / 8, 1. / 8, 1. / 8, 0,
        #                 1. / 2, 1. / 2, 1. / 2, 0,
        #                 2., 2., 2., 0,
        #                 0,
        #                 1., 1.],
        #     'plan_retrain': [
        #         {'kl_factor': 1e-5,
        #          'train': [{'n_epochs': 30, 'lr': 0.01, 'type': 'normal', 'save_clean': False}]},
        #         {'kl_factor': 1e-6,
        #          'train': [{'n_epochs': 10, 'lr': 0.001, 'type': 'normal', 'save_clean': False}]}
        #     ]
        # }
    ]

    for task in task_celeba:
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
                'pruning_threshold': -1  # 没用，完全为了占位置
            },
            plan_retrain=task['plan_retrain'],
            path_model=task.get('path_model', None),
            retrain=task.get('retrain', True),

            # 文件名后缀
            suffix=task.get('suffix', None)
        )
