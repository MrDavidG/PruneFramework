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

# gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'

# gpu 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

if __name__ == '__main__':
    task_celeba = [
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_a',
            'plan_prune': {'n_epochs': 21, 'lr': 0.01},
            'path_model': '../exp_files/celeba_a-vgg512-2020-01-30_12-01-08/tr02-epo020-acc0.9028',
        },
        {
            'model_name': 'vgg512',
            'data_name': 'celeba',
            'task_name': 'celeba_b',
            'plan_prune': {'n_epochs': 0, 'lr': 0.01},
            # 'path_model': '../exp_files/celeba_b-vgg512-2020-01-30_16-58-08/tr00-epo010-acc0.8903'
            'path_model': '../exp_files/celeba_b-vgg512-cvpr-2020-06-01_15-57-06/epoch19-cr0.0115-crf0.0491'
        }
    ]

    task_lenet5 = [
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_a',
            'plan_prune': {'n_epochs': 17, 'lr': 0.01},
            'path_model': '../exp_files/fashionmnist_a-lenet5-2019-12-19_14-46-21/tr02-epo020-acc0.9605'
        },
        {
            'model_name': 'lenet5',
            'data_name': 'fashionmnist',
            'task_name': 'fashionmnist_b',
            'plan_prune': {'n_epochs': 17, 'lr': 0.01},
            'path_model': '../exp_files/fashionmnist_b-lenet5-2019-12-19_14-51-13/tr02-epo020-acc0.9637',
        }
    ]

    task_lfw15_vib_list = [
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_0',
            # 'path_model': '../exp_files/lfw15_0-vgg128-2019-12-05_12-36-04/tr02-epo020-acc0.9023',
            'path_model': '../exp_files/lfw15_0-vgg128-cvpr-2020-06-02_09-20-23/epoch0-cr0.0031-crf0.0074'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_1',
            # 'path_model': '../exp_files/lfw15_1-vgg128-2019-12-05_12-46-30/tr02-epo020-acc0.8415',
            'path_model': '../exp_files/lfw15_1-vgg128-cvpr-2020-06-02_09-37-01/epoch7-cr0.0005-crf0.0075'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_2',
            # 'path_model': '../exp_files/lfw15_2-vgg128-2019-12-05_12-56-54/tr02-epo020-acc0.8503',
            'path_model': '../exp_files/lfw15_2-vgg128-cvpr-2020-06-02_09-40-56/epoch28-cr0.0011-crf0.0074'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_3',
            # 'path_model': '../exp_files/lfw15_3-vgg128-2019-12-05_13-07-07/tr02-epo020-acc0.8662',
            'path_model': '../exp_files/lfw15_3-vgg128-cvpr-2020-06-02_09-46-41/epoch28-cr0.0010-crf0.0068'
        },
        {
            'model_name': 'vgg128',
            'data_name': 'lfw',
            'task_name': 'lfw15_4',
            # 'path_model': '../exp_files/lfw15_4-vgg128-2019-12-06_03-25-14/tr02-epo020-acc0.8744',
            'path_model': '../exp_files/lfw15_4-vgg128-cvpr-2020-06-02_10-08-25/epoch0-cr0.0007-crf0.0069'
        }
    ]

    for task in [task_lfw15_vib_list[4]]:
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
