# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: deepfashion
@time: 2019-05-20 22:24

Description. 
"""
import pandas as pd
import pickle
import os
import numpy as np
import cv2


def get_only_train_val():
    path_ori_partition = '/local/home/david/Downloads/DeepFashion/list_eval_partition.txt'
    # 存放train和val imgs的文件
    path_train_partition = '/local/home/david/Downloads/DeepFashion/list_train_imgs.txt'
    path_val_partition = '/local/home/david/Downloads/DeepFashion/list_val_imgs.txt'

    # 先把train和test数据分开
    file_labels = open(path_ori_partition, 'r')
    num = 0
    n_train = 0
    n_val = 0
    with open(path_train_partition, 'w') as file_labels_train:
        with open(path_val_partition, 'w') as file_labels_val:
            for line in file_labels:
                if num > 1:
                    content_list = line.split()
                    if content_list[1] == 'train':
                        file_labels_train.write(content_list[0] + '\n')
                        n_train += 1
                    elif content_list[1] == 'val':
                        file_labels_val.write(content_list[0] + '\n')
                        n_val += 1
                num += 1
    print('Num of train: {}, Num of val: {}'.format(n_train, n_val))


def get_20_labels():
    """
    提取出来20个label，最好文件的顺序也按照这个顺序来进行
    :return:
    """
    path = '/local/home/david/Downloads/DeepFashion/list_attr_img.txt'
    path_labels = '/local/home/david/Downloads/DeepFashion/list_attr_img_20.txt'

    content = pd.read_csv(path, delim_whitespace=True, header=None)
    print(content.shape)
    print(content.values[:, :21].shape)
    data = pd.DataFrame(content.values[:, :21])
    data.to_csv(path_labels, sep=' ', index=False, header=False)


def order_is_right():
    """
    验证labels数据集中的顺序和partition文件中imgs的顺序是一样的
    :return:
    """
    path_labels = '/local/home/david/Downloads/DeepFashion/list_attr_img_20.txt'
    path_ori_partition = '/local/home/david/Downloads/DeepFashion/list_eval_partition.txt'
    labels = pd.read_csv(path_labels, delim_whitespace=True, header=None)
    partition = pd.read_csv(path_ori_partition, delim_whitespace=True, header=1)

    assert (labels.values.shape[0], partition.values.shape[0])

    a = labels.values[:, 0]
    b = partition.values[:, 0]

    print((a == b).min())
    # 两者的顺序是完全一样的


def partition_spe():
    """
    去掉partition文件中的重复空格，使得可以使用pandas处理
    :return:
    """
    # 存放所有partition的文件
    path_ori_partition = '/local/home/david/Downloads/DeepFashion/list_eval_partition.txt'
    file = open(path_ori_partition, 'r')

    file_w = open('/local/home/david/Downloads/DeepFashion/partition_without_blank.txt', 'w')

    n = 0
    for line in file:
        if n == 0 or n == 1:
            n = n + 1
            continue
        content_list = line.split()

        if len(content_list) != 2:
            print(line)
            print('Error: Size does not equal 2.')
            break
        else:
            file_w.write(' '.join(content_list) + '\n')
    # 现在partition_without_blank.txt是所有图片的partition


def only_train_and_val():
    """
    在labels里面和partition里面去掉所有的test
    :return:
    """
    path_partition = '/local/home/david/Downloads/DeepFashion/partition_without_blank.txt'
    path_labels = '/local/home/david/Downloads/DeepFashion/list_attr_img.txt'

    partition = pd.read_csv(path_partition, delim_whitespace=True, header=None).values
    labels = pd.read_csv(path_labels, delim_whitespace=True, header=None).values

    assert (partition.shape[0], labels.shape[0])

    # 确认两者imgs顺序完全一样
    print('两者imgs顺序完全一样: {}'.format((partition[:, 0] == labels[:, 0]).min()))

    # 把test去掉,剩下的全都是train和val
    index_list = partition[:, 1] != 'test'

    # val和train的partition
    partition_list = (partition[index_list, 1] == 'train').astype(int)
    # pickle.dump(partition_list, open('/local/home/david/Downloads/DeepFashion/partition_deepfashion.pickle', 'wb'))

    print('一共有imgs{}'.format(len(partition_list)))

    pd.DataFrame(labels[index_list, :]).to_csv('/local/home/david/Downloads/DeepFashion/labels_1000_train_val.txt',
                                               sep=' ', index=False, header=False)

    print('labels文件的shape为{}'.format(labels[index_list, :].shape))


def verification():
    """
    验证imgs都存在并且可以根据目录找到
    :return:
    """
    path = '/local/home/david/Datasets/deepfashion/labels_20_train_val.txt'

    path_imgs = pd.read_csv(path, delim_whitespace=True, header=None).values[:, 0]

    print(path_imgs.shape)

    n = 0
    for path in path_imgs:
        if os.path.exists('/img/' + path):
            n += 1
        else:
            print('文件不存在: {}'.format(path))
            break

    print('共{}imgs均存在并匹配'.format(n))


def if_has_0():
    path = '/local/home/david/Datasets/deepfashion/labels_20_train_val.txt'

    labels = pd.read_csv(path, delim_whitespace=True, header=None).values[:, 1:]

    print((labels == 0).max())

    print(labels.shape)
    print(np.sum(labels == 1))
    print(np.sum(labels == -1))
    # 已确认其中不包含0，全部为-1和1


def change_dict():
    path = '/local/home/david/Datasets/deepfashion/labels_20_train_val.txt'

    content = pd.read_csv(path, delim_whitespace=True, header=None).values

    for i in range(content.shape[0]):
        content[i, 0] = '/img/' + content[i, 0]

    pd.DataFrame(content).to_csv('/local/home/david/Datasets/deepfashion/labels_20_train_val_path.txt',
                                 sep=' ', index=False, header=False)


def construct_labels():
    # [249222,1001] 已经剔除了test
    path = '/local/home/david/Downloads/DeepFashion/labels_1000_train_val.txt'
    content = pd.read_csv(path, delim_whitespace=True, header=None).values
    # [:,1000]
    labels = content[:, 1:]

    # 从新取得文件头
    files = pd.read_csv('/local/home/david/Datasets/deepfashion/labels_20_train_val_path.txt', delim_whitespace=True,
                        header=None).values[:, 0]

    files = np.expand_dims(files, 1)

    assert (files, content[:, 0])

    for i in range(10):
        # 每一行做or，结果为[n_sample, 1],值为true或false
        # 首先==1的为true，避免了-1的干扰，-1变成0，在求和，得出的是一行的家和，这个时候如果sum为0则是一行一个1都没有
        label = np.sum(labels[:, i * 100:i * 100 + 100] == 1, axis=1) > 0
        print('第{}个label为1的概率为{}'.format(i, label.sum() / label.shape[0]))
        # 变成1/-1
        label_res = label.astype(int) * 2 - 1
        label_res = np.expand_dims(label_res, 1)
        # 加入结果里面
        files = np.concatenate((files, label_res), axis=1)

    pd.DataFrame(files).to_csv('/local/home/david/Datasets/deepfashion/labels_10_train_val_path.txt', sep=' ',
                               index=False, header=False)


if __name__ == '__main__':
    construct_labels()
