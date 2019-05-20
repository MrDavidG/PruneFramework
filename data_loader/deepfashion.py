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
    path_labels = '/local/home/david/Downloads/DeepFashion/list_attr_img_20.txt'
    path_ori_partition = '/local/home/david/Downloads/DeepFashion/list_eval_partition.txt'
    labels = pd.read_csv(path_labels, delim_whitespace=True, header=None)
    partition = pd.read_csv(path_ori_partition, delim_whitespace=True, header=1)

    assert(labels.values.shape[0], partition.values.shape[0])

    for i in range(labels.values.shape[0]):
        if labels.values[i, 0] != partition.values[i, 0]:
            print('两个的order不一样！！！！')

    print('haha')


if __name__ == '__main__':
    path = '/local/home/david/Downloads/DeepFashion'

    order_is_right()

