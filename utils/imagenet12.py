import tensorflow as tf

import os
import scipy.io as scio
import shutil
import numpy as np


def preprocess():
    path_meta = '/local/home/david/Datasets/ILSVRC2012_devkit_t12/data/meta.mat'
    path_train = '/local/home/david/Datasets/ILSVRC2012/train/'
    path_val = '/local/home/david/Datasets/ILSVRC2012/val/'
    path_val_ground_truth = '/local/home/david/Datasets//ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

    # rename for train dataset
    # 读取mat
    dic = scio.loadmat(path_meta)
    array_synset = dic['synsets']

    for map in array_synset:
        ILSVRC_id = int(map[0][0][0][0])
        WNID = map[0][1][0]

        if ILSVRC_id > 1000:
            break

        os.rename(path_train + WNID, path_train + str(ILSVRC_id))

    # Put the images belong to the same category in validation set into the same fold
    # mkdir
    for i in range(1000):
        if not os.path.exists(path_val + str(i + 1)):
            os.mkdir(path_val + str(i + 1))

    file_ground_truth = open(path_val_ground_truth, 'r')

    for i, label_val in enumerate(file_ground_truth):
        # index of the image: 00000001—00050000

        index_label = str(i + 1).zfill(8)
        name_img_val = 'ILSVRC2012_val_' + index_label + '.JPEG'

        print(name_img_val, str(int(label_val)))
        if os.path.exists(path_val + name_img_val):
            if not os.path.exists(path_val + str(int(label_val)) + '/' + name_img_val):
                shutil.move(path_val + name_img_val, path_val + str(int(label_val)) + '/')
            else:
                print(name_img_val + '已存在')


def func():
    # 把imagenet/val数据集改变一下
    path_val = '/local/home/david/Downloads/val/'
    path_val_new = '/local/home/david/Datasets/imagenet12/val/'
    path_train_new = '/local/home/david/Datasets/imagenet12/train/'

    if not os.path.exists('/local/home/david/Datasets/imagenet12'):
        os.mkdir('/local/home/david/Datasets/imagenet12')
    if not os.path.exists(path_val_new):
        os.mkdir(path_val_new)
    if not os.path.exists(path_train_new):
        os.mkdir(path_train_new)

    for label in os.listdir(path_val):
        path_val_label = path_val + label + '/'
        path_val_label_new = path_val_new + label + '/'
        path_train_label_new = path_train_new + label + '/'

        if not os.path.exists(path_val_label_new):
            os.mkdir(path_val_label_new)
        if not os.path.exists(path_train_label_new):
            os.mkdir(path_train_label_new)

        imgs_list = os.listdir(path_val_label)

        imgs_set = set(imgs_list)
        imgs_train = np.random.choice(imgs_list, int(len(imgs_list) * 0.8), replace=False)
        imgs_val = list(imgs_set - set(imgs_train))

        for img in imgs_train:
            shutil.copy(path_val_label + img, path_train_label_new + img)
        for img in imgs_val:
            shutil.copy(path_val_label + img, path_val_label_new + img)

def true_label():
    path = '/local/home/david/Downloads/val/'

    path_label = '/local/home/david/Downloads/image_label.txt'

    for i in range(1000):
        if not os.path.exists('/local/home/david/Downloads/val/' + str(i)):
            os.mkdir('/local/home/david/Downloads/val/' + str(i))

    for line in open(path_label, 'r'):
        [img, label] = line.replace('\n', '').split(' ')
        shutil.copy(path + img, path + label + '/' + img)

if __name__ == '__main__':
    func()
