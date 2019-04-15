import tensorflow as tf

import os
import scipy.io as scio
import shutil

if __name__ == '__main__':
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
