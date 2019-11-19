# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: ImageDataGenerator
@time: 2019-03-27 20:38

Description. 
"""

from data_loader.data_generator import DataGenerator

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import json
import os


class ImageDataGenerator(DataGenerator):
    def __init__(self):
        pass

    @staticmethod
    def parse_image_train(file_name, label, dataset_name, means_tensor, stds_tensor, n_classes, resize_dict):
        # Load
        image_string = tf.read_file(file_name)
        channel_size = 3
        if dataset_name in ['mnist']:
            channel_size = 1
        image = tf.cast(tf.image.decode_jpeg(image_string, channels=channel_size), dtype=tf.float32)

        # Resize
        resize_length = resize_dict.get(dataset_name, 64)
        image = tf.image.resize_image_with_crop_or_pad(image, resize_length, resize_length)

        if dataset_name in ['celeba', 'deepfashion', 'lfw']:
            image = tf.reverse(image, axis=[-1])

        if dataset_name not in ['imagenet12', 'imagenet12_large', 'celeba', 'deepfashion', 'lfw']:
            image = image / 255.

        if dataset_name not in ['mnist', 'cifar10']:
            image = (image - means_tensor) / stds_tensor

        if dataset_name not in resize_dict.keys():
            # flip the image with the probability of 0.5
            image = tf.image.random_flip_left_right(image)

        if type(label) == int:
            label = tf.one_hot(indices=label, depth=n_classes)

        return image, label

    @staticmethod
    def parse_image_val(file_name, label, dataset_name, means_tensor, stds_tensor, n_classes, resize_dict):
        # Load
        image_string = tf.read_file(file_name)
        channel_size = 3
        if dataset_name in ['mnist']:
            channel_size = 1
        image = tf.cast(tf.image.decode_jpeg(image_string, channels=channel_size), dtype=tf.float32)

        # Resize
        resize_length = resize_dict.get(dataset_name, 64)
        image = tf.image.resize_image_with_crop_or_pad(image, resize_length, resize_length)

        if dataset_name in ['celeba', 'deepfashion', 'lfw']:
            image = tf.reverse(image, axis=[-1])

        if dataset_name not in ['imagenet12', 'imagenet12_large', 'celeba', 'deepfashion', 'lfw']:
            image = image / 255.

        if dataset_name not in ['mnist', 'cifar10']:
            image = (image - means_tensor) / stds_tensor

        if type(label) == int:
            label = tf.one_hot(indices=label, depth=n_classes)

        return image, label

    @staticmethod
    def load_dataset(cfg, data_file='../dataset/datasets_mean_std.pickle'):
        def get_partition(path_partition, n_samples):
            if os.path.exists(path_partition):
                partition_list = pickle.load(open(path_partition, 'rb'))
            else:
                partition_list = np.ones(n_samples)
                partition_list[np.random.choice(len(filepath_list), size=len(filepath_list) // 5, replace=False)] = 0
                pickle.dump(partition_list, open(path_partition, 'wb'))
            return partition_list

        dataset_name = cfg['data']['dataset_name']
        path_partition = cfg['data']['path_partition']
        path_data = ''

        # 数据的组织形式不同
        if dataset_name.startswith('celeba'):
            # Data is stored in one directory
            filepath_list = np.array(
                ['/local/home/david/Datasets/celeba/img_align_celeba/' + x for x in
                 sorted(os.listdir('/local/home/david/Datasets/celeba/img_align_celeba/'))])

            # Read labels according to dataset_name
            if dataset_name == 'celeba1':
                labels_all = pd.read_csv('/local/home/david/Datasets/celeba/list_attr_celeba.txt',
                                         delim_whitespace=True, header=None).values[:, 1:21]
            elif dataset_name == 'celeba2':
                labels_all = pd.read_csv('/local/home/david/Datasets/celeba/list_attr_celeba.txt',
                                         delim_whitespace=True, header=None).values[:, 21:]
            elif dataset_name == 'celeba':
                labels_all = pd.read_csv('/local/home/david/Datasets/celeba/list_attr_celeba.txt',
                                         delim_whitespace=True, header=None).values[:, 1:]
            labels_list = np.array(labels_all, dtype=np.float32)

            # Split train and test dataset
            partition_list = get_partition(path_partition, len(filepath_list))

            n_classes = labels_list.shape[1]

            filepath_list_train = filepath_list[partition_list == 1]
            filepath_list_val = filepath_list[partition_list == 0]

            labels_train = labels_list[partition_list == 1]
            labels_val = labels_list[partition_list == 0]

            dataset_name = 'celeba'

        elif dataset_name.startswith('lfw'):
            # Read paths of images
            imgs_labels = pd.read_csv('/local/scratch/labels_deepfunneled.txt', delim_whitespace=True,
                                      header=None).values
            filepath_list = np.array(imgs_labels[:, 0])

            if dataset_name == 'lfw1':
                labels_all = imgs_labels[:, 1:21]
            elif dataset_name == 'lfw2':
                labels_all = imgs_labels[:, 21:41]
            elif dataset_name == 'lfw':
                labels_all = imgs_labels[:, 1:41]
            else:
                labels_all = imgs_labels[:, json.loads(cfg['data']['index'])]

            labels_list = np.array(labels_all, dtype=np.float32)

            # Split train and test dataset
            partition_list = get_partition(path_partition, len(filepath_list))

            n_classes = labels_list.shape[1]

            filepath_list_train = filepath_list[partition_list == 1]
            filepath_list_val = filepath_list[partition_list == 0]

            labels_train = labels_list[partition_list == 1]
            labels_val = labels_list[partition_list == 0]

            # 这里是用来指导下一步如何做图片处理的
            dataset_name = 'lfw'

        elif dataset_name.startswith('deepfashion'):

            content = pd.read_csv('/local/home/david/Datasets/deepfashion/labels_10_train_val_path.txt', sep=' ',
                                  header=None).values

            filepath_list = content[:, 0]

            if dataset_name == 'deepfashion1':
                labels_all = content[:, 1:6]
            elif dataset_name == 'deepfashion2':
                labels_all = content[:, 6:11]
            elif dataset_name == 'deepfashion':
                labels_all = content[:, 1:]

            labels_list = np.array(labels_all, dtype=np.float32)

            # Split train and test dataset
            partition_list = get_partition(path_partition, len(filepath_list))
            n_classes = labels_list.shape[1]

            filepath_list_train = filepath_list[partition_list == 1]
            filepath_list_val = filepath_list[partition_list == 0]

            labels_train = labels_list[partition_list == 1]
            labels_val = labels_list[partition_list == 0]

            dataset_name = 'deepfashion'
            path_data = '/local/home/david/Datasets/deepfashion/img/'
        # 这段代码是对imagenet用的
        # else:
        #     # Data is stored in directories of train and val
        #     imgs_path_train = imgs_path + 'train/'
        #     imgs_path_val = imgs_path + 'val/'
        #     if dataset_name == 'imagenet12':
        #         classes_path_list_train = np.array([imgs_path_train + str(x) + '/' for x in range(1000)])
        #         classes_path_list_val = np.array([imgs_path_val + str(x) + '/' for x in range(1000)])
        #     else:
        #         classes_path_list_train = np.array(
        #             [imgs_path_train + x + '/' for x in sorted(os.listdir(imgs_path_train))])
        #         classes_path_list_val = np.array([imgs_path_val + x + '/' for x in sorted(os.listdir(imgs_path_val))])
        #     n_classes = len(classes_path_list_train)
        #
        #     filepath_list_train = []
        #     labels_train = []
        #     for i in range(len(classes_path_list_train)):
        #         classes_path = classes_path_list_train[i]
        #         # the path of the images under this label directory
        #         tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
        #         # the file path of all the training images
        #         filepath_list_train += tmp
        #         # the labels of all the training images (the labels are encoded by the order in the list)
        #         labels_train += (np.ones(len(tmp)) * i).tolist()
        #
        #     filepath_list_train = np.array(filepath_list_train)
        #     labels_train = np.array(labels_train, dtype=np.int32)
        #
        #     filepath_list_val = []
        #     labels_val = []
        #     for i in range(len(classes_path_list_val)):
        #         classes_path = classes_path_list_val[i]
        #         tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
        #         filepath_list_val += tmp
        #         labels_val += (np.ones(len(tmp)) * i).tolist()
        #
        #     filepath_list_val = np.array(filepath_list_val)
        #     labels_val = np.array(labels_val, dtype=np.int32)

        # Get mean and std values
        handler = open(data_file, 'rb')
        dict_mean_std = pickle.load(handler)

        means = np.array(dict_mean_std[dataset_name + 'mean'], dtype=np.float32)
        means_tensor = tf.constant(np.expand_dims(np.expand_dims(means, axis=0), axis=0))
        stds = np.array(dict_mean_std[dataset_name + 'std'], dtype=np.float32)
        stds_tensor = tf.constant(np.expand_dims(np.expand_dims(stds, axis=0), axis=0))

        # Get number of samples
        n_sample_train = len(labels_train)
        total_batches_train = n_sample_train // cfg['basic'].getint('batch_size') + 1
        n_samples_val = len(labels_val)

        # Build tf graph
        file_paths_train = tf.constant(filepath_list_train)
        labels_train = tf.constant(labels_train)

        file_paths_val = tf.constant(filepath_list_val)
        labels_val = tf.constant(labels_val)

        resize_dict = {
            'imagenet12': 224,
            'imagenet12_large': 224,
            'celeba': 72,
            'lfw': 72,
            'gtsrb': 72,
            'omniglot': 72,
            'svhn': 72,
            'daimlerpedcls': 72,
            'deepfashion': 72,
            'cifar10': 32
        }

        # % construct input pipeline
        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=100000)

        dataset_train = dataset_train.map(
            map_func=lambda x, y: ImageDataGenerator.parse_image_train(path_data + x, y, dataset_name, means_tensor,
                                                                       stds_tensor,
                                                                       n_classes, resize_dict),
            num_parallel_calls=cfg['basic'].getint('cpu_cores'))
        dataset_train = dataset_train.batch(cfg['basic'].getint('batch_size'))
        dataset_train = dataset_train.prefetch(buffer_size=1)

        dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))

        dataset_val = dataset_val.map(
            map_func=lambda x, y: ImageDataGenerator.parse_image_val(path_data + x, y, dataset_name, means_tensor,
                                                                     stds_tensor,
                                                                     n_classes, resize_dict),
            num_parallel_calls=cfg['basic'].getint('cpu_cores'))
        dataset_val = dataset_val.batch(cfg['basic'].getint('batch_size'))
        dataset_val = dataset_val.prefetch(buffer_size=1)

        return dataset_train, dataset_val, total_batches_train, n_sample_train, n_samples_val

# if __name__ == '__main__':
#     handler = open('../dataset/decathlon_mean_std.pickle', 'rb')
#     dict_mean_std = pickle.load(handler, encoding='bytes')
#     print(dict_mean_std)
