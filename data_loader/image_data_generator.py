# encoding: utf-8
"""

@version: 1.0
@license: Apache Licence
@file: ImageDataGenerator
@time: 2019-03-27 20:38

Description. 
"""

from data_loader.data_generator import DataGenerator
from utils.json import read_l
from utils.json import read_i
from utils.json import read_s

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os


class ImageDataGenerator(DataGenerator):
    def __init__(self):
        pass

    @staticmethod
    def parse_img_train(file_name, label, n_classes, channels, h, w, process, mean_tensor, std_tensor):
        # Load
        img_string = tf.read_file(file_name)
        img = tf.cast(tf.image.decode_jpeg(img_string, channels=channels), dtype=tf.float32)
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)

        # process
        if 'reverse' in process:
            img = tf.reverse(img, axis=[-1])
        if 'normalize_value' in process:
            img = img / 255.
        if 'normalize_distribution' in process:
            img = (img - mean_tensor) / std_tensor
        if 'flip' in process:
            img = tf.image.random_flip_left_right(img)
        if type(label) == int:
            label = tf.one_hot(indices=label, depth=n_classes)
        return img, label

    @staticmethod
    def parse_img_val(file_name, label, n_classes, channels, h, w, process, mean_tensor, std_tensor):
        # Load
        img_string = tf.read_file(file_name)
        img = tf.cast(tf.image.decode_jpeg(img_string, channels=channels), dtype=tf.float32)
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)

        # process
        if 'reverse' in process:
            img = tf.reverse(img, axis=[-1])
        if 'normalize_value' in process:
            img = img / 255.
        if 'normalize_distribution' in process:
            img = (img - mean_tensor) / std_tensor
        if type(label) == int:
            label = tf.one_hot(indices=label, depth=n_classes)
        return img, label

    @staticmethod
    def load_dataset(cfg):
        def get_partition(path_partition, n_samples):
            if os.path.exists(path_partition):
                partition_list = pickle.load(open(path_partition, 'rb'))
            else:
                partition_list = np.ones(n_samples)
                partition_list[np.random.choice(len(filepath_list), size=len(filepath_list) // 5, replace=False)] = 0
                pickle.dump(partition_list, open(path_partition, 'wb'))
            return partition_list

        data_name = cfg['data']['name']
        path_data = cfg['path']['path_dataset']

        # 处理labels
        str_labels = read_s(cfg, 'task', 'labels')
        str_labels = str_labels.replace('[', '').replace(']', '')
        labels = list()
        for item in str_labels.split(','):
            if item.count('-') > 0:
                start, end = item.split('-')
                # TODO: 要不要加1，lfw需要，其他的不清楚还,fashionmnist也需要
                labels += [_ + 1 for _ in range(int(start), int(end))]
            else:
                labels.append(int(item) + 1)
        labels.sort()

        if data_name == 'celeba':
            path_img = path_data + 'img_align_celeba/'
            path_label = path_data + 'list_attr_celeba.txt'
            # label范围, 根据task_name来决定

            filepath_list = np.array([path_img + x for x in sorted(os.listdir(path_img))])
            labels_list = np.array(pd.read_csv(path_label, delim_whitespace=True, header=None).values[:, labels],
                                   dtype=np.float32)
        elif data_name == 'lfw':
            # Read paths of images
            data = pd.read_csv('/local/scratch/labels_deepfunneled.txt', delim_whitespace=True, header=None).values
            # labels

            filepath_list = np.array(data[:, 0])
            labels_list = np.array(data[:, labels], dtype=np.float32)
        elif data_name == 'fashionmnist':
            path_img = path_data + 'labels.csv'
            data = pd.read_csv(path_img, delim_whitespace=True, header=None).values

            filepath_list = np.array([path_data + 'img/' + x for x in data[:, 0]])
            labels_list = np.array(data[:, labels], dtype=np.float32)
        elif data_name == 'deepfashion':
            data = pd.read_csv('/local/home/david/Datasets/deepfashion/labels_1000_train_val.txt',
                               delim_whitespace=True, header=None).values

            filepath_list = np.array([path_data + x for x in data[:, 0]])
            labels_list = np.array(data[:, labels], dtype=np.float32)

        # Split train and test dataset
        path_partition = cfg['data']['path_partition']
        partition_list = get_partition(path_partition, len(filepath_list))

        n_classes = read_i(cfg, 'data', 'n_classes')

        filepath_list_train = filepath_list[partition_list == 1]
        filepath_list_val = filepath_list[partition_list == 0]

        labels_train = labels_list[partition_list == 1]
        labels_val = labels_list[partition_list == 0]

        # Get number of samples
        n_sample_train = len(labels_train)
        total_batches_train = n_sample_train // read_i(cfg, 'train', 'batch_size')
        n_samples_val = len(labels_val)

        # Build tf graph
        file_paths_train = tf.constant(filepath_list_train)
        labels_train = tf.constant(labels_train)

        file_paths_val = tf.constant(filepath_list_val)
        labels_val = tf.constant(labels_val)

        # Process
        process = read_l(cfg, 'data', 'process')

        # Obtain mean and std
        mean = read_l(cfg, 'data', 'mean')
        std = read_l(cfg, 'data', 'std')
        mean_tensor = tf.constant(np.expand_dims(np.expand_dims(np.array(mean, dtype=np.float32), axis=0), axis=0))
        std_tensor = tf.constant(np.expand_dims(np.expand_dims(np.array(std, dtype=np.float32), axis=0), axis=0))

        # Obtain h and w
        h, w = read_l(cfg, 'data', 'length')
        channels = read_i(cfg, 'data', 'channels')

        # Construct input pipeline
        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=100000)

        dataset_train = dataset_train.map(
            map_func=lambda x, y: ImageDataGenerator.parse_img_train(x, y, n_classes, channels, h, w, process,
                                                                     mean_tensor, std_tensor),
            num_parallel_calls=read_i(cfg, 'basic', 'cpu_cores'))
        dataset_train = dataset_train.batch(read_i(cfg, 'train', 'batch_size'), drop_remainder=True)
        dataset_train = dataset_train.prefetch(buffer_size=1)

        dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
        dataset_val = dataset_val.map(
            map_func=lambda x, y: ImageDataGenerator.parse_img_val(x, y, n_classes, channels, h, w, process,
                                                                   mean_tensor, std_tensor),
            num_parallel_calls=read_i(cfg, 'basic', 'cpu_cores'))
        dataset_val = dataset_val.batch(read_i(cfg, 'train', 'batch_size'), drop_remainder=True)
        dataset_val = dataset_val.prefetch(buffer_size=1)

        return dataset_train, dataset_val, total_batches_train, n_sample_train, n_samples_val
