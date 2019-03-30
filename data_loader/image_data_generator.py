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
import numpy as np
import pickle
import os


class ImageDataGenerator(DataGenerator):
    def __init__(self):
        pass

    @staticmethod
    def parse_image_train(file_name, label, dataset_name, means_tensor, stds_tensor, n_classes):
        image_string = tf.read_file(file_name)

        image = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32) / 255.
        if dataset_name in ['gtsrb', 'omniglot', 'svhn', 'daimlerpedcls', 'mnist']:
            # image = tf.image.resize_image_with_crop_or_pad(image, 72, 72)
            image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        else:
            # image = tf.random_crop(image, size=[64, 64, 3])
            image = tf.random_crop(image, size=[224, 224, 1])
            # flip the image with the probability of 0.5
            image = tf.image.random_flip_left_right(image)
        image = (image - means_tensor) / stds_tensor

        label = tf.one_hot(indices=label, depth=1000)
        return image, label

    @staticmethod
    def parse_image_val(file_name, label, dataset_name, means_tensor, stds_tensor, n_classes):
        image_string = tf.read_file(file_name)
        image = tf.cast(tf.image.decode_jpeg(image_string), dtype=tf.float32) / 255.
        if dataset_name in ['gtsrb', 'omniglot', 'svhn', 'daimlerpedcls']:
            image = tf.image.resize_image_with_crop_or_pad(image, 72, 72)
        else:
            # image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
            image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        image = (image - means_tensor) / stds_tensor

        # label = tf.one_hot(indices=label, depth=n_classes)
        label = tf.one_hot(indices=label, depth=1000)
        return image, label

    @staticmethod
    def load_dataset(batch_size, cpu_cores, dataset_name, imgs_path, data_file='../datasets/decathlon_mean_std.pickle'):
        handler = open(data_file, 'rb')
        dict_mean_std = pickle.load(handler, encoding='bytes')

        means = np.array(dict_mean_std[bytes(dataset_name + 'mean', encoding='utf-8')], dtype=np.float32)
        means_tensor = tf.constant(np.expand_dims(np.expand_dims(means, axis=0), axis=0))
        stds = np.array(dict_mean_std[bytes(dataset_name + 'std', encoding='utf-8')], dtype=np.float32)
        stds_tensor = tf.constant(np.expand_dims(np.expand_dims(stds, axis=0), axis=0))

        imgs_path_train = imgs_path + 'train/'
        imgs_path_val = imgs_path + 'val/'
        classes_path_list_train = np.array([imgs_path_train + x + '/' for x in sorted(os.listdir(imgs_path_train))])
        classes_path_list_val = np.array([imgs_path_val + x + '/' for x in sorted(os.listdir(imgs_path_val))])
        n_classes = len(classes_path_list_train)

        filepath_list_train = []
        labels_train = []
        for i in range(len(classes_path_list_train)):
            classes_path = classes_path_list_train[i]
            # the path of the images under this label directory
            tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
            # the file path of all the training images
            filepath_list_train += tmp
            # the labels of all the training images (the labels are encoded by the order in the list)
            labels_train += (np.ones(len(tmp)) * i).tolist()

        filepath_list_train = np.array(filepath_list_train)
        labels_train = np.array(labels_train, dtype=np.int32)

        n_sample_train = len(labels_train)
        total_batches_train = n_sample_train // batch_size + 1

        filepath_list_val = []
        labels_val = []
        for i in range(len(classes_path_list_val)):
            classes_path = classes_path_list_val[i]
            tmp = [classes_path + x for x in sorted(os.listdir(classes_path))]
            filepath_list_val += tmp
            labels_val += (np.ones(len(tmp)) * i).tolist()

        filepath_list_val = np.array(filepath_list_val)
        labels_val = np.array(labels_val, dtype=np.int32)

        n_samples_val = len(labels_val)

        file_paths_train = tf.constant(filepath_list_train)
        labels_train = tf.constant(labels_train)

        file_paths_val = tf.constant(filepath_list_val)
        labels_val = tf.constant(labels_val)

        # % construct input pipeline
        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=100000)

        dataset_train = dataset_train.map(
            map_func=lambda x, y: ImageDataGenerator.parse_image_train(x, y, dataset_name, means_tensor, stds_tensor,
                                                                       n_classes), num_parallel_calls=cpu_cores)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(buffer_size=1)

        dataset_hessian = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
        dataset_hessian = dataset_hessian.shuffle(buffer_size=100000)
        dataset_hessian = dataset_hessian.map(
            map_func=lambda x, y: ImageDataGenerator.parse_image_val(x, y, dataset_name, means_tensor, stds_tensor,
                                                                     n_classes), num_parallel_calls=cpu_cores)
        dataset_hessian = dataset_hessian.batch(batch_size)
        dataset_hessian = dataset_hessian.prefetch(buffer_size=1)

        dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))
        dataset_val = dataset_val.map(
            map_func=lambda x, y: ImageDataGenerator.parse_image_val(x, y, dataset_name, means_tensor, stds_tensor,
                                                                     n_classes), num_parallel_calls=cpu_cores)
        dataset_val = dataset_val.batch(batch_size)
        dataset_val = dataset_val.prefetch(buffer_size=1)

        return dataset_train, dataset_val, dataset_hessian, total_batches_train, n_sample_train, n_samples_val


if __name__ == '__main__':
    handler = open('../datasets/decathlon_mean_std.pickle', 'rb')
    dict_mean_std = pickle.load(handler, encoding='bytes')
    print(dict_mean_std)
