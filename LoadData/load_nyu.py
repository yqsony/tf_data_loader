"""
load NYU dataset in tensorflow
Quan Yuan
2020-04-31
"""
import tensorflow as tf
import time
import numpy
import h5py
import keras.datasets.fashion_mnist
import random

def load_v2_raw_data(data_path):
    image_list = []
    depth_list = []
    with h5py.File(data_path, 'r') as f:
        # print list(f.keys())
        raw_depths = numpy.array(f['rawDepths'])
        images = numpy.array(f['images'])
        for image, depth in zip(images, raw_depths):
            depth = numpy.expand_dims(depth, 0)
            image_list.append(image.astype(numpy.float32))
            depth_list.append(depth)
    return image_list, depth_list


def generate_v2_data(data_path):
    image_list, depth_list = load_v2_raw_data(data_path)
    while True:
        index = random.randint(0, len(image_list)-1)
        yield image_list[index], depth_list[index]


def generator_v2(generate_v2_data, output_shape):
    dataset = tf.data.Dataset.from_generator(generate_v2_data, output_types=tf.float32, output_shapes=output_shape, )
    return dataset