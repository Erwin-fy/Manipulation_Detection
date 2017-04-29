# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from PIL import Image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 25000


def read_and_decode(filename):
    """Helper to read image and label

    Args:
        filename: database's name
    
    Returns:
        Variable Tensor
    """

    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [227, 227, 1])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)

    return image, label

def train_inputs(data_dir, batch_size):
    image, label = read_and_decode(data_dir)

    min_fraction_of_examples_in_queue = 0.9
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    
    num_preprocess_threads = 16

     #使用shuffle_batch可以随机打乱输入
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size, 
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    return image_batch, label_batch

def test_inputs(data_dir, batch_size):
    image, label = read_and_decode(data_dir)

    min_fraction_of_examples_in_queue = 0.9
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TEST *
                           min_fraction_of_examples_in_queue)
    
    num_preprocess_threads = 16

    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size, 
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
        
    return image_batch, label_batch
