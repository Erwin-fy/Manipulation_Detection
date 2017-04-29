# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from PIL import Image


def create_record(directory):
    """Helper to create database
    """

    #当前工作目录
    cwd = os.getcwd()
    #当前目录上一级目录
    dir = os.path.dirname(cwd)
    print (dir)
    writer = tf.python_io.TFRecordWriter(dir + "/" + directory+".tfrecords")
    classes = os.listdir(dir + "/" + directory)
    print(classes)
    for index, name in enumerate(classes):
        class_path = dir + "/" + directory + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            #index = min(index, 1)
            #print(img_path, index)
            img = Image.open(img_path)
            image = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
            writer.write(example.SerializeToString())
        print(img_path, index)
        
    writer.close()
    
