# -*- coding: utf-8 -*-

import os
import string
import tensorflow as tf
from PIL import Image


def create_record(srcDir):
    """Helper to create database
    """
    train_writer = tf.python_io.TFRecordWriter("../train_imgs.tfrecords")
    test_writer = tf.python_io.TFRecordWriter("../test_imgs.tfrecords")
    
    classes = os.listdir(srcDir)
    print classes
    for index, name in enumerate(classes):
        class_path = srcDir + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            image = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))

            image_id = string.atoi(img_name[0:len(img_name)-4])
            print image_id

            if image_id <= 20000:
                train_writer.write(example.SerializeToString())
            else:
                test_writer.write(example.SerializeToString())

        print img_path, index
        
    train_writer.close()
    test_writer.close()
