#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:59:39 2018

@author: wsw
"""

# make tfrecords

import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_tfrecords_example(image,label=None,training=True):
    tfrecords_example = {}
    tfrecords_example['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
    if not training:
        tfrecords_example['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
    example = tf.train.Example(features=tf.train.Features(feature=tfrecords_example))
    serialized_example = example.SerializeToString()
    return serialized_example

def make_tfrecords(dataDir,dataFileName,labelFileName=None,training=True):
    dataPath = os.path.join(dataDir,dataFileName)
    allDatas = np.load(dataPath)
    total_num = len(allDatas)
    if labelFileName is not None:
        labelPath = os.path.join(dataDir,labelFileName)
        labels = np.load(labelPath)
    dataType = os.path.splitext(dataFileName)[0]
    
    if training:
        step = 0
        # TFRecord writer
        writer = tf.python_io.TFRecordWriter(os.path.join(dataDir,dataType+'_train.tfrecords'))
        print('----Making Training %s TFRecords----'%dataType)
        for image in allDatas:
            serialized_example = get_tfrecords_example(image)
            writer.write(serialized_example)
            step += 1
            print('\r>>>Step:{:5d}/Total:{:5d}'.format(step,total_num),end='',flush=True)
        writer.close()
        print('\n----Making Training TFRecords Finished!!!')
    else:
        writer = tf.python_io.TFRecordWriter(os.path.join(dataDir,dataType+'_test.tfrecords'))
        print('----Making Testing %s TFRecords----'%dataType)
        for i in range(total_num):
            image = allDatas[i]
            label = labels[i]
            serialized_example = get_tfrecords_example(image,label=label,training=False)
            writer.write(serialized_example)
            print('\r>>>Step:{:5d}/Total:{:5d}'.format(i+1,total_num),end='',flush=True)
        writer.close()
        print('\n----Making Testing TFRecords Finished!!!')

if __name__ == '__main__':
    dataType = 'liberty'
    dataDir = './brownData/%s_triplet_patches_data'%dataType
    dataFileName = '%s_100k_triplet_patch_image.npy'%dataType
    # labelFileName = '%s_100k_patch_pairs_labels.npy'%dataType
    labelFileName = None
    make_tfrecords(dataDir,dataFileName,labelFileName=labelFileName,training=True)
