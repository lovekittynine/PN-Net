#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:40:59 2018

@author: wsw
"""

# create dataset

import tensorflow as tf
import os

tf.reset_default_graph()

def parse_train_example(serialized_example):
    example = tf.parse_single_example(serialized_example,
                                      features={'image':tf.FixedLenFeature([],tf.string)})
    image = example['image']
    image = tf.decode_raw(image,out_type=tf.uint8)
    image = tf.reshape(image,shape=[64,64,3])
    # PNNet require input size 32*32*1
    image = tf.image.resize_images(image,size=[32,32])
    # simple normalization 
    image = image/255.0
    # image = (image-0.487)/0.189
    return image

# test or valid
def parse_test_example(serialized_example):
    features={'image':tf.FixedLenFeature([],tf.string),
              'label':tf.FixedLenFeature([],tf.float32)}  
    example = tf.parse_single_example(serialized_example,
                                      features=features)
    image = example['image']
    image = tf.decode_raw(image,out_type=tf.uint8)
    image = tf.reshape(image,shape=[64,64,2])
    # PNNet require input size 32*32*1
    image = tf.image.resize_images(image,size=[32,32])
    # simple normalization 
    image = image/255.0
    # image = (image-0.487)/0.189
    label = example['label']
    return image,label

def make_train_dataset(dataDir,dataName,epoch=30,batchsize=128,buffer=100000):
    tfrecords_path = os.path.join(dataDir,dataName)
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_train_example,num_parallel_calls=4)
    # create train dataset
    dataset = dataset.repeat(epoch).shuffle(buffer).batch(batchsize)
    # dataset = dataset.repeat(epoch).batch(batchsize)
    return dataset

def make_test_dataset(dataDir,dataName,epoch=1,batchsize=128,buffer=10000):
    tfrecords_path = os.path.join(dataDir,dataName)
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_test_example,num_parallel_calls=4)
    # create test dataset
    dataset = dataset.repeat(epoch).shuffle(buffer).batch(batchsize)
    return dataset

def test_dataset():
    """
    This function is to test generated dataset
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    dataDir = './brownData/liberty_triplet_patches_data'
    dataName = 'liberty_500k_triplet_patch_image_train.tfrecords'
    # create dataset
    dataset = make_train_dataset(dataDir,dataName,epoch=1,batchsize=1000)
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    step = 0
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                images_batch = sess.run(images)
                step += 1
                mean = np.mean(images_batch)
                print('step',step,mean)
                # print('Image Shape:',images_batch.shape)
#                for i in range(128):
#                    plt.figure()
#                    img = images_batch[i]
#                    img = np.concatenate([img[:,:,0],img[:,:,1],img[:,:,2]],axis=1)
#                    plt.imshow(img)
#                    plt.show()
#                coord.request_stop()
#                coord.join(threads)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    test_dataset()
