#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:11:59 2018

@author: ws
"""

# PNNet test fpr95 value

import tensorflow as tf
from PN_Net import build_model
from Dataset import make_test_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import os

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test():
    # test dataset configuration
    testdataType = 'yosemite'
    test_dataDir = './brownData/%s_test_patch_pairs'%testdataType
    test_dataName = '%s_100k_patch_pairs_image_test.tfrecords'%testdataType
    epoch = 1
    batchsize=256
    # create test dataset
    test_dataset = make_test_dataset(dataDir=test_dataDir,
                                     dataName=test_dataName,
                                     epoch=epoch,
                                     batchsize=batchsize)
    test_iter = test_dataset.make_one_shot_iterator()
    # get next batch
    imgs_batch,labels_batch = test_iter.get_next()
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32,shape=[None,32,32,2])
    # build model
    with tf.name_scope('PN_Net'):
        pair_distance = branch_output(xs)
    
    # model configuration
    with tf.name_scope('aux_config'):
        restore = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        
    with tf.Session(config=config) as sess:
        ckpt = './model/BN-soft-model.ckpt-780000'
        print('>>>Loading model from %s'%ckpt)
        restore.restore(sess,ckpt)
        print('>>>Loading model from %s finished!!!'%ckpt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        testNums = 100000
        totalSteps = (testNums//256)+1
        try:
            step = 0
            labels = []
            distances = []
            print('>>>Testing %s dataset'%testdataType)
            while not coord.should_stop():
                imgs,labs = sess.run([imgs_batch,labels_batch])
                batch_dist = sess.run(pair_distance,feed_dict={xs:imgs})
                step += 1
                labels.extend(labs.tolist())
                distances.extend(batch_dist.tolist())
                print('\r>>>Step:{:04d}/Total:{:04d}'.format(step,totalSteps),end=' ',flush=True)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            print('\n>>>Testing Finished!!!')
            compute_valid_roc(labels,distances)
        coord.join(threads)

      
def branch_output(xs):
    patch1,patch2 = tf.split(xs,num_or_size_splits=2,axis=-1)
    with tf.variable_scope('branch1'):
        descriptor1 = build_model(patch1)
        tf.get_variable_scope().reuse_variables()
        descriptor2 = build_model(patch2)
        pair_distance = tf.sqrt(tf.reduce_sum(tf.square(descriptor1-descriptor2),axis=-1))
        return pair_distance      
       
def compute_valid_roc(labels,distance):
    '''
    Note:this a metric learning so match pairs have more little distance
         non-match pairs have more bigger distance,but standard roc compute
         assume better match have better scores
         so we need to modify our socre using maximum of score subtract all
         score
    '''
    labels = np.uint8(labels)
    pos_index = np.where(labels==1)
    pos_distance = np.array(distance)[pos_index[0]]
    neg_index = np.where(labels==0)
    neg_distance = np.array(distance)[neg_index[0]]
    plt.figure()
    plt.plot(list(range(50000)),pos_distance,'r*',label='match-distance')
    plt.plot(list(range(50000)),neg_distance,'gp',label='nonmatch-distance')
    plt.legend()
    
    # reverse distance to reassure match pair have bigger score
    # non-macth pair have little score
    reverse_dist = np.max(distance)-distance
    auc = roc_auc_score(labels,reverse_dist)
    fpr,tpr,thresholds = roc_curve(labels,reverse_dist)
    index = np.argmin(np.abs(tpr-0.95))
    fpr95 = fpr[index]
    print('>>>@fpr95:%.5f'%fpr95,'auc:%.3f'%auc)
    plt.figure()
    plt.plot(fpr,tpr,'r-')
    plt.title('ROC-curve')
    plt.show()


if __name__ == '__main__':
    test()
