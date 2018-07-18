#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:48:33 2018

@author: wsw
"""

# PN-Net construct 
# PN-Net contain three branches which they are shared weights

import tensorflow as tf
slim = tf.contrib.slim

def build_model(inputs,is_training=True):
    '''
    Note:In order to share variable operation must set opName
    '''
    init = tf.contrib.layers.xavier_initializer_conv2d()
    batchnorm_param = {'decay':0.9,
                       'scale':True,
                       'is_training':is_training,
                       'updates_collections':None,
                       'zero_debias_moving_mean':True}
    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.tanh,
                        weights_initializer=init,
                        biases_initializer=init,
                        weights_regularizer=slim.l2_regularizer(1e-6)
                        ):
        with slim.arg_scope([slim.conv2d],
                            stride=1,
                            padding='VALID',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batchnorm_param
                            ):
            net = slim.conv2d(inputs,num_outputs=32,kernel_size=(7,7),scope='conv1')
            net = slim.max_pool2d(net,kernel_size=[2,2],scope='max_pool')
            net = slim.conv2d(net,num_outputs=64,kernel_size=[6,6],scope='conv2')
            # check if output shape is [batch,8,8,64]
            # print(net.shape)
            net = slim.flatten(net,scope='flatten')
            net = slim.fully_connected(net,num_outputs=256,scope='fc')
            return net
    
if __name__ == '__main__':
    xs = tf.random_normal(shape=[10,32,32,1])
    with tf.variable_scope('branch1'):
        branch_out = build_model(xs)
    print(branch_out.shape)