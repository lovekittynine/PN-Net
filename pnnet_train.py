#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 10:38:41 2018

@author: wsw
"""

# construct full PNNet to train

import tensorflow as tf
from PN_Net import build_model
from Dataset import make_train_dataset,make_test_dataset
import os
import numpy as np
import logging
from sklearn.metrics import roc_auc_score,roc_curve


tf.reset_default_graph()

# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def loggingInit(filename='./run_records.log'):
    # Global logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
                        datefmt='%Y/%m/%d-%H:%M:%S',
                        filename=filename,
                        filemode='w')
    # set stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s-%(message)s',
                                  datefmt='%Y/%m/%d-%H:%M:%S')
    console.setFormatter(formatter)
    # add handler to root logging
    logging.getLogger().addHandler(console)

def pnnet(inputs):
    patch1,patch2,patch3 = tf.split(inputs,num_or_size_splits=3,axis=-1)
    assert patch1.get_shape().as_list()[1:]==[32,32,1],'patch demension not matching'
    with tf.variable_scope('branch1'):
        branch1_out = build_model(patch1)
        # start shared variables
        tf.get_variable_scope().reuse_variables()
        branch2_out = build_model(patch2)
        branch3_out = build_model(patch3)
    return branch1_out,branch2_out,branch3_out
    

def PNSoft(branch1,branch2,branch3):
    '''
    compute pnsoft loss
    patch1,patch2 is match,patch3 is not matching with patch1 and patch2
    '''
    # compute p_distance match pairs distance
    p_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch2),axis=1))
    # clip distance less than 50.0
    p_distance = clip_distance(p_distance)
    # compute two non-match pair distance
    n1_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch3),axis=1))
    n2_distance = tf.sqrt(tf.reduce_sum(tf.square(branch2-branch3),axis=1))
    # select softnegative loss
    n_distance = tf.minimum(n1_distance,n2_distance)
    # n_distance = n1_distance
    # clip distance less than 50.0
    n_distance = clip_distance(n_distance)
    # compute loss
    pos_softmax = tf.exp(p_distance)/(tf.exp(p_distance)+tf.exp(n_distance))
    p_loss = tf.reduce_sum(tf.square(pos_softmax))
    neg_softmax = tf.exp(n_distance)/(tf.exp(p_distance)+tf.exp(n_distance))
    n_loss = tf.reduce_sum(tf.square(1-neg_softmax))
   
    tf.losses.add_loss(p_loss+n_loss)
    # get total loss
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('p_loss',p_loss)
    tf.summary.scalar('n_loss',n_loss)
    tf.summary.scalar('total_loss',total_loss)
    tf.summary.histogram('p_distance',p_distance)
    tf.summary.histogram('n_distance',n_distance)
    return total_loss,p_loss,n_loss


def PNSigmoid(branch1,branch2,branch3):
    '''
    compute pnsigmoid loss
    '''
    # compute p_distance match pairs distance
    p_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch2),axis=1))
    # clip distance less than 50.0
    p_distance = clip_distance(p_distance)
    # compute two non-match pair distance
    n1_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch3),axis=1))
    n2_distance = tf.sqrt(tf.reduce_sum(tf.square(branch2-branch3),axis=1))
    # select softnegative loss
    n_distance = tf.minimum(n1_distance,n2_distance)
    # clip distance less than 50.0
    n_distance = clip_distance(n_distance)
    # compute loss
    pos_sigmoid = tf.nn.sigmoid(p_distance)
    p_loss = tf.reduce_sum(tf.square(pos_sigmoid))
    neg_sigmoid = tf.nn.sigmoid(n_distance)
    n_loss = tf.reduce_sum(tf.square(1-neg_sigmoid))
    
    tf.losses.add_loss(p_loss+n_loss)
    # get total loss
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('p_loss',p_loss)
    tf.summary.scalar('n_loss',n_loss)
    tf.summary.scalar('total_loss',total_loss)
    tf.summary.histogram('p_distance',p_distance)
    tf.summary.histogram('n_distance',n_distance)
    return total_loss,p_loss,n_loss

def PNLog(branch1,branch2,branch3):
    '''
    compute pnlog loss
    L(T) = (log(dist(p1,p2)))^2+[C-log{min(dist(p1,n),dist(p2,n))}]^2
    1.Let C=2
    '''
    # compute p_distance match pairs distance
    p_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch2),axis=1))
    # clip distance in order not qual 0.0
    p_distance = tf.clip_by_value(p_distance,1e-2,tf.reduce_max(p_distance))
    # compute two non-match pair distance
    n1_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1-branch3),axis=1))
    n2_distance = tf.sqrt(tf.reduce_sum(tf.square(branch2-branch3),axis=1))
    # select softnegative loss
    n_distance = tf.minimum(n1_distance,n2_distance)
    # clip distance in order not qual 0.0
    n_distance = tf.clip_by_value(n_distance,1e-2,tf.reduce_max(n_distance))
    # compute loss
    p_loss = tf.reduce_sum(tf.square(tf.log(p_distance)))
    n_loss = tf.reduce_sum(tf.square(2-tf.log(n_distance)))
    
    tf.losses.add_loss(p_loss+n_loss)
    # get total loss
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('p_loss',p_loss)
    tf.summary.scalar('n_loss',n_loss)
    tf.summary.scalar('total_loss',total_loss)
    tf.summary.histogram('p_distance',p_distance)
    tf.summary.histogram('n_distance',n_distance)
    return total_loss,p_loss,n_loss
    
def clip_distance(x):
    '''
    compute exp(x) and clip value in order not to exceed tf.float32 cant express
    '''
    return tf.clip_by_value(x,0.0,50.0)

def summary_variables():
    varlist = tf.trainable_variables()
    for var in varlist:
        name = var.op.name
        tf.summary.histogram(name,var)


def compute_valid_roc(labels,distance):
    '''
    Note:this a metric learning so match pairs have more little distance
         non-match pairs have more bigger distance,but standard roc compute
         assume better match have better scores
         so we need to modify our socre using maximum of score subtract all
         score
    '''
    # reverse distance to reassure match pair have bigger score
    # non-macth pair have little score
    reverse_dist = np.max(distance)-distance
    auc = roc_auc_score(labels,reverse_dist)
    fpr,tpr,thresholds = roc_curve(labels,reverse_dist)
    index = np.argmin(np.abs(tpr-0.95))
    fpr95 = fpr[index]
    logging.info('Valid AUC:{:.3f} @fpr95:{:.5f}'.format(auc,fpr95))

   

def branch_output(xs):
    patch1,patch2 = tf.split(xs,num_or_size_splits=2,axis=-1)
    with tf.variable_scope('branch1',reuse=True):
        descriptor1 = build_model(patch1)
        descriptor2 = build_model(patch2)
        pair_distance = tf.sqrt(tf.reduce_sum(tf.square(descriptor1-descriptor2),axis=-1))
        return pair_distance
    
def train():
    
    # loggingInit
    loggingInit()
    # train dataset config
    traindataDir = './brownData/liberty_triplet_patches_data'
    traindataName = 'liberty_500k_triplet_patch_image_train.tfrecords'
    # valid  dataset config
    validdataDir = './brownData/liberty_test_patch_pairs'
    validdataName = 'liberty_10k_patch_pairs_image_valid.tfrecords'
    epoch = 200
    batchsize = 128
    # create train dataset
    train_dataset = make_train_dataset(dataDir=traindataDir,
                                       dataName=traindataName,
                                       epoch=epoch,
                                       batchsize=batchsize)
    train_iter = train_dataset.make_one_shot_iterator()
    # create valid dataset
    valid_dataset = make_test_dataset(dataDir=validdataDir,
                                      dataName=validdataName,
                                      epoch=None,
                                      batchsize=batchsize)
    valid_iter = valid_dataset.make_initializable_iterator()
    # valid dataset need to run initializer explictly
    valid_iter_init = valid_iter.initializer
    
    # get image batch
    img_batch = train_iter.get_next()
    # get valid image and label batch
    valid_images,valid_labels = valid_iter.get_next()
    
    with tf.name_scope('valid_inputs'):
        xs = tf.placeholder(tf.float32,shape=[None,32,32,2])
        
    # build network
    with tf.name_scope('PN_Net'):
        branch1_out,branch2_out,branch3_out = pnnet(img_batch)
        # valid one branch out
        pair_distance = branch_output(xs)
    # loss
    with tf.name_scope('losses'):
        # modify
        total_loss,p_loss,n_loss = PNSoft(branch1_out,branch2_out,branch3_out)
        
    with tf.name_scope('optimizer'):
        global_step = tf.train.create_global_step()
        lr = tf.train.exponential_decay(1e-3,
                                        global_step,
                                        decay_steps=50000,
                                        decay_rate=0.999)
        optimizer = tf.train.AdamOptimizer(lr)
        #grads = optimizer.compute_gradients(total_loss)
        #grads_update = [(tf.where(tf.is_nan(grad),tf.zeros(grad.shape),grad),var) for grad,var in grads]
        train_op = optimizer.minimize(total_loss,global_step)
        #train_op = optimizer.apply_gradients(grads_update,global_step)
    
    summary_variables()
    
    with tf.name_scope('aux_config'):
        saver = tf.train.Saver(max_to_keep=5)
        restore = tf.train.Saver()
        writer = tf.summary.FileWriter('./logs',tf.get_default_graph(),flush_secs=10)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        # merge op
        merge_op = tf.summary.merge_all()
    
    with tf.Session(config=sess_config) as sess:
        tf.global_variables_initializer().run()
        # restore model
        ckpt = tf.train.latest_checkpoint('./model')
        if ckpt:
            logging.info('Restore parameters from %s'%ckpt)
            restore.restore(sess,ckpt)
            logging.info('Loading Model Finished!!!')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            epoch = 1
            while not coord.should_stop():
                loss_value,pos_loss,neg_loss,_ = sess.run([total_loss,p_loss,n_loss,train_op])
                step = global_step.eval()
                fmt = 'Epoch[{:03d}]-Step:{:06d}-Ploss:{:6.4f}-Nloss:{:6.4f}-Total_Loss:{:6.4f}'.format(epoch,
                            step,
                            pos_loss,
                            neg_loss,
                            loss_value)
                if step%500 == 0:
                    logging.info(fmt)
                    summaries = sess.run(merge_op)
                    writer.add_summary(summaries,step)
                    
                if step%3900==0:
                    epoch += 1
                    saver.save(sess,'./model/model.ckpt',global_step=step,write_meta_graph=False)
                    # compute valid
                    distance = []
                    labels = []
                    # run iterator initializer
                    sess.run(valid_iter_init)
                    for i in range(79):
                        valid_imgs,valid_labs = sess.run([valid_images,valid_labels])
                        pair_dist = sess.run(pair_distance,feed_dict={xs:valid_imgs})
                        distance.extend(pair_dist.tolist())
                        labels.extend(valid_labs.tolist())
                    compute_valid_roc(labels,distance)
                        
                
        except tf.errors.OutOfRangeError:
            logging.info('Traing Finished!!!')
            coord.request_stop()
        coord.join(threads)


def main(argv=None):
    train()
    
    
if __name__ == '__main__':
    tf.app.run(main)
   
          
