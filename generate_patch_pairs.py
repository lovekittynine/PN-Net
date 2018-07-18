#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:53:45 2018

@author: wsw
"""
# generate patch pairs

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

dataType = 'liberty'

data_dir = '/media/wsw/资料/Brown_dataset/%s'%dataType

match_info_file = 'm50_10000_10000_0.txt'

patch_names = sorted(glob.glob(data_dir+'/*.bmp'))

preprocessed_data_dir = './%s_test_patch_pairs'%dataType

if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)

def generate_patch(match_file):
    data = np.loadtxt(os.path.join(data_dir,match_file),dtype=int,usecols=[0,1,3,4])
    patches_ID = data[:,[0,2]]
    Point_ID = data[:,[1,3]]
    # generate label
    Y = np.equal(Point_ID[:,0],Point_ID[:,1])
    Y = np.where(Y==True,1,0)
    nums = len(Point_ID)
    patch_images = np.zeros(shape=(nums,64,64,2),dtype=np.uint8)
    for i,patch_id in enumerate(patches_ID):
        id1 = patch_id[0]
        id2 = patch_id[1]
        # print('original patch id1:%d--patch id2:%d'%(id1,id2))
        img1_id,offset1 = divmod(id1,256)
        img2_id,offset2 = divmod(id2,256)
        # patch coordinate in a full image block
        x1,y1 = divmod(offset1,16)
        x2,y2 = divmod(offset2,16)
        #print('img_id1',img1_id,'x1:%d,y1:%d'%(x1,y1))
        #print('img_id2',img2_id,'x2:%d,y2:%d'%(x2,y2))
        # imread a specific image block
        img1 = plt.imread(patch_names[img1_id],0)
        img2 = plt.imread(patch_names[img2_id],0)
        # channel concatenate
        patch1 = np.expand_dims(img1[x1*64:(x1+1)*64,y1*64:(y1+1)*64],axis=2)
        patch2 = np.expand_dims(img2[x2*64:(x2+1)*64,y2*64:(y2+1)*64],axis=2)
        img_pair = np.dstack((patch1,patch2))
        patch_images[i,:,:,:] = img_pair
        
        #print(img_pair.shape)
        #plt.imshow(img_pair)
        #plt.show()
        view_bar(i+1,nums)
        
        
    print('\nStarting to save patch images and labels')
    np.save(os.path.join(preprocessed_data_dir,'%s_10k_patch_pairs_image.npy'%dataType),
            patch_images)
    np.save(os.path.join(preprocessed_data_dir,'%s_10k_patch_pairs_labels.npy'%dataType),
            Y)
    print('Finished!!!')
   


def view_bar(step,total_nums):
    rate = step/total_nums
    rate_num = int(rate*40)
    r = '\r[%s%s]%d%%\t step-%d/%d'%('>'*rate_num,'-'*(40-rate_num),rate*100,step,total_nums)
    print(r,end='',flush=True)
    
    
if __name__ == '__main__':
    generate_patch(match_info_file)
