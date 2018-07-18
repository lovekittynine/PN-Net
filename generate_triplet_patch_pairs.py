#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:33:19 2018

@author: wsw
"""

# prepare datasets
# construct triplet patches

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

dataName = 'notredame'
patchNums = 500000

data_dir = '/media/wsw/资料/Brown_dataset/%s'%dataName
match_info_file = 'm50_%d_%d_0.txt'%(patchNums,patchNums)
patch_names = sorted(glob.glob(data_dir+'/*.bmp'))
preprocessed_data_dir = './%s_triplet_patches_data'%dataName

print('----------------Generating %s----------------'%dataName)

if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)

def generate_patch(match_file):
    data = np.loadtxt(os.path.join(data_dir,match_file),dtype=int,usecols=[0,1,3,4])
    patches_ID = data[:,[0,2]]
    Point_ID = data[:,[1,3]]
    # get match index and non-match index
    match_index = np.where(Point_ID[:,0]==Point_ID[:,1])
    nonmatch_index = np.where(Point_ID[:,0]!=Point_ID[:,1])
    # get match pairs and non match pairs
    match_pairs_ID = patches_ID[match_index[0]]
    nonmatch_pairs_ID = patches_ID[nonmatch_index[0]]
    
    nums = len(Point_ID)
    # three patches concat
    idx = 0
    patch_images = np.zeros(shape=(nums,64,64,3),dtype=np.uint8)
    for i,pos_patch_ids in enumerate(match_pairs_ID):
        for neg_id in nonmatch_pairs_ID[i]:
            pos_id1 = pos_patch_ids[0]
            pos_id2 = pos_patch_ids[1]
            # print('original patch id1:%d--patch id2:%d--patch id2:%d'%(pos_id1,pos_id2,neg_id))
            img1_id,offset1 = divmod(pos_id1,256)
            img2_id,offset2 = divmod(pos_id2,256)
            img3_id,offset3 = divmod(neg_id,256)
            # patch coordinate in a full image block
            x1,y1 = divmod(offset1,16)
            x2,y2 = divmod(offset2,16)
            x3,y3 = divmod(offset3,16)
            # print('img_id1',img1_id,'x1:%d,y1:%d'%(x1,y1))
            # print('img_id2',img2_id,'x2:%d,y2:%d'%(x2,y2))
            # print('img_id3',img3_id,'x3:%d,y3:%d'%(x3,y3))
            # imread a specific image block
            img1 = plt.imread(patch_names[img1_id],0)
            img2 = plt.imread(patch_names[img2_id],0)
            img3 = plt.imread(patch_names[img3_id],0)
            # channel concatenate
            patch1 = np.expand_dims(img1[x1*64:(x1+1)*64,y1*64:(y1+1)*64],axis=2)
            patch2 = np.expand_dims(img2[x2*64:(x2+1)*64,y2*64:(y2+1)*64],axis=2)
            patch3 = np.expand_dims(img3[x3*64:(x3+1)*64,y3*64:(y3+1)*64],axis=2)
            img_pair = np.dstack((patch1,patch2,patch3))
            patch_images[idx,:,:,:] = img_pair
            # num ++
            idx += 1
            #print(img_pair.shape)
            #plt.subplot(1,3,1)
            #plt.imshow(img_pair[:,:,0])
            #plt.subplot(1,3,2)
            #plt.imshow(img_pair[:,:,1])
            #plt.subplot(1,3,3)
            #plt.imshow(img_pair[:,:,2])
            #plt.show()
            view_bar(idx,nums)
        
        
    print('\nStarting to save patch images')
    np.save(os.path.join(preprocessed_data_dir,
                         '%s_500k_triplet_patch_image.npy'%dataName),
            patch_images)
    print('Finished!!!')
   


def view_bar(step,total_nums):
    rate = step/total_nums
    rate_num = int(rate*40)
    r = '\r[%s%s]%d%%\t step-%d/%d'%('>'*rate_num,'-'*(40-rate_num),rate*100,step,total_nums)
    print(r,end='',flush=True)
    
    
if __name__ == '__main__':
    generate_patch(match_info_file)
