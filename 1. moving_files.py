#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:09:05 2023

@author: edson2495
"""

import os
from natsort import natsorted
import shutil

os.chdir("/home/edson2495/python/tesis/prueba5")


#%%

BOTH_PATH = "imagenes/divided_imgs/both/"

#original img extension : .jpg
#mask extension : .png

#paved imgs

#reading original imgs
filename_imgs = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( BOTH_PATH ) ) ) )
#reading masks
filename_masks = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( BOTH_PATH ) ) ) )


#%% separando imgs y mascaras

IMGS_DESTINATION_PATH = "imagenes/both_divided/imgs/"
MASKS_DESTINATION_PATH = "imagenes/both_divided/original_masks/"

for i in range( len(filename_imgs) ):
#for i in range( 1 ):
    
    original_img = BOTH_PATH + filename_imgs[i]
    target_img = IMGS_DESTINATION_PATH + filename_imgs[i]
    shutil.copyfile(original_img, target_img)
    
    original_mask = BOTH_PATH + filename_masks[i]
    target_mask = MASKS_DESTINATION_PATH + filename_masks[i]
    shutil.copyfile(original_mask, target_mask)









