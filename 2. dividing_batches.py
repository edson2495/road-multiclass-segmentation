#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:04:29 2023

@author: edson2495
"""

import os
from natsort import natsorted
import shutil

os.chdir("/home/edson2495/python/tesis/prueba7")

#%%

IMGS_PATH = "imagenes/both_divided/imgs/"
MASKS_PATH = "imagenes/both_divided/original_masks/"

#original img extension : .jpg
#mask extension : .png

#paved imgs

#reading original imgs
filename_imgs = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( IMGS_PATH ) ) ) )
#reading masks
filename_masks = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( MASKS_PATH ) ) ) )


#%% Separando en 4 lotes de 550 (las 2109 imagenes)

lotes_path = "imagenes/both_divided/lotes/"
#lote2_path = "imagenes/both_divided/lotes/2/"
#lote3_path = "imagenes/both_divided/lotes/3/"
#lote4_path = "imagenes/both_divided/lotes/4/"

batch_size = 550
batch_number = 4
for i in range(batch_number):
    print(batch_size*(i+1))
    
    limit = len(filename_imgs) if batch_size*(i+1) > len(filename_imgs) else batch_size*(i+1)
    for j in range( batch_size*i, limit ):
        
        original_img = IMGS_PATH + filename_imgs[j]
        target_img = lotes_path + str(i + 1) + "/imgs/" + filename_imgs[j]
        shutil.copyfile(original_img, target_img)
        
        original_mask = MASKS_PATH + filename_masks[j]
        target_mask = lotes_path + str(i + 1) + "/masks/" + filename_masks[j]
        shutil.copyfile(original_mask, target_mask)

