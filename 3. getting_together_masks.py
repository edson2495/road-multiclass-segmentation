#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:20:04 2023

@author: edson2495
"""

import os
from natsort import natsorted
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import shutil

os.chdir("/home/edson2495/python/tesis/prueba14")


#%% Obteniendo los nombres originales

IMGS_PATH = "imagenes/both_divided/imgs/"
MASKS_PATH = "imagenes/both_divided/original_masks/"

#original img extension : .jpg
#mask extension : .png

#paved imgs

#reading original imgs
filename_imgs = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( IMGS_PATH ) ) ) )
#reading masks
filename_masks = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( MASKS_PATH ) ) ) )


#%%

lotes_path = "imagenes/both_divided/lotes/"

imgs = []
original_masks = []
generated_masks = []

batch_size = 550
batch_number = 4


C = 60

#for bn in range(batch_number):
for bn in range(2,batch_number):#lotes 1 y 2 ya estaban
    print("LOTE ACTUAL :",bn+1)

    #getting all filenames from the exported masks
    exported_masks_filenames = natsorted(os.listdir(lotes_path + str(bn+1) + "/exported_masks/"))

    limit = len(filename_imgs) if batch_size*(bn+1) > len(filename_imgs) else batch_size*(bn+1)
    for i in range( batch_size*bn, limit ):
        print("\t NRO IMG ACTUAL :",i)
        
        #filtering only labels for the current img
        filtered_exported_masks_filenames = list( filter(lambda filename : "task-"+str(C + i + 1)+"-" in filename, exported_masks_filenames) )
        
        #extra
        if len( filtered_exported_masks_filenames ) == 0:
            print("\t Terminar antes este for interno de imgs")
            break
        
        img = cv2.imread(lotes_path + str(bn+1) + "/imgs/"+filename_imgs[i])
        original_mask = cv2.imread(lotes_path + str(bn+1) + "/masks/"+filename_masks[i], cv2.IMREAD_GRAYSCALE)
        
        
        exported_masks = []
        
        for exported_mask_filename in filtered_exported_masks_filenames:
            exported_mask = cv2.imread(lotes_path + str(bn+1) + "/exported_masks/"+exported_mask_filename, cv2.IMREAD_GRAYSCALE)
            
            if "Unpaved" in exported_mask_filename:
                unpaved_mask = np.where(exported_mask>0, 2, exported_mask) # making it a binary img
                exported_masks.append(unpaved_mask)
                
            else: #paved
                paved_mask = np.where(exported_mask>0, 1, exported_mask) # making it a binary img
                exported_masks.append(paved_mask)
        
        if len(exported_masks) > 0:
            mask = np.maximum.reduce( exported_masks ) #predomina el valor 2
            
            #considering only from the original mask
            mask[ original_mask == 0  ] = 0
            
            generated_masks.append(mask)
            #print("\t d :",np.unique(mask))#
            
            cv2.imwrite(lotes_path + str(bn+1) + "/generated_masks/"+filename_masks[i],mask) #genera menos ruido
            #quizas lo mejor sería trabajarlo asi o usar .npy
            #np.save(lotes_path + str(bn+1) + "/generated_masks/"+filename_masks[:-4]+".npy",mask)
            #break






#%% Moving the generated mask files

FINAL_PATH = "imagenes/divided_imgs/generated_imgs_with_masks/"

for bn in range(batch_number):
    print("LOTE ACTUAL :",bn+1)

    #getting all filenames from the exported masks
    generated_mask_filenames = natsorted(os.listdir(lotes_path + str(bn+1) + "/generated_masks/"))
    
    for filename in generated_mask_filenames:
        
        original_img = lotes_path + str(bn+1) + "/imgs/" + filename[:-9] + "_sat.jpg"
        target_img = FINAL_PATH + filename[:-9] + "_sat.jpg"
        shutil.copyfile(original_img, target_img)
        
        original_mask = lotes_path + str(bn+1) + "/generated_masks/" + filename
        target_mask = FINAL_PATH + filename
        shutil.copyfile(original_mask, target_mask)

        #break
    #break




#%%



