#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:45:03 2023

@author: edson2495
"""

import os
#os.chdir("/home/edson2495/python/tesis/prueba14")

from datetime import datetime
#import numpy as np

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import tensorflow as tf #usar la version 2.9.1, la version 2.10 a la fecha de hoy 04-12-2022 tiene un issue
from tensorflow.keras.optimizers import Adam, SGD
from models import UNet, Attention_UNet, Attention_ResUNet, dice_coef, dice_coef_loss, jaccard_coef
from metrics import precision, recall, f1_score
#from unetr import build_unetr_2d

import pandas as pd


#from segnet_model import segnet


#%% Variables

seed = 24
batch_size = 16 #see
n_classes = 3


scaler = MinMaxScaler()


def preprocess_data(img, mask, num_class):
    
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    mask = to_categorical(mask, num_class)
    
    
    return (img,mask)
    


def trainGenerator(train_img_path, train_mask_path, num_class):
    
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,#see
        seed = seed)#me, they have to be the exact same see , very important
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)#31 min
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)
        
        
AUG_PATH = "keras_augmentation/"

train_img_path = AUG_PATH + "train_images/"
train_mask_path = AUG_PATH + "train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class = n_classes)

val_img_path = AUG_PATH + "val_images/"
val_mask_path = AUG_PATH + "val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class = n_classes)

test_img_path = AUG_PATH + "test_images/"
test_mask_path = AUG_PATH + "test_masks/"
test_img_gen = trainGenerator(test_img_path, test_mask_path, num_class = n_classes)



#%%

num_train_imgs = len(os.listdir(AUG_PATH + "train_images/train/"))
num_val_images = len(os.listdir(AUG_PATH + "val_images/val/"))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size




#######################################
#Parameters for model


NUM_CLASSES = n_classes
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3
#num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
epochs = 50
#lr = 7e-3
lr = 1e-4
PATIENCE = 5


number_test = "4"
name_prueba = "p14_test" + str(number_test)
type_model = "UNet"

start1 = datetime.now()


checkpoint_path = "checkpoints/" + str(number_test) + "/"
checkpoint_filename = checkpoint_path + start1.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(name_prueba)+"_"+str(type_model)+"_weights_{epoch:02d}epochs_{val_loss:.4f}val_loss_{val_jaccard_coef:.4f}val_jaccard_coef.hdf5"


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filename,
        verbose = 1,
        save_best_only = True,
        save_weights_only = False),
    tf.keras.callbacks.EarlyStopping(
        patience = PATIENCE,
        verbose = 1,
        monitor = "val_loss",
        mode = "auto",
        restore_best_weights = True
    ),
    tf.keras.callbacks.CSVLogger(
        "models/"+str(number_test)+"/training.csv", 
        separator = ',', 
        append = False
    )
]



model = UNet(input_shape = input_shape, NUM_CLASSES = NUM_CLASSES, dropout_rate = 0.20)



model.compile(
    optimizer = Adam(learning_rate = lr),
    loss = sm.losses.categorical_focal_jaccard_loss, 
    metrics=[
        precision,
        recall,
        f1_score,
        jaccard_coef
    ]
)



print("Inicio :",start1)

history = model.fit(
    train_img_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = 1,
    validation_data = val_img_gen,
    validation_steps = val_steps_per_epoch,
    callbacks = callbacks
)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1 - start1
print("Inicio :",start1)
print("Fin :",stop1)
print("UNet execution time is: ", execution_time_Unet)

model_name = start1.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(name_prueba)+"_"+type_model+"_"+str(epochs)+"epochs_"+str(batch_size)+"bs_"+str(lr)+"lr_C_focal_noResize.hdf5"

model.save("models/"+str(number_test)+"/"+model_name)

#%% Guardando el history

print("Guardando modelo\n")

unet_history_df = pd.DataFrame(history.history)

csv_name = "models/"+str(number_test)+"/"+start1.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(name_prueba)+"_"+type_model+"_"+str(epochs)+"epochs_"+str(batch_size)+"bs_"+str(lr)+"lr_noResize_history_df.csv"

with open(csv_name, mode='w') as f:
    unet_history_df.to_csv(f)








