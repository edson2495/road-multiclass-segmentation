#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:45:03 2023

@author: edson2495
"""

import os
#os.chdir("/home/edson2495/python/tesis/prueba12")
os.environ["SM_FRAMEWORK"] = "tf.keras"

from natsort import natsorted

from unetr_2d import build_unetr_2d

from tensorflow.keras.optimizers import Adam
from metrics import precision, recall, f1_score
from models import jaccard_coef
import segmentation_models as sm
from datetime import datetime
import pandas as pd

from custom_generator import DataGenerator
import tensorflow as tf

#from tensorflow_addons.optimizers import AdamW



#%% Paths and variables

TRAIN_IMAGES_PATH = "keras_augmentation/train_images/train/"
TRAIN_MASKS_PATH = "keras_augmentation/train_masks/train/"
VAL_IMAGES_PATH = "keras_augmentation/val_images/val/"
VAL_MASKS_PATH = "keras_augmentation/val_masks/val/"
#TEST_IMAGES_PATH = "keras_augmentation/test_images/test/"
#TEST_MASKS_PATH = "keras_augmentation/test_masks/test/"

HEIGHT = 256
WIDTH = 256
PATCH_SIZE = 16
NUMBER_PATCHES = int( (HEIGHT/PATCH_SIZE)*(WIDTH/PATCH_SIZE) )
NUMBER_CHANNELS = 3
NUM_CLASSES = 3

epochs = 50
lr = 1e-4
batch_size = 15
PATIENCE = 5

number_test = "6"
name_prueba = "p14_test" + str(number_test)
type_model = "UNETR"


#%% Reading filenames

train_img_filenames = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( TRAIN_IMAGES_PATH ) ) ) )
train_mask_filenames = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( TRAIN_MASKS_PATH ) ) ) )

val_img_filenames = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( VAL_IMAGES_PATH ) ) ) )
val_mask_filenames = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( VAL_MASKS_PATH ) ) ) )

#test_img_filenames = natsorted( list( filter( lambda filename : ".jpg" in filename, os.listdir( TEST_IMAGES_PATH ) ) ) )
#test_mask_filenames = natsorted( list( filter( lambda filename : ".png" in filename, os.listdir( TEST_MASKS_PATH ) ) ) )

#%% Other consts

num_train_imgs = len(train_img_filenames)
num_val_images = len(val_img_filenames)
#num_train_imgs = 100
#num_val_images = 20
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size

#%% Data generator

training_generator = DataGenerator(
    PATH_IMGS = TRAIN_IMAGES_PATH,
    PATH_LABELS = TRAIN_MASKS_PATH,
    list_IDs = train_img_filenames,
    list_labels = train_mask_filenames,
    #list_IDs = train_img_filenames[:100],
    #list_labels = train_mask_filenames[:100],
    batch_size = batch_size,
    width = WIDTH,
    height = HEIGHT,
    n_channels = NUMBER_CHANNELS,
    n_classes = NUM_CLASSES,
    patch_size = PATCH_SIZE,
    number_patches = NUMBER_PATCHES,
    shuffle = True
)


validation_generator = DataGenerator(
    PATH_IMGS = VAL_IMAGES_PATH,
    PATH_LABELS = VAL_MASKS_PATH,
    list_IDs = val_img_filenames,
    list_labels = val_mask_filenames,
    #list_IDs = val_img_filenames[:20],
    #list_labels = val_mask_filenames[:20],
    batch_size = batch_size,
    width = WIDTH,
    height = HEIGHT,
    n_channels = NUMBER_CHANNELS,
    n_classes = NUM_CLASSES,
    patch_size = PATCH_SIZE,
    number_patches = NUMBER_PATCHES,
    shuffle = True
)



#%% Defining model

config = {}
config["image_size"] = HEIGHT#w y h
config["num_layers"] = 12
config["hidden_dim"] = 768
config["mlp_dim"] = 3072
config["num_heads"] = 12
config["dropout_rate"] = 0.2
config["num_patches"] = NUMBER_PATCHES
config["patch_size"] = PATCH_SIZE
config["num_channels"] = NUMBER_CHANNELS
config["num_classes"] = NUM_CLASSES


model = build_unetr_2d(config)
#model.summary()


#%% Training model

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
    training_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = 1,
    validation_data = validation_generator,
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











