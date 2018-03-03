#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:21:03 2018

@author: halil
"""

import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

image_size = 28
channel_count = 1 # 1:grayscale, 3:rgb
label_count = 10 # sınıf sayısı (10 rakam olduğu için)
test_data_ratio = 0.2 # %20 test, %80 eğitim
images_folder = "images"
dataset_folder = "dataset"

def getImage(images_folder):
    image = imread(images_folder, flatten= True if channel_count == 1 else False)
    image = imresize(image, (image_size, image_size, channel_count))
    return image

def listdirWithoutHidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
labels = listdirWithoutHidden(images_folder) #os.listdir(images_folder)
X, Y = [], []

for i, label in enumerate(labels):
    labels_folder = images_folder + "/" + label
    
    for image_name in listdirWithoutHidden(labels_folder):
        if os.path.isdir(image_name): 
            image = getImage(labels_folder + "/" + image_name)
            X.append(image)
            Y.append(i)

X = np.array(X).astype('float32')/255.
X = X.reshape(X.shape[0], image_size, image_size, channel_count)
Y = np.array(Y).astype('float32')
Y = to_categorical(Y, label_count)

if not os.path.exists(dataset_folder + "/"):
    os.makedirs(dataset_folder + "/")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_data_ratio, random_state=53)

np.save(dataset_folder + "/x_train.npy", x_train)
np.save(dataset_folder + "/x_test.npy", x_test)
np.save(dataset_folder + "/y_train.npy", y_train)
np.save(dataset_folder + "/y_test.npy", y_test)
