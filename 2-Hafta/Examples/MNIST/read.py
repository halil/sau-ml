#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:52:18 2018

@author: halil
"""

import numpy as np

dataset_folder = "dataset"

x_train = np.load(dataset_folder + "/x_train.npy")
x_test = np.load(dataset_folder + "/x_test.npy")
y_train = np.load(dataset_folder + "/y_train.npy")
y_test = np.load(dataset_folder + "/y_test.npy")

print(x_train)