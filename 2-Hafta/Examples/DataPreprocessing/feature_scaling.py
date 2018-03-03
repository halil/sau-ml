#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:16:35 2018

@author: halil
"""
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ["preg", "plas", "press", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = pd.read_csv(url, names=names)

dataset = dataframe.values
x = dataset[:, 0:8]
y = dataset[:, 8]

#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
minmaxScaler = MinMaxScaler(feature_range=(-3,3))
rescaled_x = minmaxScaler.fit_transform(x)
#np.set_printoptions(precision=4)
#print(rescaled_x[0:5,:])

# Standart Scaler
from sklearn.preprocessing import StandardScaler
standartScaler = StandardScaler().fit(x)
standart_x = standartScaler.transform(x)


# Normalizer
from sklearn.preprocessing import Normalizer
normalizer = Normalizer().fit(x)
normalized_x = normalizer.transform(x)
np.set_printoptions(precision=4)
print(normalized_x[0:5,:])