#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:58:44 2018

@author: halil
"""
import numpy as np
import pandas as pd

dataset = pd.read_csv("./salary/Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

print(x)
print(y)