#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:02:21 2018

@author: halil
"""

# y = b0 + b1*x1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./datasets/Salary_Data.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 53)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predictions = regressor.predict(x_test)

"""plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Maaş - Tecrübe (Eğitim Seti)")
plt.xlabel("Tecrübe(Yıl)")
plt.ylabel("Maaş(Dolar)")
plt.show()"""

plt.scatter(x_test, y_test, color = "red")
plt.plot(x_test, y_predictions, color = "blue")
plt.title("Maaş - Tecrübe (Test Seti)")
plt.xlabel("Tecrübe(Yıl)")
plt.ylabel("Maaş(Dolar)")
plt.show()