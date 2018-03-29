#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:52:42 2018

@author: halil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./datasets/data.csv")
x = dataframe.iloc[:, 1:2].values
y = dataframe.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(x, y)

y_pred = regressor.predict(9.1)

plt.subplot(1, 2, 1)
plt.scatter(x, y, color = "red")
plt.step(x, regressor.predict(x), color = "blue")
plt.title("Rastgele Orman Model")
plt.xlabel("Pozisyon")
plt.ylabel("Maaş")
plt.show()

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Rastgele Orman (Yüksek Hassasiyetli)")
plt.xlabel("Pozisyon")
plt.ylabel("Maaş")
plt.show()