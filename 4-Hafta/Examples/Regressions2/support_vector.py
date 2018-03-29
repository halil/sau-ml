#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:27:35 2018

@author: halil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./datasets/data.csv")
x = dataframe.iloc[:, 1:2].values
y = dataframe.iloc[:, 2].values
y = np.reshape(y, (len(y), 1))

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaled_x = scaler_x.fit_transform(x)
scaled_y = scaler_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

regressor2 = SVR(kernel="rbf")
regressor2.fit(scaled_x, scaled_y)

plt.subplot(1, 2, 1)
plt.scatter(x, y, color = "red")
plt.plot(x, regressor.predict(x), color = "blue")
plt.title("SVR Model")
plt.xlabel("Pozisyon")
plt.ylabel("Maaş")
plt.show()

plt.subplot(1, 2, 2)
plt.scatter(x, y, color = "red")
plt.plot(x, scaler_y.inverse_transform(regressor2.predict(scaled_x)), color = "blue")
plt.title("SVR Model (Öz Nitelik Ölçeklenmiş)")
plt.xlabel("Pozisyon")
plt.ylabel("Maaş")
plt.show()

y_pred = regressor.predict(7.1)
y_pred1 = scaler_y.inverse_transform(regressor2.predict(scaler_x.transform(np.array([[7.1]]))))