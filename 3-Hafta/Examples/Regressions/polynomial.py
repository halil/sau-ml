#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:02:58 2018

@author: halil
"""

# y = b0 + b1*x1 + b2*x1^2 + b3*x1^3 + ... + bn*x1^n

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./datasets/Position_Salaries.csv")

x = dataframe.iloc[:, 1:2].values
y = dataframe.iloc[:, 2].values
"""x1 = dataframe.iloc[:, :-1].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x1[:, 0] = encoder.fit_transform(x1[:, 0])
x1 = x1.astype(int)"""

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
x_poly = poly_regressor.fit_transform(x)
poly_regressor.fit(x_poly, y)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(x_poly, y)

"""plt.scatter(x, y, color = "red")
plt.plot(x, linear_regressor.predict(x), color = "blue")
plt.title("Pozisyon - Maaş Lineer Regresyon")
plt.xlabel("Pozisyon Seviyesi")
plt.ylabel("Maaş")
plt.show()"""

"""plt.scatter(x, y, color = "red")
plt.plot(x, linear_regressor2.predict(x_poly), color = "blue")
plt.title("Pozisyon - Maaş Polinom Regresyon")
plt.xlabel("Pozisyon Seviyesi")
plt.ylabel("Maaş")
plt.show()"""

x_new = np.arange(min(x), max(x), 0.1)
x_new = x_new.reshape(len(x_new), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_new, linear_regressor2.predict(poly_regressor.fit_transform(x_new)), color = "blue")
plt.title("Pozisyon - Maaş Polinom Regresyon (Hassas)")
plt.xlabel("Pozisyon Seviyesi")
plt.ylabel("Maaş")
plt.show()

p1 = linear_regressor.predict(6.5)
p2 = linear_regressor2.predict(poly_regressor.fit_transform(6.5))




















