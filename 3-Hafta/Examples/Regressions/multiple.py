#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:03:11 2018

@author: halil
"""

# y = b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./datasets/50_Startups.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
x[:, 3] = label_encoder.fit_transform(x[:, 3])
onehot_encoder = OneHotEncoder(categorical_features = [3])
x = onehot_encoder.fit_transform(x).toarray()

x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 53)

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

y_predictions = linear_regressor.predict(x_test)

from statsmodels.formula.api import OLS
x_new = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

x_opt = x_new[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = x_opt).fit()

x_opt = x_new[:, [0, 1, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = x_opt).fit()

x_opt = x_new[:, [0, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = x_opt).fit()

x_opt = x_new[:, [0, 3, 5]]
regressor_OLS = OLS(endog = y, exog = x_opt).fit()

x_opt = x_new[:, [0, 3]]
regressor_OLS = OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())