#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:54:28 2018

@author: halil
"""

import pandas as pd
import numpy as np

titanic_df = pd.read_csv("./titanic/train.csv")
test_df = pd.read_csv("./titanic/test.csv")

titanic_df.info()

titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1
test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

titanic_df["Sex"] = titanic_df["Sex"].astype(int)
test_df["Sex"] = test_df["Sex"].astype(int)
titanic_df.info()

titanic_df["Age_Filled"] = titanic_df["Age"]

titanic_df["Age_Filled"] = titanic_df["Age_Filled"].fillna(titanic_df["Age"].mean())
print("Age Median : ", titanic_df["Age"].median())
print("Age Mean : ", titanic_df["Age"].mean())