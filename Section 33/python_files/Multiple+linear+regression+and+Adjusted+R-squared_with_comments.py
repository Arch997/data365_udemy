#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:42:28 2024

@author: charles

# Multiple regression model to check if future GPA scores are dependent on other variables
# including SAT scores
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import seaborn as sms

sms.set()

# Load the data
data = pd.read_csv("../1.02.+Multiple+linear+regression.csv")
data.head()

print(data.describe())

# Define the regression variables
y = data["GPA"]
x1 = data[["SAT", "Rand 1,2,3"]]


# Add constant
x = sm.add_constant(x1)

results = sm.OLS(y, x).fit()

summary = results.summary()

print(summary)

yhat = 0.2960 + x1["SAT"] * 0.0017 + x1["Rand 1,2,3"] * -0.0083

