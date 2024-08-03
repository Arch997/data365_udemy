#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:52:37 2024

@author: charles
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import pandas as pd
import seaborn as sns

sns.set()

# Load the data
data = pd.read_csv("../../Section 33/1.02.+Multiple+linear+regression.csv")

print(data.describe())

"""Create the regression"""

# Declare the Y and Xs
y = data["GPA"]
x = data[["SAT", "Rand 1,2,3"]]

reg = LinearRegression()

reg.fit(x, y)

coefs = reg.coef_


# Get the R-squared
r_squared = reg.score(x, y)

"""
  Formula for the Adjusted R-squared
  
  R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}$
  
  where n = number of observations; p = number of predictors (features)
"""

print(x.shape)

adj_r_squared = 1 - (1-r_squared) * (84 - 1) / (84 - 2 - 1)

intercept = reg.intercept_

predictions = reg.predict(x)

x_pd = pd.DataFrame({"SAT": x["SAT"], "Rand 1,2,3": x["Rand 1,2,3"], "Predicted GPA": predictions})

# Declare the F regression
f_reg = f_regression(x, y)

# Get the P-values
p_values = f_reg[1].round(3)

# Mao the P-values to their respective features in the needed order.
p_values_map = dict(zip(x, p_values))
