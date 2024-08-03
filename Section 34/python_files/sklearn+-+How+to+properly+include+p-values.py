#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 00:15:15 2024

@author: charles
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import f_regression
import seaborn as sns

import os
import sys
sys.path.append("../../modules")


from custom_linear_regression import CustomLinearRegressionWithPValues, adjust_r2

sns.set()

data = pd.read_csv("../../Section 33/1.02.+Multiple+linear+regression.csv")
# data = pd.read_csv('../../Section 33/real_estate_price_size_year.csv')

x = data[["SAT", "Rand 1,2,3"]]
y = data["GPA"]

print(y.shape)


# Create the regression based on custom model
reg_with_pvalues = CustomLinearRegressionWithPValues()
reg_with_pvalues.fit(x, y)
intercept = reg_with_pvalues.intercept_

r2 = reg_with_pvalues.score(x, y)
adj_r2 = adjust_r2(x, y, reg_with_pvalues)

f_reg = f_regression(x, y)

p_vals_f_reg = f_reg[1].round(3)

print(p_vals_f_reg)

print(reg_with_pvalues.p)

reg_summary = pd.DataFrame(data=x.columns.values, columns=["Features"])
reg_summary["Coefficients"] = reg_with_pvalues.coef_
reg_summary["p-values"] = reg_with_pvalues.p.round(3)
reg_summary["R-squared"] = r2
reg_summary["Adjusted R-squared"] = adj_r2
reg_summary["const"] = intercept

predictions = reg_with_pvalues.predict(x)
prediction_summary = pd.DataFrame({"SAT": x["SAT"], "Rand 1,2,3": x["Rand 1,2,3"], "Predicted GPA": predictions})










