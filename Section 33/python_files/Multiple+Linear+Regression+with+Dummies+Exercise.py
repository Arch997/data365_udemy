#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 02:58:15 2024

@author: charles
"""

import matplotlib as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

sns.set()

raw_data = pd.read_csv('../real_estate_price_size_year_view.csv')

raw_data.head()

raw_data.describe()

# Create a dummy variable for "view"
data = raw_data.copy()

data["view"] = data["view"].map({"Sea view": 1, "No sea view": 0})
print(data)
print(data.describe())


y = data["price"]
x1 = data[["size", "year", "view"]]

# Add a constant. Esentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)

# Fit the model according to OLS
results = sm.OLS(y, x).fit()

print(results.summary())