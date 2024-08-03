#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:07:04 2024

@author: charles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv('../real_estate_price_size.csv')
data.head()

data.describe()

y = data['price']
x1 = data['size']

plt.scatter(x1, y)

# NAme your axes
plt.xlabel("SIZE", fontsize=20)
plt.ylabel("PRICE", fontsize=20)

# Display the plot
plt.show()

# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an independent x
results = sm.OLS(y, x).fit()

summary = results.summary()
print(summary)

plt.scatter(x1,y)

yhat = 223.1787 *x1 +101900
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
# Label the axes
plt.xlabel('SIZE', fontsize = 20)
plt.ylabel('PRICE', fontsize = 20)
plt.show()