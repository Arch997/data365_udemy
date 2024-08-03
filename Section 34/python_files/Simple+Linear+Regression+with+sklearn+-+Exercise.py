#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:30:29 2024

@author: charles
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

sns.set()

# Import the data
data = pd.read_csv('../../Section 32/real_estate_price_size.csv')
print(data.head())

# Create the regression
x = data['size']
y = data['price']

print(x.shape)

x_matrix = x.values.reshape(-1, 1)

print(x_matrix.shape)

reg = LinearRegression()

# Fit linear model
reg.fit(x_matrix, y)

coeffs = reg.coef_

r_squared = reg.score(x_matrix, y)

intercept = reg.intercept_

predictions = reg.predict(x_matrix)

size_to_price = dict(zip(x, predictions))

x_pd = pd.DataFrame({"Size": x})
x_pd["Predicted Price"] = predictions
print(x_pd)

x_pd.to_csv("size_to_price.csv", index=True)

yhat = coeffs * x_matrix + intercept


plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Actual Price")

plt.plot(x, yhat, linewidth=4, label="Regression Line")
plt.xlabel("Size")
plt.ylabel("Price")

plt.title("Size vs Price")

plt.legend()
plt.show()
