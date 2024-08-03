#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:04:49 2024

@author: charles
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

data = pd.read_csv('../1.01.+Simple+linear+regression.csv')

print(data.head())

# Create the regression
# Define the features

# Input feature
x = data["SAT"]

# Output target
y = data["GPA"]

print(x.shape)
print(y.shape)

# In order to feed x to sklearn, it should be a 2D array (a matrix)
# Therefore, we must reshape it 
# Note that this will not be needed when we've got more than 1 feature (as the inputs will be a 2D array by default)

x_matrix = x.values.reshape(-1, 1)
print(x_matrix.shape)

reg = LinearRegression()

reg.fit(x_matrix, y)

# Get the R-squared of the model
r_squared = reg.score(x_matrix, y)

# Get the coeffients of the model
coeffients = reg.coef_

# Get the intercept (constant)
intercept = reg.intercept_

# Make predictions according to the model
predictions = reg.predict(x_matrix)

"""
    We can map the predicted GPAs to our SAT scores data set in a few ways
"""
# Create a dataframe from the feature. Use the 1D x since we are feeding to Pandas
x_pd = pd.DataFrame({"SAT": x})
print(x_pd)

# Add Predicted GPA as a key-value pair to the new dataframe object. The value here is the predictions object previously fitted
x_pd["Predicted GPA"] = predictions
print(x_pd)

# Or create a dictionary to map SAT scores to predicted GPAs
sat_to_gpa = dict(zip(x, predictions))

# Optionally, you can save the results to a CSV file for further analysis
results = pd.DataFrame({'SAT': x, 'Predicted GPA': predictions})
results.to_csv('sat_to_gpa_predictions.csv', index=False)

# Plotting the regression line

# Get the yhat (predicted value)
# yhat can also be represented as the predictions object in our code
yhat = coeffients * x_matrix + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual GPA')

# plt.plot(x, predictions, color='red', linewidth=2, label='Regression Line')
plt.plot(x, yhat, color='red', linewidth=2, label='Regression Line')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.title('SAT Score vs GPA')
plt.legend()
plt.show()



