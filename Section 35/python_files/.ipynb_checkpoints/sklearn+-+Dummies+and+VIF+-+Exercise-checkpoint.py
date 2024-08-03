#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:05:39 2024

@author: charles
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import sys

sns.set()

sys.path.append("../../modules")

from custom_linear_regression import CustomLinearRegressionWithPValues, adjust_r2

# Load the raw data as csv
raw_data = pd.read_csv("../1.04.+Real-life+example.csv")

# Check for missing values
data_no_mv = raw_data.dropna(axis=0)

# Drop the Model column
data_no_mv = data_no_mv.drop("Model", axis=1)


# Exploring the PDFs
"""plt.figure(figsize=(10, 5))
sns.histplot(data_no_mv["Price"], stat="density", label="Original")"""

# Manage outliers from the data to have a more normally distributed data
q_price = data_no_mv['Price'].quantile(0.99)

# Filter the 'data_no_mv' DataFrame to include only rows where the 'Price' column is less than the 99th percentile value 'q'
data_1 = data_no_mv[data_no_mv['Price'] < q_price]


q_mileage = data_1["Mileage"].quantile(0.99)
data_2 = data_1[data_1["Mileage"] < q_mileage]


data_3 = data_2[data_2["EngineV"] < 6.5]


q_year = data_3["Year"].quantile(0.01)
data_4 = data_3[data_3["Year"] > q_year]

"""plt.figure(figsize=(10, 5))
sns.histplot(data_4["Year"])"""

data_cleaned = data_4.reset_index(drop=True)

# Checking the OLS assumptions
"""f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned["Year"], data_cleaned["Price"])
ax1.set_title("Price and Year")
ax2.scatter(data_cleaned["Mileage"], data_cleaned["Price"])
ax2.set_title("Mileage and Price")
ax3.scatter(data_cleaned["EngineV"], data_cleaned["Price"])
ax3.set_title("Engine and Price")"""

# Log transformation to handle linearity
# Calculate the natural logarithm of the 'Price' column in the 'data_cleaned' DataFrame
log_price = np.log(data_cleaned["Price"])
data_cleaned["log_price"] = log_price

# Confirm assumptions after log trans
"""f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned["Year"], data_cleaned["log_price"])
ax1.set_title("Price and Year")
ax2.scatter(data_cleaned["Mileage"], data_cleaned["log_price"])
ax2.set_title("Mileage and Log Price")
ax3.scatter(data_cleaned["EngineV"], data_cleaned["log_price"])
ax3.set_title("Engine and Log Price")"""

data_cleaned = data_cleaned.drop("Price", axis=1)

# One-hot encode the categorical variables
categorical_features = ['Brand', 'Body', 'Engine Type', 'Registration', ] # "Model"
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features, drop_first=True)

# Transform the categorical variables to numeric values
data_encoded = data_encoded.astype({col: 'int' for col in data_encoded.select_dtypes(include='bool').columns})


# Check for multicollinearity
variables_for_vif = data_cleaned[["EngineV", "Year", "Mileage"]]
vif_data = pd.DataFrame()

vif_data["VIF"] = [variance_inflation_factor(variables_for_vif.values, i) for i in range (variables_for_vif.shape[1])]
vif_data["Features"] = variables_for_vif.columns

# Drop the dependent variable - log_price
# To actually assess multicollinearity for the predictors, we have to drop 'log_price'. 
# The multicollinearity assumption refers only to the idea that the independent variables shoud not be collinear.
var_for_vif_all_vars = data_encoded.drop("log_price", axis=1)
vif_data_2 = pd.DataFrame()

vif_data_2["VIF"] = [variance_inflation_factor(var_for_vif_all_vars, i) for i in range(var_for_vif_all_vars.shape[1])]
vif_data_2["Features"] = var_for_vif_all_vars.columns

# Drop year for no multicoll
data_no_multicoll = data_encoded.drop(["Year", "log_price"], axis=1)

vif_data_3 = pd.DataFrame()

vif_data_3["VIF"] = [variance_inflation_factor(data_no_multicoll, i) for i in range(data_no_multicoll.shape[1])]
vif_data_3["Features"] = data_no_multicoll.columns

data_encoded = data_encoded.drop("Year", axis=1)


# Define the inputs and target
target = data_encoded["log_price"]
inputs = data_encoded.drop("log_price", axis=1)


# Select top 10 features based on K-statistic
selector = SelectKBest(score_func=f_regression, k=15)
X_selected = selector.fit_transform(inputs, target)

selected_features = inputs.columns[selector.get_support()]
scores = selector.scores_[selector.get_support()]
p_values = selector.pvalues_[selector.get_support()]

selected_stats = pd.DataFrame({
      "Features": selected_features, 
      "Score": scores,
      "P-value": p_values.round(3)}
    )


# Split test and training data
X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=365)

# Scale the features (fit on training data and transform both training and test data)
scaler = StandardScaler()

# Standardise the inputs
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)


regressor = CustomLinearRegressionWithPValues()

regressor.fit(x_train_scaled, y_train)

# Evaluate the model
train_r2 = regressor.score(x_train_scaled, y_train)
test_r2 = regressor.score(x_test_scaled, y_test)
coefs = regressor.coef_
intercept = regressor.intercept_

# Make predictions on test score
test_yhat_log = regressor.predict(x_test_scaled)
train_yhat_log = regressor.predict(x_train_scaled)

test_yhat_exp = np.exp(test_yhat_log)
train_yhat_exp = np.exp(train_yhat_log)

test_yhat_summary = pd.DataFrame(X_test, columns=selected_features)
test_yhat_summary.insert(0, "Test Predictions", test_yhat_exp)
test_yhat_summary.insert(1, "Actual Price", np.exp(target))

plt.scatter(y_test, test_yhat_log, alpha=0.2)
plt.xlabel("Target - Test", size=20)
plt.ylabel("Test Predictions", size=20)
plt.ylim(6,13)
plt.xlim(6, 13) 
plt.show()

train_yhat_summary = pd.DataFrame(X_train, columns=selected_features)
train_yhat_summary.insert(0, "Predictions", train_yhat_exp)
train_yhat_summary.insert(1, "Actual Price",np.exp( target))

plt.scatter(y_train, train_yhat_log, alpha=0.2)
plt.xlabel("Target", size=20)
plt.ylabel("Predicted Price", size=20)
plt.ylim(6,13)
plt.xlim(6, 13) 
plt.show()

# Plot the Residuals
plt.figure(figsize=(10,5))
sns.histplot(y_train  - train_yhat_log, stat="density")
plt.title("Residuals PDF")

# Show the weights
weights = pd.DataFrame(selected_features, columns=["Features"])
weights["Weights"] = coefs




