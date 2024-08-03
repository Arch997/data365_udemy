#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 00:59:13 2024

@author: charles
"""

import numpy as np
import scipy.stats as stat
from sklearn import linear_model

class CustomLinearRegressionWithPValues(linear_model.LinearRegression):
    """
        LinearRegression class after sklearn's, but calculate t-statistics
        and p-values for model coefficients (betas).
        Additional attributes available after .fit()
        are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
        which is (n_features, n_coefs)
        This class sets the intercept to 0 by default, since usually we include it
        in X.
    """
    # Nothing changes in init
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=1, positive=False):
        """self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.copy_X = copy_X"""
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        
    def fit(self, X, y, n_jobs=1):
        super().fit(X, y)
        
        # Calculate SSE (sum of squared errors)
        # and SE (standard error)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        
        # compute the t-statistic for each feature
        self.t = self.coef_ / se
        
        # find the p-value for each feature
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        
        return self
    
def adjust_r2(x,y, regression):
    """
    Calculate adjusted R-squared value given X and Y

    Parameters
    ----------
    x : TYPE - Pandas dataframe
        DESCRIPTION. - The inputs (features) of the model
    y : TYPE - Pandas dataframe
        DESCRIPTION. The dependent var (target) of the model

    Returns
    -------
    adjusted_r2 : TYPE - float
        DESCRIPTION. Adjusted R-square of the regression

    """
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
    