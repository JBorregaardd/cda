#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:05:30 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS breakdown in the Overfitting Regime (n approx m).

# 1. Load and Expand Data
print('Loading Wine Quality data...')
# TASK: Load data and use PolynomialFeatures(degree=2) to expand features to m=77.

# 2. Unstable Split
# TASK: Use a very small training fraction (e.g., train_size=0.05 or n=80) 
# to make n approach m.

# 3. Fit OLS
# TASK: Fit LinearRegression on the expanded features.

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Note if the test error "explodes" compared to the stable regime.

# 5. Visualization
# TASK: Create a bar chart (use log scale for y-axis if error is very high).