#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:04:43 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS behavior in the Stable Regime (n >> m).

# 1. Load Data (Original 11 features)
print('Loading Wine Quality data...')
wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
X = StandardScaler().fit_transform(wine.data)
y = wine.target.values.astype(float)

# 2. Stable Split: Large n, Small m
# n_train approx 1200, m = 11. Ratio ~ 100:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit OLS
model = LinearRegression().fit(X_train, y_train)

# 4. Evaluate
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

print('-' * 40)
print(f'STABLE REGIME DIAGNOSTIC (n >> m)')
print(f'Samples (n): {X_train.shape[0]} | Features (m): {X_train.shape[1]}')
print(f'Training MSE: {train_mse:.4f}')
print(f'Test MSE:     {test_mse:.4f}')
print(f'Generalization Gap: {test_mse - train_mse:.4f}')
print('-' * 40)

# 5. Visualization
plt.figure(figsize=(8, 5))
errors = [train_mse, test_mse]
labels = ['Training Error', 'Test Error']
plt.bar(labels, errors, color=['blue', 'red'], alpha=0.7)
plt.ylabel('Mean Squared Error')
plt.title('OLS Performance: Stable Regime (n >> m)')
plt.grid(axis='y', alpha=0.3)
plt.show()