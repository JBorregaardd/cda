#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 15:45:35 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe the behavior of Ridge Regression on a 
# standard real-world dataset (n >> m regime).

# 1. Load the dataset
print('Fetching Wine Quality data...')
wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
X_raw = wine.data
y = wine.target.values.astype(float)

# 2. Data Preprocessing
# n = ~1600 samples, m = 11 features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Standard 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('-' * 40)
print(f'Dataset Statistics:')
print(f'Training samples (n): {X_train.shape[0]}')
print(f'Features (m): {X_train.shape[1]}')
print('-' * 40)

# 3. Complexity Sweep (Ridge)
# Sweeping from very high lambda (simple) to very low lambda (complex)
lambdas = np.logspace(-3, 5, 50)
train_errors = []
test_errors = []

print('Running Ridge complexity sweep...')
for l in lambdas:
    model = Ridge(alpha=l).fit(X_train, y_train)
    
    # Calculate Mean Squared Error (MSE)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, train_preds))
    test_errors.append(mean_squared_error(y_test, test_preds))

# 4. Visualizing Training vs. Test Error
plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_errors, 'b-', label='Training Error (Observable)', lw=2)
plt.plot(lambdas, test_errors, 'r-', label='Test Error (Generalization)', lw=2)

# Invert axis: Complexity increases as lambda decreases (moving to the right)
plt.xscale('log')
plt.gca().invert_xaxis() 
plt.xlabel('Model Complexity (Lower Lambda) ->')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Training vs. Test Error (Original Features)')
plt.legend()
plt.grid(alpha=0.2)

# Identify and mark the optimal complexity
opt_idx = np.argmin(test_errors)
opt_lambda = lambdas[opt_idx]
plt.axvline(opt_lambda, color='green', linestyle='--', alpha=0.5)
plt.text(opt_lambda, plt.gca().get_ylim()[1]*0.9, ' Optimal Alpha', color='green')

plt.show()

print('\nObservations:')
print('1. In this n >> m regime, the model is very stable.')
print('2. Training error and Test error are quite close.')
print(f'3. Optimal alpha is approximately {opt_lambda:.2f}.')
print('-' * 40)