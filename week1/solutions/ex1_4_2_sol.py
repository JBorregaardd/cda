#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 15:45:35 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Understand the relationship between parameters (m) 
# and samples (n) and how their ratio dictates the 'Overfitting Regime'.

# 1. Load the dataset
print('Fetching Wine Quality data...')
wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
df = wine.frame
# Ensure numeric target
df['class'] = df['class'].astype(float)

# 2. Data Preprocessing & Feature Engineering
# We use PolynomialFeatures (degree 2) to increase m (parameters) 
# This helps demonstrate overfitting in a dataset that is otherwise too 'stable'.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(wine.data)

# Scale the expanded feature set
X = StandardScaler().fit_transform(X_poly)
y = df['class'].values

# To illustrate the m vs n logic, we deliberately use a small training set
# n = samples, m = features (parameters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

n_train = X_train.shape[0]
m_features = X_train.shape[1]

print('-' * 40)
print('COMPLEXITY DIAGNOSTIC:')
print(f'Number of training samples (n): {n_train}')
print(f'Number of features/parameters (m): {m_features}')
print(f'Ratio (m/n): {m_features/n_train:.2f}')
print('-' * 40)

# 3. Complexity Sweep (Ridge)
# We sweep from very high lambda (simple/stable) to very low lambda (complex/overfitting)
lambdas = np.logspace(-3, 5, 50)
train_errors = []
test_errors = []

print('Running complexity sweep...')
for l in lambdas:
    model = Ridge(alpha=l).fit(X_train, y_train)
    
    # Store errors
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

# 4. Visualizing the Mismatch (The U-Shape)
plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_errors, 'b-', label='Training Error (Observable)', lw=2)
plt.plot(lambdas, test_errors, 'r-', label='Test Error (Generalization)', lw=2)

# Invert axis: Complexity increases as lambda decreases (moving to the right)
plt.xscale('log')
plt.gca().invert_xaxis() 
plt.xlabel('Model Complexity (Lower Lambda) ->')
plt.ylabel('Mean Squared Error')
plt.title(f'm vs n Learning Moment (m={m_features}, n={n_train})')
plt.legend()
plt.grid(alpha=0.2)

# Set y-limit to focus on the 'valley' of the U-shape
# plt.ylim(0, np.percentile(test_errors, 90))

# Highlight the Optimal Complexity (The Sweet Spot)
opt_idx = np.argmin(test_errors)
opt_lambda = lambdas[opt_idx]
plt.axvline(opt_lambda, color='green', linestyle='--', alpha=0.5)
plt.text(opt_lambda, plt.gca().get_ylim()[1]*0.9, ' Optimal Complexity', color='green')

plt.show()

print('\nTHE LEARNING MOMENT: m << n vs. m ≈ n')
print('1. High Complexity (Right): Test error rises as the model fits noise.')
print('2. Low Complexity (Left): Both errors are high (Underfitting).')
print('3. The gap between curves represents the optimism of training data.')
print('-' * 40)