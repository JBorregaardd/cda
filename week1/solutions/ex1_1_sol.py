#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 15:42:52 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Learning Objective: Observe coefficient instability linked to collinearity
# and calculate the empirical Bias-Variance decomposition.

# --- SECTION 1: Parameters ---
np.random.seed(42)
n_samples = 100
n_simulations = 500 
sigma = 1.0          
rho = 0.98           
beta_true = np.array([2, 0])
x_test = np.array([[1, 1]]) 
target_val = (x_test @ beta_true)[0]

def generate_data(n, rho, sigma):
    '''Generates synthetic data with correlated features.'''
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    X = np.random.multivariate_normal(mean, cov, n)
    y = X @ beta_true + np.random.normal(0, sigma, n)
    return X, y

# --- SECTION 2: Simulation ---
all_betas = []
all_preds = []

print(f'Running {n_simulations} OLS simulations...')
for _ in range(n_simulations):
    X_train, y_train = generate_data(n_samples, rho, sigma)
    model = LinearRegression().fit(X_train, y_train)
    
    all_betas.append(model.coef_)
    all_preds.append(model.predict(x_test)[0])

all_betas = np.array(all_betas)
all_preds = np.array(all_preds)

# --- SECTION 3: Calculations ---
mean_beta = np.mean(all_betas, axis=0)
var_beta = np.var(all_betas, axis=0)

# Prediction stats at x_test
expected_pred = np.mean(all_preds)
bias_sq = (expected_pred - target_val)**2
variance = np.var(all_preds)

print('-' * 40)
print(f'Results for OLS (rho={rho}, n={n_samples}):')
print(f'True Beta:            {beta_true}')
print(f'Mean Estimated Beta:  {mean_beta.round(4)}')
print(f'Beta Variance:        {var_beta.round(4)}')
print('-' * 40)
print(f'Prediction Bias^2:    {bias_sq:.4f}')
print(f'Prediction Variance:  {variance:.4f}')
print(f'Total EPE at x_test:  {(bias_sq + variance + sigma**2):.4f}')
print('-' * 40)

# --- SECTION 4: Visualization ---
plt.figure(figsize=(10, 5))
plt.hist(all_betas[:, 0], bins=30, alpha=0.5, label='Beta 1 (True=2)')
plt.hist(all_betas[:, 1], bins=30, alpha=0.5, label='Beta 2 (True=0)')
plt.axvline(2, color='blue', linestyle='--', alpha=0.8)
plt.axvline(0, color='orange', linestyle='--', alpha=0.8)
plt.title(f"Instability of OLS Coefficients (rho={rho})")
plt.xlabel("Coefficient Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()