#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:43:08 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Learning Objective: Observe coefficient instability and its link to collinearity.

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
    '''
    TASK: Generate synthetic data.
    1. Create X with two features correlated by rho (use multivariate_normal).
    2. Generate y = X @ beta_true + noise.
    '''
    # YOUR CODE HERE
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n)
    y = X @ beta_true + np.random.normal(0, sigma, n)
    return X, y

# --- SECTION 2: Simulation ---
# TASK: Run a loop for n_simulations.
all_betas = []
all_preds = []

print('Running simulations...')
# for _ in range(n_simulations):
#     X, y = generate_data(...)
#     model = ...
#     # Store results
for _ in range(n_simulations):
    X, y = generate_data(n_samples, rho, sigma)
    model = LinearRegression().fit(X, y)
    all_betas.append(model.coef_)
    pred = model.predict(x_test)
    all_preds.append(pred[0])


# --- SECTION 3: Calculations ---
# TASK: Calculate the following metrics:
# 1. Mean and Variance of the estimated coefficients (betas).
beta_mean = np.mean(all_betas, axis=0)
beta_variance = np.var(all_betas, axis=0)
# 2. The Bias^2 at x_test.
# 3. The Variance at x_test.

bias_sq = (np.mean(all_preds) - target_val)**2
variance = np.var(all_preds)

betas = np.asarray(all_betas)
# --- SECTION 4: Visualization ---
# TASK: Create a histogram of estimated Beta 1 and Beta 2.

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(betas[:, 0], bins=30, alpha=0.7, label=r'$\hat{\beta}_1$')
plt.hist(betas[:, 1], bins=30, alpha=0.7, label=r'$\hat{\beta}_2$')
plt.axvline(beta_true[0], linestyle='dashed', linewidth=2, label=r'True $\beta_1$')
plt.axvline(beta_true[1], linestyle='dashed', linewidth=2, label=r'True $\beta_2$')
plt.title('Distribution of Estimated Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Frequency')
plt.legend()


plt.subplot(1, 2, 2)
plt.hist(all_preds, bins=30, alpha=0.7, label=r'$\hat{y}(x_{test})$')
plt.axvline(target_val, color='red', linestyle='dashed', linewidth=2, label='True Value')
plt.title('Distribution of Predictions at x_test')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()