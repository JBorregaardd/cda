#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:29:49 2026

@author: sned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def wine_audit_one_se():
    '''
    Implementing the One-SE rule on the Wine Quality dataset.
    '''
    # Load data (Ensure the file is in your directory)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    
    # For now, we assume data is loaded and split into X and y
    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    # Complexity sweep
    lambdas = np.logspace(-2, 6, 25)
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    kf.get_n_splits()  # Just to initialize the generator    
    cv_means = []
    cv_ses = []

    print('--- Starting 10-Fold Cross-Validation ---')
    
    for l in lambdas:
        fold_errors = []
        # TODO: Loop through kf.split(X)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=l).fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            fold_errors.append(mse)
        

        mean_error = np.mean(fold_errors)
        se_error = np.std(fold_errors) / np.sqrt(K)
        cv_means.append(mean_error)
        cv_ses.append(se_error)
        # 1. Fit Ridge model with alpha=l on training folds
       
        # 2. Calculate MSE on the validation fold
        # 3. Append to fold_errors
        
        # TODO: Calculate Mean and SE for this lambda
        # Mean = np.mean(fold_errors)
        # SE = np.std(fold_errors) / np.sqrt(K)



    # TODO: Identify lambda_min (index of the lowest cv_mean)
    idx_min = np.argmin(cv_means)
    lambda_min = lambdas[idx_min]
    print(f'Lambda with lowest CV error: {lambda_min:.4f}')
    
    # TODO: Apply the One-SE Rule
    # 1. threshold = cv_means[idx_min] + cv_ses[idx_min]
    # 2. Find the largest lambda where cv_mean <= threshold
    
    threshold = cv_means[idx_min] + cv_ses[idx_min]
    idx_one_se = np.where(np.array(cv_means) <= threshold)[0]
    lambda_one_se = lambdas[idx_one_se[-1]]  # Largest lambda meeting the condition
    print(f'Lambda selected by One-SE rule: {lambda_one_se:.4f}')
    # --- VISUALIZATION ---
    # TODO: Create the CV Error Plot
    # Use plt.errorbar(lambdas, cv_means, yerr=cv_ses, color=DTU_RED)
    # Add a horizontal line for the threshold using DTU_NAVY
    # Set x-axis to log scale
    plt.errorbar(lambdas, cv_means, yerr=cv_ses, color=DTU_RED)
    plt.axhline(y=threshold, color=DTU_NAVY, linestyle='--')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('CV Error (MSE)')
    plt.title('Cross-Validation Error with One-SE Rule')
    plt.show()
    print('Audit Complete.')

if __name__ == '__main__':
    wine_audit_one_se()