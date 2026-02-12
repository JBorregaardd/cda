import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Quantify the Bias-Variance tradeoff and Training vs. Test 
# error behavior using empirical data simulations (oracle knowledge of f(x)).

# 1. Setup Parameters
np.random.seed(42)
n_train = 100
n_test = 500         # Large test set to accurately estimate EPE
n_simulations = 500  # Number of 'parallel universes' to average over
sigma = 1.0          # Irreducible Noise level
beta_true = np.array([2, 0])
x_eval = np.array([[1, 1]]) # Point for specific Bias/Var analysis
target_val = (x_eval @ beta_true)[0]
lambdas = np.logspace(-2, 5, 25)

# Generate Fixed Design Matrices
# We fix X to observe the variation caused by noise (epsilon) alone
mean = [0, 0]
cov = [[1, 0.98], [0.98, 1]] # High correlation for instability
X_train = np.random.multivariate_normal(mean, cov, n_train)
X_test = np.random.multivariate_normal(mean, cov, n_test)

# 2. Simulation Loop
emp_bias_sq = []
emp_variance = []
mean_train_mse = []
mean_test_mse = []

print(f'Simulating {n_simulations} universes for each complexity level...')
for l in lambdas:
    iteration_preds = []
    iteration_train_mse = []
    iteration_test_mse = []
    
    for _ in range(n_simulations):
        # Generate y_train and y_test with fresh noise epsilon ~ N(0, sigma^2)
        y_train = (X_train @ beta_true) + np.random.normal(0, sigma, n_train)
        y_test = (X_test @ beta_true) + np.random.normal(0, sigma, n_test)
        
        # Fit Ridge model
        model = Ridge(alpha=l).fit(X_train, y_train)
        
        # 1. Prediction at the fixed evaluation point (for Bias/Var)
        pred_eval = model.predict(x_eval)[0]
        iteration_preds.append(pred_eval)
        
        # 2. Training and Test MSE (Empirical Error)
        train_p = model.predict(X_train)
        test_p = model.predict(X_test)
        iteration_train_mse.append(mean_squared_error(y_train, train_p))
        iteration_test_mse.append(mean_squared_error(y_test, test_p))
    
    # Aggregate results for this lambda
    avg_pred = np.mean(iteration_preds)
    emp_bias_sq.append((avg_pred - target_val)**2)
    emp_variance.append(np.var(iteration_preds))
    mean_train_mse.append(np.mean(iteration_train_mse))
    mean_test_mse.append(np.mean(iteration_test_mse))

# 3. Visualization
plt.figure(figsize=(12, 7))

# Plot Empirical Error Curves (The 'What')
plt.plot(lambdas, mean_train_mse, 'b-', label='Mean Training MSE', lw=2.5)
plt.plot(lambdas, mean_test_mse, 'r-', label='Mean Test MSE (EPE Proxy)', lw=2.5)

# Plot Empirical Bias/Var Components at x_eval (The 'Why')
plt.plot(lambdas, emp_bias_sq, 'r--', alpha=0.6, label='Empirical Bias^2 (at x_eval)')
plt.plot(lambdas, emp_variance, 'b--', alpha=0.6, label='Empirical Variance (at x_eval)')

# Irreducible noise floor
plt.axhline(sigma**2, color='gray', linestyle=':', label='Noise Floor (σ²)')

plt.xscale('log')
plt.gca().invert_xaxis() # Complexity increases to the right (lower lambda)
plt.xlabel('Model Complexity (Lower Lambda) ->')
plt.ylabel('Error Value')
plt.title('Empirical Error Decomposition: Training, Test, Bias, and Variance')
plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
plt.grid(alpha=0.2)

# Optimal point marking
opt_idx = np.argmin(mean_test_mse)
plt.axvline(lambdas[opt_idx], color='green', linestyle='-.', alpha=0.4)
plt.text(lambdas[opt_idx], plt.ylim()[1]*0.9, ' Sweet Spot', color='green', fontweight='bold')

plt.tight_layout()
plt.show()

print('-' * 40)
print('Key Empirical Observations:')
print(f'1. Irreducible Error Floor (sigma^2): {sigma**2}')
print(f'2. Minimal observed Test MSE:         {np.min(mean_test_mse):.4f}')
print(f'3. Training error is zero-biased and optimistic at high complexity.')
print(f'4. The U-shape of Test MSE is driven by rising Variance at high complexity.')
print('-' * 40)