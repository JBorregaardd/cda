import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DTU Colors for plotting
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def info_leakage_audit():
    '''
    Simulating pure noise to catch Data Leakage.
    '''
    np.random.seed(42)
    N, M = 50, 1000
    # Create pure random noise
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    print('--- Workflow A:Leakage ---')
    # TODO: Implement the 'Leaky' workflow
    # 1. Standardize the WHOLE dataset (X) using StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # 2. Calculate absolute correlation between each feature in X_scaled and y
    corr = np.array([np.corrcoef(X_scaled[:, i], y)[0, 1] for i in range(M)])
    abs_corr = np.abs(corr)
    # 3. Select the indices of the top 10 features with highest correlation
    top_10_indices = np.argsort(abs_corr)[-10:]
    # 4. Create X_selected containing only these 10 features
    X_selected = X_scaled[:, top_10_indices]
    # 5. Split (X_selected, y) into 50/50 train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.5, random_state=42)
    # 6. Fit LinearRegression on training and print Test R^2
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_a = model.score(X_test, y_test)
    print(f'Workflow A (Leaky) Test R^2: {r2_a:.3f}')
    # Placeholder for plot data (absolute correlations)
    corrs_a = abs_corr[top_10_indices]

    print('\n--- Workflow B: The Audit (No Leakage) ---')
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 2. Scale (fit on train only)
    scaler_b = StandardScaler().fit(X_train)
    X_train_scaled = scaler_b.transform(X_train)
    X_test_scaled  = scaler_b.transform(X_test)

    # 3. Correlations on train only (ABS like version A)
    corr_b = np.array([
        np.abs(np.corrcoef(X_train_scaled[:, i], y_train)[0, 1])
        for i in range(M)
    ])

    # 4. Select top 10 features (use these indices directly)
    top_indices_b = np.argsort(corr_b)[-10:]

    # Optional: keep sorted correlation values for reporting
    corrs_b_train = np.sort(corr_b[top_indices_b])[::-1]

    # 5. Subset using top_indices_b
    X_train_selected = X_train_scaled[:, top_indices_b]
    X_test_selected  = X_test_scaled[:, top_indices_b]

    # 6. Fit and score
    model_b = LinearRegression().fit(X_train_selected, y_train)
    r2_b = model_b.score(X_test_selected, y_test)
    
    
    print(f'Workflow B (non-leaky) Test R^2: {r2_b:.3f}')

     # --- VISUALIZATION OF THE EVIDENCE ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 11), corrs_a, color=DTU_RED)
    plt.title('Leaky Correlations (Cheating)', color=DTU_NAVY, fontsize=14)
    plt.ylabel('Abs. Correlation with y')
    plt.xlabel('Top 10 Features (Whole Data)')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    # Note: We plot the correlations of these features on the TEST set for Workflow B
    # to show they don't actually hold up.
    test_corrs_b = np.array([np.abs(np.corrcoef(X_test_selected[:, i], y_test)[0, 1]) for i in range(10)])
    plt.bar(range(1, 11), test_corrs_b, color=DTU_NAVY)
    plt.title('non-leaky Correlations (Audited)', color=DTU_NAVY, fontsize=14)
    plt.ylabel('Abs. Correlation with y')
    plt.xlabel('Top 10 Features (Selected on Train)')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print('\nVERDICT:')
    print('Workflow A is a scientific crime because information from the test set')
    print('labels leaked into the feature selection step. By choosing features')
    print('that correlate with the labels across the entire dataset, we found')
    print('spurious patterns that happen to exist in the test set by chance.')
    print('Workflow B shows that when selection is properly isolated, these')
    print('patterns disappear on unseen data.')

if __name__ == '__main__':
    info_leakage_audit()