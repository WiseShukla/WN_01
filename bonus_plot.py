"""
Question 6 (BONUS): Localization Accuracy vs Training Set Size
==============================================================
Author: Adarsh Shukla (PhD25001)
Course: Wireless Networks, IIIT Delhi

Created the Python script to evaluates how the KNN localization accuracy changes
as the training set size increases.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("BONUS: KNN ACCURACY vs TRAINING SET SIZE")
print("=" * 70)

# ============================================================================
# STEP 1: Load and Split Data
# ============================================================================
print("\n Loading data and creating train/test split...")

df = pd.read_csv('datatrain.csv')

feature_cols = ['Dist_A0', 'Dist_A1', 'Dist_A2', 'Dist_A3',
                'Dist_A4', 'Dist_A5', 'Dist_A6', 'Dist_A7']
target_cols = ['Target_X', 'Target_Y', 'Target_Z']

# Shuffle data with fixed random seed for reproducibility
np.random.seed(42)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Hold out last 20% as fixed test set
split_idx = int(0.8 * len(df_shuffled))
test_data = df_shuffled.iloc[split_idx:]
full_train = df_shuffled.iloc[:split_idx]

X_test = test_data[feature_cols].values
y_test = test_data[target_cols].values

print(f"  Training pool: {len(full_train)} examples")
print(f"  Test set: {len(test_data)} examples")

# ============================================================================
# STEP 2: Evaluate at Different Training Sizes
# ============================================================================
print("\n Evaluating KNN at different training set sizes...")

percentages = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_sizes = []
mean_errors = []
median_errors = []
p90_errors = []

best_k = 5  # Use K=5 consistently

print(f"\n  Using K = {best_k}")
print(f"\n  {'% Train':>8s} | {'N':>5s} | {'Mean (m)':>9s} | {'Median (m)':>11s} | {'90th % (m)':>11s}")
print(f"  {'-'*60}")

for pct in percentages:
    n = max(best_k, int(pct / 100 * len(full_train)))
    
    X_tr = full_train[feature_cols].values[:n]
    y_tr = full_train[target_cols].values[:n]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_test)
    
    knn = KNeighborsRegressor(
        n_neighbors=min(best_k, n),
        weights='distance',
        metric='euclidean'
    )
    knn.fit(X_tr_s, y_tr)
    
    y_pred = knn.predict(X_te_s)
    errors = np.sqrt(np.sum((y_pred - y_test)**2, axis=1))
    
    train_sizes.append(n)
    mean_errors.append(errors.mean())
    median_errors.append(np.median(errors))
    p90_errors.append(np.percentile(errors, 90))
    
    print(f"  {pct:>6d}% | {n:>5d} | {errors.mean():>9.4f} | {np.median(errors):>11.4f} | {np.percentile(errors, 90):>11.4f}")

# ============================================================================
# STEP 3: Generate Bar Plot
# ============================================================================
print("\n Generating bar plot...")

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(train_sizes))
bar_width = 0.3

bars1 = ax.bar(x - bar_width, mean_errors, bar_width, 
               label='Mean Error', color='#4472C4', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, median_errors, bar_width, 
               label='Median Error', color='#ED7D31', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + bar_width, p90_errors, bar_width, 
               label='90th Percentile Error', color='#A5A5A5', edgecolor='white', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Training Set Size (number of examples)', fontsize=12, fontweight='bold')
ax.set_ylabel('Localization Error (meters)', fontsize=12, fontweight='bold')
ax.set_title('KNN Localization Accuracy vs Training Set Size\n(K=5, Distance-Weighted, Euclidean)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{n}\n({p}%)' for n, p in zip(train_sizes, percentages)], fontsize=9)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(p90_errors) * 1.3)

plt.tight_layout()
plt.savefig('accuracy_vs_training_size.png', dpi=150, bbox_inches='tight')

print(f"  Plot saved to 'bonus_accuracy_vs_training_size.png'")
