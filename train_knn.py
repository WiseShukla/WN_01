"""
Question 6: KNN-based Localization Algorithm - Training Script
==============================================================
Author: Adarsh Shukla (PhD25001)
Course: Wireless Networks, IIIT Delhi

Created this python script to trains a K-Nearest Neighbor (KNN) model for radio client
localization using distance measurements from 8 anchor nodes.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 70)
print("KNN LOCALIZATION - TRAINING")
print("=" * 70)

# ============================================================================
# STEP 1: Load Training Data
# ============================================================================
print("\n  Loading training data...")

train_df = pd.read_csv('datatrain.csv')
print(f"  Loaded {len(train_df)} training examples")

# Define feature columns (distances to 8 anchors)
feature_cols = ['Dist_A0', 'Dist_A1', 'Dist_A2', 'Dist_A3',
                'Dist_A4', 'Dist_A5', 'Dist_A6', 'Dist_A7']

# Define target columns (true 3D coordinates)
target_cols = ['Target_X', 'Target_Y', 'Target_Z']

X_train = train_df[feature_cols].values
y_train = train_df[target_cols].values

print(f"  Features: {X_train.shape[0]} samples × {X_train.shape[1]} anchors")
print(f"  Targets: {y_train.shape[1]} coordinates (X, Y, Z)")

# ============================================================================
# STEP 2: Standardize Features
# ============================================================================
print("\n Standardizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"  Features scaled (mean ≈ 0, std ≈ 1)")

# ============================================================================
# STEP 3: Find Optimal K via Cross-Validation
# ============================================================================
print("\n  Finding optimal K using 5-fold cross-validation...")

k_values = range(1, 21)
cv_scores = []

print(f"\n  {'K':>4s}  {'CV RMSE (m)':>12s}")
print(f"  {'-'*20}")

for k in k_values:
    knn = KNeighborsRegressor(
        n_neighbors=k,
        weights='distance',
        metric='euclidean'
    )
    
    scores = cross_val_score(knn, X_train_scaled, y_train, 
                            cv=5, scoring='neg_mean_squared_error')
    
    mean_rmse = np.sqrt(-scores.mean())
    cv_scores.append(mean_rmse)
    
    print(f"  {k:>4d}  {mean_rmse:>12.4f}")

best_k = k_values[np.argmin(cv_scores)]
best_rmse = min(cv_scores)

print(f"\n  Best K = {best_k} (CV RMSE = {best_rmse:.4f} m)")

# ============================================================================
# STEP 4: Train Final Model
# ============================================================================
print(f"\n Training final KNN model with K = {best_k}...")

final_model = KNeighborsRegressor(
    n_neighbors=best_k,
    weights='distance',
    metric='euclidean'
)
final_model.fit(X_train_scaled, y_train)

print(f"  Model trained successfully")

# ============================================================================
# STEP 5: Evaluate on Training Set
# ============================================================================
print("\n Evaluating training set performance...")

y_train_pred = final_model.predict(X_train_scaled)
train_errors = np.sqrt(np.sum((y_train_pred - y_train)**2, axis=1))

print(f"\n  Training Set Localization Error:")
print(f"    Mean:            {train_errors.mean():.4f} m")
print(f"    Median:          {np.median(train_errors):.4f} m")
print(f"    90th percentile: {np.percentile(train_errors, 90):.4f} m")

# ============================================================================
# STEP 6: Save Model
# ============================================================================
print("\n Saving trained model...")

with open('knn_model.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'scaler': scaler,
        'best_k': best_k
    }, f)

print(f"  Model will be saved to 'knn_model.pkl'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
