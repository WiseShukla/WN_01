"""
Question 6: KNN-based Localization Algorithm - Testing Script
=============================================================
Author: Adarsh Shukla (PhD25001)
Course: Wireless Networks, IIIT Delhi

Created this python script that loads the trained KNN model and predicts 3D locations
for the test set, saving results to 'test_predictions.csv'.
"""

import pandas as pd
import pickle

print("=" * 70)
print("KNN LOCALIZATION - TESTING")
print("=" * 70)

# ============================================================================
# STEP 1: Load Trained Model
# ============================================================================
print("\n Loading trained model...")

with open('knn_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
scaler = saved['scaler']
best_k = saved['best_k']

print(f"  Model loaded (K = {best_k})")

# ============================================================================
# STEP 2: Load Test Data
# ============================================================================
print("\n Loading test data...")

test_df = pd.read_csv('datatest.csv')

feature_cols = ['Dist_A0', 'Dist_A1', 'Dist_A2', 'Dist_A3',
                'Dist_A4', 'Dist_A5', 'Dist_A6', 'Dist_A7']

X_test = test_df[feature_cols].values

print(f"  Loaded {len(X_test)} test examples")

# ============================================================================
# STEP 3: Scale Features and Predict
# ============================================================================
print("\n Scaling features and predicting locations...")

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

print(f"  Predicted {len(y_pred)} locations")
print(f"\n  Sample predictions (first 5):")
print(f"    {'Target_X':>10s}  {'Target_Y':>10s}  {'Target_Z':>10s}")
print(f"    {'-'*36}")
for i in range(min(5, len(y_pred))):
    print(f"    {y_pred[i, 0]:10.4f}  {y_pred[i, 1]:10.4f}  {y_pred[i, 2]:10.4f}")

# ============================================================================
# STEP 4: Save Predictions
# ============================================================================
print("\n Saving predictions to CSV...")

output_df = pd.DataFrame(y_pred, columns=['Target_X', 'Target_Y', 'Target_Z'])
output_df.to_csv('test_predictions.csv', index=False)

print(f"  Saved to 'test_predictions.csv'")
print(f"\n  Prediction summary:")
print(output_df.describe().round(4).to_string())
print(f"\nOutput file: test_predictions.csv ({len(output_df)} predictions)")