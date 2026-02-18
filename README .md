# KNN Localization - Question 6

**Author:** Adarsh Shukla (PhD25001)  
**Course:** Wireless Networks, IIIT Delhi

## What This Does

Predicts the 3D location (X, Y, Z) of a radio client based on its distance to 8 fixed anchor nodes using K-Nearest Neighbors.

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Files

- `train_knn.py` - Train the model
- `test_knn.py` - Generate predictions  
- `bonus_plot.py` - Plot accuracy vs training size (optional)
- `datatrain.csv` - Training data (635 examples)
- `datatest.csv` - Test data (424 examples)

## How to Run

### 1. Train the model
```bash
python train_knn.py
```
This finds the best K value using cross-validation and saves the trained model to `knn_model.pkl`.

### 2. Generate predictions
```bash
python test_knn.py
```
Loads the trained model and creates `test_predictions.csv` with predicted coordinates for all 424 test examples.

### 3. BONUS: Plot accuracy vs training size (optional)
```bash
python bonus_plot.py
```
Shows how error decreases as training data increases. Saves plot to `bonus_accuracy_vs_training_size.png`.

## Output Format

`test_predictions.csv` contains 3 columns:
```csv
Target_X,Target_Y,Target_Z
2.5143,3.1872,1.2456
1.8923,2.7654,0.9876
...
```

## How It Works

1. Standardize all distance features (zero mean, unit variance)
2. Find K nearest training examples using Euclidean distance
3. Average their locations with distance weighting (closer = more weight)
4. Best K is found using 5-fold cross-validation

## Expected Result 

1. optimal k value will come in between 3-5

## Troubleshooting

**"File not found"** → Put `datatrain.csv` and `datatest.csv` in same folder as scripts

**"No module named sklearn"** → Run `pip install scikit-learn`

**"knn_model.pkl not found"** → Run `train_knn.py` first before `test_knn.py`

