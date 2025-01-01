#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo

# Fetch and clean data
forest_fires = fetch_ucirepo(id=162)
X = forest_fires.data.features
y = forest_fires.data.targets

# include the 'target' as reference for visualization
X['area'] = y

# remove outliers
X_cleaned = X[(X['area'] <= 600) & (X['rain'] <= 6) & (X['FFMC'] > 25)]

# the features that are the most promising to capture the area trend
features_of_interest = ['temp', 'FFMC', 'DC']
X_prepared = X_cleaned[features_of_interest]
y_prepared = X_cleaned['area']

# Calculate percentiles and define subsets
percentiles = np.percentile(y_prepared, [0, 50, 70, 87, 98, 100])
subsets = {
    'set1': (y_prepared <= percentiles[1]),
    'set2': (y_prepared > percentiles[1]) & (y_prepared <= percentiles[2]),
    'set3': (y_prepared > percentiles[2]) & (y_prepared <= percentiles[3]),
    'set4': (y_prepared > percentiles[3]) & (y_prepared <= percentiles[4]),
    'set5': (y_prepared > percentiles[4]),
}

# split into train and test subsets (80% train, 20% test)
X_train_test, y_train_test = {}, {}
for key, mask in subsets.items():
    X_train_test[key], y_train_test[key] = train_test_split(X_prepared[mask], y_prepared[mask], test_size=0.2, random_state=42)

# models to apply for each set
models = {
    'set1': LinearRegression(),
    'set2': LinearRegression(),
    'set3': SVR(),
    'set4': SVR(epsilon=1, gamma=0.1),
    'set5': HuberRegressor(),
}

# perform model trainings
for key, model in models.items():
    model.fit(X_train_test[key][0], y_train_test[key][0])

# store the predictions for full prepared dataset
y_pred_full = []
for i in range(len(X_prepared)):
    X_sample = X_prepared.iloc[i:i+1]  # keeps the DataFrame structure
    if y_prepared.iloc[i] <= percentiles[1]:
        y_pred_full.append(models['set1'].predict(X_sample)[0])
    elif y_prepared.iloc[i] <= percentiles[2]:
        y_pred_full.append(models['set2'].predict(X_sample)[0])
    elif y_prepared.iloc[i] <= percentiles[3]:
        y_pred_full.append(models['set3'].predict(X_sample)[0])
    elif y_prepared.iloc[i] <= percentiles[4]:
        y_pred_full.append(models['set4'].predict(X_sample)[0])
    else:
        y_pred_full.append(models['set5'].predict(X_sample)[0])

# final evaluation of overall performance
mse_full = mean_squared_error(y_prepared, y_pred_full)
r2_full = r2_score(y_prepared, y_pred_full)
print(f"Overall MSE: {mse_full:.4f}, Overall R²: {r2_full:.4f}")
