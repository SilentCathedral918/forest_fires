# Forest Fire Prediction - Project Documentation

## Project Overview

This project focuses on predicting the burned area of forests using data from the UCI Forest Fires dataset. 
The primary goal is to capture the relationship between various environmental features and the area burned by forest fires. 
The data is highly skewed and contains outliers, which influenced the choice of model and preprocessing steps.

## Process Overview

### 1. **Data Preparation**
   - **Initial Dataset:** The dataset from UCI provides well-prepared `X` (features) and `y` (target). However, the target (`area` of the burn) is not included in `X`, so it was included in `X` for visualization and to capture trends.
   - **Visualization:** Using `sns.pairplot`, relationships between each feature and the target (burned area) were explored. This helped understand which features have a significant correlation with the target.

### 2. **Outlier Detection and Filtering**
   - **Identifying Outliers:** Outliers were visually identified in the dataset, which created a skewed trend for the model. This is important for improving model training by focusing on more relevant data points.
   - **Filtering Outliers:** A set of conditions was applied to remove outliers and ensure the training process would not be influenced by extreme values.

### 3. **Feature Selection**
   - **Selecting Important Features:** The most promising features that correlate with the burned area were selected for model training. These features were identified through exploratory analysis and visualization.
   - **Creating Prepared Dataset:** A final prepared dataset (`X_prepared` for features and `y_prepared` for target) was constructed based on the selected features.

### 4. **Data Splitting**
   - **Percentile-Based Splitting:** Since the dataset was highly skewed, a decision was made to split it into five sets based on percentiles ([0, 50, 70, 87, 98, 100]) to handle different ranges of the target variable. This step ensures the model doesn't overfit on the skewed data.
   - **Train/Test Split:** Each subset was split into training (80%) and testing (20%) sets to validate the model's performance.

### 5. **Model Selection and Experimentation**
   - **Model Exploration:** Various regression models were tested for each subset, as the data required different models for different ranges of the target.
     - **Set 1 & Set 2:** Linear Regression (for the less skewed data).
     - **Set 3 & Set 4:** Support Vector Regression (SVR, fine-tuned with hyperparameters for better fitting).
     - **Set 5:** Huber Regressor (to handle semi-outliers and robust regression).
   - **Evaluation:** The models were evaluated on each subset using Mean Squared Error (MSE) and R² scores. This process was iterative, with adjustments made to the models based on their performance.

### 6. **Final Model Evaluation**
   - **Combining Predictions:** The final step involved combining predictions from all subsets, ensuring that the correct model was applied to each sample.
   - **Performance Evaluation:** The overall performance of the combined predictions was evaluated using MSE and R². The final model achieved the following evaluation metrics:
     - **Overall MSE:** 50.2654
     - **Overall R²:** 0.9361

## Conclusion

The project demonstrates how to handle skewed datasets and outliers in regression problems. 
By splitting the data based on percentiles, applying different models to subsets of the data, and evaluating them iteratively, the project showcases a comprehensive approach to predicting the burned area of forest fires.

