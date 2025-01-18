# Energy Consumption Clustering and Prediction Project

This was the course project of M Kartik (EC22B1013) and J Sai Sasank (EC22B1087) and 
this project focuses on clustering apartments based on their monthly energy consumption patterns and building predictive models to forecast future energy usage. The goal is to assist power plants in anticipating energy demand and optimizing their production schedules.

## Project Overview

1. **Data Clustering**:
   - Monthly energy consumption data is used to compute a correlation matrix.
   - K-Means clustering is applied to group apartments with similar energy consumption patterns.

2. **Modeling**:
   - Separate predictive models are trained for each cluster.
   - Kernel Ridge Regression is used to predict monthly energy consumption based on lagged features.

3. **Visualization**:
   - Model performance for each cluster is visualized by comparing actual vs. predicted energy consumption.

## Dataset

The dataset consists of the following columns:
- `time`: Timestamps for energy consumption data.
- `apartment_id`: Unique identifier for each apartment.
- `energy_consumption`: Energy consumption values for each timestamp.

### Preprocessing Steps
1. Convert `time` column to a datetime format.
2. Aggregate energy consumption data at the monthly level.
3. Compute the correlation matrix for clustering purposes.
4. Create lagged features for predictive modeling.

## Project Structure

- **Data Preparation**:
  Aggregation and feature engineering to create a dataset suitable for clustering and regression modeling.

- **Clustering**:
  K-Means clustering is performed on the correlation matrix derived from monthly energy consumption.

- **Model Training**:
  For each cluster:
  - Data is resampled to monthly levels.
  - Lagged features are created (e.g., `lag_1`, `lag_3`).
  - Kernel Ridge Regression models are trained to predict energy consumption.

- **Evaluation**:
  Model performance is evaluated using the \( R^2 \) score.

## Key Functions

### 1. `perform_clustering(corr_matrix, n_clusters=3)`
- **Input**: Correlation matrix, number of clusters.
- **Output**: Dictionary mapping apartments to their respective clusters.

### 2. `monthly_cluster_specific_modeling(df, apartment_clusters, n_clusters=3)`
- **Input**: Dataset, cluster assignments, number of clusters.
- **Output**: Dictionary containing trained models for each cluster.

### 3. `add_lagged_features(df, lags=[1, 3])`
- **Input**: Dataset, list of lags.
- **Output**: Dataset with additional lagged feature columns.

### 4. `fit_kernel_ridge_model(X_train, y_train, X_test, y_test, alpha=0.5, gamma=0.1)`
- **Input**: Training and test data, hyperparameters.
- **Output**: Trained Kernel Ridge Regression model.

### 5. `visualize_model_performance(cluster_models, df, apartment_clusters, n_clusters=3)`
- **Input**: Cluster models, dataset, cluster assignments.
- **Output**: Plots of actual vs. predicted energy consumption for each cluster.

## Dependencies

- Python 3.8+
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## How to Run

1. **Prepare the Data**:
   Ensure the dataset is available as `combined_data.csv` and contains the required columns.

2. **Run the Code**:
   Execute the Python script to:
   - Perform clustering.
   - Train models for each cluster.
   - Visualize model performance.

3. **Evaluate Results**:
   Check the \( R^2 \) score and the visualized plots to assess model performance.

## Challenges and Improvements

### Current Challenges
- Negative \( R^2 \) scores may indicate underfitting or suboptimal feature engineering.
- Clustering may not effectively group apartments with similar patterns if the correlation matrix is insufficient.

### Future Improvements
1. Incorporate additional features (e.g., weather data, time of day).
2. Test alternative models, such as Gradient Boosting or LSTM for time series predictions.
3. Perform grid search for hyperparameter tuning.
4. Experiment with different aggregation levels (e.g., weekly data).

## Acknowledgments
This project uses energy consumption data to explore clustering and predictive modeling techniques, with the goal of improving demand forecasting for power plants.

