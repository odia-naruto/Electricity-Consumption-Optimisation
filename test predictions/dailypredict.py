import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Load and process data
combine_data = pd.read_csv('combined_data.csv')
combine_data['time'] = pd.to_datetime(combine_data['time'])

# Aggregate data to monthly level to create the correlation matrix for clustering
monthly_energy = combine_data.groupby(['apartment_id', combine_data["time"].dt.month])['energy_consumption'].sum()
monthly_energy = monthly_energy.unstack(level=0).fillna(0)

# Compute correlation matrix for clustering
monthly_corr = monthly_energy.corr()

# Clustering function
def perform_clustering(corr_matrix, n_clusters=3):
    # Apply K-Means clustering on the correlation matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(corr_matrix)
    
    # Map apartments to their clusters
    apartment_clusters = {apartment: clusters[i] for i, apartment in enumerate(corr_matrix.columns)}
    
    return apartment_clusters

# Cluster assignment for each apartment
apartment_clusters = perform_clustering(monthly_corr)

# Function to pull relevant data and train models for each cluster
def daily_cluster_specific_modeling(combine_data, apartment_clusters, n_clusters=3):
    cluster_models = {}
    
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        # Filter data for relevant apartments in this cluster
        cluster_data = combine_data[combine_data['apartment_id'].isin(cluster_apartments)]
        
        # Resample data to daily aggregations for each apartment
        cluster_data.set_index('time', inplace=True)
        
        daily_data = cluster_data.groupby('apartment_id')['energy_consumption'].resample('D').sum().reset_index()
        
        # Feature engineering - adding lagged features to daily data
        daily_data = add_lagged_features(daily_data, lags=[1, 7])
        
        # Drop rows with NaN values from lagged features
        daily_data = daily_data.dropna()
        
        # Prepare training and testing sets for daily data
        X_daily = daily_data[['lag_1', 'lag_7']]
        y_daily = daily_data['energy_consumption']
        X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(X_daily, y_daily, test_size=0.2, random_state=42)
        
        # Train a Kernel Ridge Regression model for daily data
        daily_model = fit_kernel_ridge_model(X_train_daily, y_train_daily, X_test_daily, y_test_daily)
        
        # Store models for each cluster
        cluster_models[cluster] = {'daily_model': daily_model}
    
    return cluster_models

# Function to add lagged features
def add_lagged_features(combine_data, lags=[1, 7]):
    for lag in lags:
        combine_data[f'lag_{lag}'] = combine_data.groupby('apartment_id')['energy_consumption'].shift(lag)
    return combine_data

# Function to fit Kernel Ridge Regression model
def fit_kernel_ridge_model(X_train, y_train, X_test, y_test, alpha=0.1, gamma=0.9):
    # Initialize and fit the Kernel Ridge Regression model
    kr_model = KernelRidge(alpha=alpha, gamma=gamma)
    kr_model.fit(X_train, y_train)
    
    # Evaluate the model
    score = kr_model.score(X_test, y_test)
    print(f"Kernel Ridge Regression Model Score: {score}")
    
    return kr_model

# Train models for each cluster
cluster_models = daily_cluster_specific_modeling(combine_data, apartment_clusters)

# Visualize the model performance
def visualize_model_performance(cluster_models, combine_data, apartment_clusters, n_clusters=3):
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        if cluster_apartments:
            apartment = cluster_apartments[0]  # Select one apartment from the cluster
            # Filter data for the relevant apartment in this cluster
            apartment_data = combine_data[combine_data['apartment_id'] == apartment]
            
            # Resample data to daily aggregations for the apartment
            apartment_data.set_index('time', inplace=True)
            daily_data = apartment_data['energy_consumption'].resample('D').sum().reset_index()
            
            # Ensure daily_data has apartment_id as a column
            daily_data['apartment_id'] = apartment
            # Feature engineering - adding lagged features to daily data
            daily_data = add_lagged_features(daily_data, lags=[1, 7])
            
            # Drop rows with NaN values from lagged features
            daily_data = daily_data.dropna()
            
            # Prepare data for daily model visualization
            X_daily = daily_data[['lag_1', 'lag_7']]
            y_daily = daily_data['energy_consumption']
            daily_model = cluster_models[cluster]['daily_model']
            y_pred_daily = daily_model.predict(X_daily)
            
            # Plot daily model performance for the apartment
            plt.figure(figsize=(14, 6))
            plt.plot(daily_data['time'], y_daily, label='Actual Daily Consumption')
            plt.plot(daily_data['time'], y_pred_daily, label='Predicted Daily Consumption')
            plt.title(f'Cluster {cluster} Apartment {apartment} Daily Model Performance')
            plt.xlabel('Time')
            plt.ylabel('Energy Consumption')
            plt.legend()
            plt.show()

# Visualize the model performance for each apartment in each cluster
visualize_model_performance(cluster_models, combine_data, apartment_clusters)