import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and process dataw
combined_data = pd.read_csv('combined_data.csv')
combined_data['time'] = pd.to_datetime(combined_data['time'])

# Aggregate data to monthly level to create the correlation matrix for clustering
monthly_energy = combined_data.groupby(['apartment_id', combined_data["time"].dt.month])['energy_consumption'].sum()
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
def weekly_cluster_specific_modeling(combined_data, apartment_clusters, n_clusters=3):
    cluster_models = {}
    
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        # Filter data for relevant apartments in this cluster
        cluster_data = combined_data[combined_data['apartment_id'].isin(cluster_apartments)]
        
        # Resample data to daily and weekly aggregations for each apartment
        cluster_data.set_index('time', inplace=True)
        
        daily_data = cluster_data.groupby('apartment_id')['energy_consumption'].resample('D').sum().reset_index()
        weekly_data = cluster_data.groupby('apartment_id')['energy_consumption'].resample('W').sum().reset_index()
        
        # Feature engineering - adding lagged features to daily and weekly data
        daily_data = add_lagged_features(daily_data, lags=[1, 7])
        weekly_data = add_lagged_features(weekly_data, lags=[1, 4])
        
        # Drop rows with NaN values from lagged features
        daily_data = daily_data.dropna()
        weekly_data = weekly_data.dropna()
        
        # Prepare training and testing sets for daily data
        X_daily = daily_data[['lag_1', 'lag_7']]
        y_daily = daily_data['energy_consumption']
        X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(X_daily, y_daily, test_size=0.2, random_state=42)
        
        # Train a Kernel Ridge Regression model for daily data
        daily_model = fit_kernel_ridge_model(X_train_daily, y_train_daily, X_test_daily, y_test_daily)
        
        # Prepare training and testing sets for weekly data
        X_weekly = weekly_data[['lag_1', 'lag_4']]
        y_weekly = weekly_data['energy_consumption']
        X_train_weekly, X_test_weekly, y_train_weekly, y_test_weekly = train_test_split(X_weekly, y_weekly, test_size=0.2, random_state=42)
        
        # Train a Kernel Ridge Regression model for weekly data
        weekly_model = fit_kernel_ridge_model(X_train_weekly, y_train_weekly, X_test_weekly, y_test_weekly)
        
        # Store models for each cluster
        cluster_models[cluster] = {'daily_model': daily_model, 'weekly_model': weekly_model}
    
    return cluster_models

# Function to add lagged features
def add_lagged_features(combined_data, lags=[1, 7]):
    for lag in lags:
        combined_data[f'lag_{lag}'] = combined_data.groupby('apartment_id')['energy_consumption'].shift(lag)
    return combined_data

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
cluster_models = weekly_cluster_specific_modeling(combined_data, apartment_clusters)

# Visualize the model performance
def visualize_model_performance(cluster_models, combined_data, apartment_clusters, n_clusters=3):
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        if cluster_apartments:
            apartment = cluster_apartments[0]  # Select one apartment from the cluster
            # Filter data for the relevant apartment in this cluster
            apartment_data = combined_data[combined_data['apartment_id'] == apartment]
            
            # Resample data to weekly aggregations for the apartment
            apartment_data.set_index('time', inplace=True)
            weekly_data = apartment_data['energy_consumption'].resample('W').sum().reset_index()
            
            # Ensure weekly_data has apartment_id as a column
            weekly_data['apartment_id'] = apartment
            # Feature engineering - adding lagged features to weekly data
            weekly_data = add_lagged_features(weekly_data, lags=[1, 4])
            
            # Drop rows with NaN values from lagged features
            weekly_data = weekly_data.dropna()
            
            # Prepare data for weekly model visualization
            X_weekly = weekly_data[['lag_1', 'lag_4']]
            y_weekly = weekly_data['energy_consumption']
            weekly_model = cluster_models[cluster]['weekly_model']
            y_pred_weekly = weekly_model.predict(X_weekly)
            
            # Plot weekly model performance for the apartment
            plt.figure(figsize=(14, 6))
            plt.plot(weekly_data['time'], y_weekly, label='Actual Weekly Consumption')
            plt.plot(weekly_data['time'], y_pred_weekly, label='Predicted Weekly Consumption')
            plt.title(f'Cluster {cluster} Apartment {apartment} Weekly Model Performance')
            plt.xlabel('Time')
            plt.ylabel('Energy Consumption')
            plt.legend()
            plt.show()

# Visualize the model performance for each apartment in each cluster
visualize_model_performance(cluster_models, combined_data, apartment_clusters)
