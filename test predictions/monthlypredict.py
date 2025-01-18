import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Load and process data
df = pd.read_csv('combined_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Aggregate data to monthly level to create the correlation matrix for clustering
monthly_energy = df.groupby(['apartment_id', df["time"].dt.month])['energy_consumption'].sum()
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
def monthly_cluster_specific_modeling(df, apartment_clusters, n_clusters=3):
    cluster_models = {}
    
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        # Filter data for relevant apartments in this cluster
        cluster_data = df[df['apartment_id'].isin(cluster_apartments)]
        
        # Resample data to monthly aggregations for each apartment
        cluster_data.set_index('time', inplace=True)
        
        monthly_data = cluster_data.groupby('apartment_id')['energy_consumption'].resample('ME').sum().reset_index()
        
        # Feature engineering - adding lagged features to monthly data
        monthly_data = add_lagged_features(monthly_data, lags=[1, 3])
        
        # Drop rows with NaN values from lagged features
        monthly_data = monthly_data.dropna()
        
        # Prepare training and testing sets for monthly data
        X_monthly = monthly_data[['lag_1', 'lag_3']]
        y_monthly = monthly_data['energy_consumption']
        X_train_monthly, X_test_monthly, y_train_monthly, y_test_monthly = train_test_split(X_monthly, y_monthly, test_size=0.2, random_state=42)
        
        # Train a Kernel Ridge Regression model for monthly data
        monthly_model = fit_kernel_ridge_model(X_train_monthly, y_train_monthly, X_test_monthly, y_test_monthly)
        
        # Store models for each cluster
        cluster_models[cluster] = {'monthly_model': monthly_model}
    
    return cluster_models

# Function to add lagged features
def add_lagged_features(df, lags=[1, 3]):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('apartment_id')['energy_consumption'].shift(lag)
    return df

# Function to fit Kernel Ridge Regression model
def fit_kernel_ridge_model(X_train, y_train, X_test, y_test, alpha=0.1, gamma=0.1):
    # Initialize and fit the Kernel Ridge Regression model
    kr_model = KernelRidge(alpha=alpha, gamma=gamma)
    kr_model.fit(X_train, y_train)
    
    # Evaluate the model
    score = kr_model.score(X_test, y_test)
    print(f"Kernel Ridge Regression Model Score: {score}")
    
    return kr_model

# Train models for each cluster
cluster_models = monthly_cluster_specific_modeling(df, apartment_clusters)

# Visualize the model performance
def visualize_model_performance(cluster_models, df, apartment_clusters, n_clusters=3):
    for cluster in range(n_clusters):
        # Select apartments belonging to the current cluster
        cluster_apartments = [apt for apt, clus in apartment_clusters.items() if clus == cluster]
        
        if cluster_apartments:
            apartment = cluster_apartments[0]  # Select one apartment from the cluster
            # Filter data for the relevant apartment in this cluster
            apartment_data = df[df['apartment_id'] == apartment]
            
            # Resample data to monthly aggregations for the apartment
            apartment_data.set_index('time', inplace=True)
            monthly_data = apartment_data['energy_consumption'].resample('ME').sum().reset_index()
            
            # Ensure monthly_data has apartment_id as a column
            monthly_data['apartment_id'] = apartment
            # Feature engineering - adding lagged features to monthly data
            monthly_data = add_lagged_features(monthly_data, lags=[1, 3])
            
            # Drop rows with NaN values from lagged features
            monthly_data = monthly_data.dropna()
            
            # Prepare data for monthly model visualization
            X_monthly = monthly_data[['lag_1', 'lag_3']]
            y_monthly = monthly_data['energy_consumption']
            monthly_model = cluster_models[cluster]['monthly_model']
            y_pred_monthly = monthly_model.predict(X_monthly)
            
            # Plot monthly model performance for the apartment
            plt.figure(figsize=(14, 6))
            plt.plot(monthly_data['time'], y_monthly, label='Actual Monthly Consumption')
            plt.plot(monthly_data['time'], y_pred_monthly, label='Predicted Monthly Consumption')
            plt.title(f'Cluster {cluster} Apartment {apartment} Monthly Model Performance')
            plt.xlabel('Time')
            plt.ylabel('Energy Consumption')
            plt.legend()
            plt.show()

# Visualize the model performance for each apartment in each cluster
visualize_model_performance(cluster_models, df, apartment_clusters)