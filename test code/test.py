import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and process data
df = pd.read_csv('combined_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Aggregate data by time intervals
monthly_energy = df.groupby(['apartment_id', df["time"].dt.month])['energy_consumption'].sum()
weekly_energy = df.groupby(['apartment_id', df["time"].dt.isocalendar().week])['energy_consumption'].sum()
daily_energy = df.groupby(['apartment_id', df["time"].dt.dayofyear])['energy_consumption'].sum()

# Reshape data for correlation
monthly_energy = monthly_energy.unstack(level=0).fillna(0)
weekly_energy = weekly_energy.unstack(level=0).fillna(0)
daily_energy = daily_energy.unstack(level=0).fillna(0)

# Compute correlation matrices
monthly_corr = monthly_energy.corr()
weekly_corr = weekly_energy.corr()
daily_corr = daily_energy.corr()



# Visualize correlation matrices
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(monthly_corr, annot=False, cmap='coolwarm')
plt.title('Monthly Correlation Heatmap')

plt.subplot(1, 3, 2)
sns.heatmap(weekly_corr, annot=False, cmap='coolwarm')
plt.title('Weekly Correlation Heatmap')

plt.subplot(1, 3, 3)
sns.heatmap(daily_corr, annot=False, cmap='coolwarm')
plt.title('Daily Correlation Heatmap')

plt.tight_layout()
plt.show()
