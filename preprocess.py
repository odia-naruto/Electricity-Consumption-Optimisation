import os
import pandas as pd

# Set the path to the data directory
data_path = "data"  # Change this to the actual path where your folders are stored

# Create an empty list to store individual apartment dataframes
all_data = []

# Loop through each year directory (2014, 2015, 2016)
for year in ["2015"]:
    year_path = os.path.join(data_path, year)  # Path to the current year's folder
    
    # Check if the year path exists
    if not os.path.exists(year_path):
        continue  # Skip if the folder does not exist
    
    # Loop through each CSV file in the year's folder
    for file_name in os.listdir(year_path):
        if file_name.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(year_path, file_name)
            
            # Load the CSV file into a DataFrame without headers
            df = pd.read_csv(file_path, header=None)
            
            # Assign column names to the DataFrame
            df.columns = ["time", "energy_consumption"]  # Set the column names
            
            # Filter out rows where energy consumption is zero
            df = df[df["energy_consumption"] != 0]
            
            # Convert 'time' column to datetime format
            df["time"] = pd.to_datetime(df["time"])  # Adjust the format if needed
            
            # Add columns for year and apartment_id (extracted from file name)
            df["apartment_id"] = file_name.split('_')[0]  # Extract apartment ID from the file name
            df["apartment_id"] = df["apartment_id"].str.extract(r'(\d+)')
            # Append the filtered DataFrame to the list
            all_data.append(df)

# Combine all individual DataFrames into one DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(combined_data.head())
combined_data.to_csv('combined_data.csv', index=False)