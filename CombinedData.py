#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:09:49 2024

@author: ivieirivbogbe
"""
import pandas as pd
import os

# Set the working directory
os.chdir('/Users/ivieirivbogbe/Desktop/SPRING 2024/Forecasting Methods - 202410 BANA-7350-1QA/Project')

# Load the datasets
df_natural_gas = pd.read_csv('NATURALGASD11.csv', parse_dates=['DATE'], index_col='DATE')
df_cpi_trans = pd.read_csv('CPIETRANS.csv', parse_dates=['DATE'], index_col='DATE')
df_vehicle_miles = pd.read_csv('TRFVOLUSM227SFWA.csv', parse_dates=['DATE'], index_col='DATE')

# Combine the dataframes
df_combined = df_natural_gas.join([df_cpi_trans, df_vehicle_miles], how='outer')

# Rename the columns
df_combined.rename(columns={
    'NATURALGASD11': 'NaturalGasConsumption',
    'CPIETRANS': 'CPITransportation',
    'TRFVOLUSM227SFWA': 'VehicleMilesTraveled'
}, inplace=True)

# Find the latest start date among all datasets
latest_start_date = max(df_natural_gas.index.min(), df_cpi_trans.index.min(), df_vehicle_miles.index.min())

# Truncate the combined DataFrame to start from the latest start date
df_combined = df_combined[df_combined.index >= latest_start_date]

# Drop rows with any null values
df_combined.dropna(inplace=True)

# Save the combined DataFrame to a CSV file in the specified folder
file_name = 'CombinedData.csv'
full_path = os.path.join('/Users/ivieirivbogbe/Desktop/SPRING 2024/Forecasting Methods - 202410 BANA-7350-1QA/Project', file_name)
df_combined.to_csv(full_path)

print(f"Combined DataFrame with no null values, starting from {latest_start_date}, saved to {full_path}")
