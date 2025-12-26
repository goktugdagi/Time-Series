"""
LSTM-Based Energy Consumption Forecasting

Problem Definition:
    - Energy consumption forecasting for power-generation planning, grid load balancing, demand prediction,
      billing optimization, and network stability.
    - Predicting future electricity consumption using historical usage patterns.
    - Objective: Build a decision-support forecasting system that estimates forward-looking household energy demand
      from past consumption signals.

Dataset:
    - UCI Individual Household Electric Power Consumption Dataset
    - Target feature: `global_active_power` (total active power in kilowatts)

Tools & Technologies:
    - LSTM (TensorFlow/Keras):
        * A special type of Recurrent Neural Network (RNN)
        * Designed to solve vanishing gradient problems in long sequences
        * Uses gated memory units: Forget Gate, Input Gate, Output Gate

Project Plan:
    1. Load and clean raw data
    2. Resample into hourly aggregates
    3. Create sliding window sequences
    4. Normalize using MinMax scaling
    5. Train LSTM model with validation split
    6. Evaluate model on the test set
    7. Forecast future 24-hour consumption
"""

import pandas as pd  # Data processing and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization library for plotting graphs

# Load the dataset downloaded from UCI repository
df = pd.read_csv(
    "household_power_consumption.txt",  # Dataset file
    sep=";",  # Columns are separated by semicolons
    na_values="?",  # Missing values are marked as '?' so convert them to NaN
    low_memory=False  # Disable chunk processing to avoid dtype inconsistencies
)

# Combine Date and Time columns into a single datetime column
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],  # Merge columns as string
    dayfirst=True,  # Use European date format (DD/MM/YYYY)
    errors="coerce"  # Invalid parsing will return NaT instead of breaking
)

# Set datetime column as the DataFrame index
df.set_index("datetime", inplace=True)

# Remove original Date and Time columns after merging
df.drop(columns=["Date", "Time"], inplace=True)

# Convert Global_active_power column to numeric values
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],  # Target column
    errors="coerce"  # If conversion fails, replace with NaN
)

# Drop rows that contain NaN in Global_active_power to avoid model training errors
df = df.dropna(subset=["Global_active_power"])

# Resample the time series into hourly averages
df_hourly = df["Global_active_power"].resample("H").mean()

# Display first rows for validation
print(df_hourly.head())

# Plot the hourly energy consumption time series
plt.figure()
plt.plot(df_hourly, label="Hourly Energy Consumption (kW)")
plt.title("Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("kW")
plt.legend()
plt.grid(True)
plt.show()

# Save the cleaned and resampled dataset for later steps
df_hourly.to_csv("df_hourly.csv")