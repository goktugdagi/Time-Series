"""
Traffic Volume Forecasting with GRU (Gated Recurrent Unit)

Problem definition: Predict traffic volume for upcoming hours based on historical traffic data.

Data: https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    - 2012 - 2018
    - hourly measurements
    - about 48,000 samples
    - target variable: traffic_volume
    - features:
        - date_time (time), holiday, temp (temperature), rain/snow, clouds_all,
        - weather_main (general weather condition)

Technologies/Tools:
    - PyTorch: GRU-based time series model
    - FastAPI: web server to serve the model as a REST API
    - Streamlit: web-based user interface

Workflow/Plan:
    - data analysis (1_load_and_explore.py)
    - preprocessing (2_preprocessing.py)
    - model training (3_train.py)
    - testing and evaluation (4_test.py)
    - FastAPI deployment (5_main_api.py)
    - FastAPI test client (6_test_requests.py)
    - Streamlit UI (7_app_streamlit.py)

Install libraries:
pip install pandas numpy matplotlib seaborn scikit-learn torch fastapi uvicorn streamlit
"""

import pandas as pd  # Import pandas for data processing and analysis
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import seaborn as sns  # Import seaborn for statistical visualizations

# Load the dataset
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")  # Read CSV file into a DataFrame
print(df.head())  # Print the first 5 rows to quickly verify the dataset structure

# Print general DataFrame information (rows/cols, dtypes, missing values, etc.)
print(df.info())  # Display schema and missing-value summary

# Check missing values per column
print(df.isnull().sum())  # Count missing values in each column

# Basic statistical summary for numerical columns
print(df.describe())  # Show summary statistics (mean, std, min/max, quartiles)

# Convert time column to datetime
df["date_time"] = pd.to_datetime(df["date_time"])  # Convert date_time from string to datetime
df.set_index("date_time", inplace=True)  # Set date_time as index for time series operations
print(df.head())  # Print head again to confirm index conversion

# Visualize the time series
plt.figure()  # Create a new figure
plt.plot(df["traffic_volume"], label="Traffic Volume")  # Plot traffic volume over time
plt.title("Traffic Volume Time Series")  # Set plot title
plt.xlabel("Date")  # Set x-axis label
plt.ylabel("Traffic Volume")  # Set y-axis label
plt.legend()  # Display legend
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Render the plot

# Hourly average traffic volume
df["hour"] = df.index.hour  # Extract hour from datetime index and create a new column

# Compute mean traffic volume by hour
hourly_avg = df.groupby("hour")["traffic_volume"].mean()  # Average traffic volume for each hour

# Show hourly averages as a bar plot (reveals daily traffic patterns)
plt.figure()  # Create a new figure
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette="viridis")  # Bar plot of hourly averages
plt.title("Average Traffic Volume by Hour of Day")  # Set plot title
plt.xlabel("Hour")  # Set x-axis label
plt.ylabel("Average Traffic")  # Set y-axis label
plt.show()  # Render the plot
