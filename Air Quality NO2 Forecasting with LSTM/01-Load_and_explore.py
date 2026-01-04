"""
Air Quality Prediction with LSTM

Problem Definition:

- Air pollution is harmful to human health. Nitrogen dioxide significantly affects human health, especially in cities.
- Let's predict nitrogen dioxide using environmental factors such as past temperature, humidity, carbon monoxide, etc.
- Let's use LSTM within a multivariate time series structure.

Dataset:
- https://archive.ics.uci.edu/dataset/360/air+quality
- Air Quality UCI
- Time interval: 1 year
- Frequency: hourly measurements
- temperature, new, sensor outputs, time information, etc.
- Prediction: nitrogen dioxide

Technologies/Tools:
- LSTM: pytorch
- FastAPI: web service
- Streamlit: user interface

Plan/Schedule:
- 01-load_and_explore.py: data loading, missing data analysis, format conversions, and basic visualization
- 02-preprocessing.py: Input and target variable definition, missing data correction via interpolation, normalization, sliding window training/test split
- 03-train.py: LSTM model training
- 04-test.py: testing
- 05-main_api.py: FastAPI with /predict endpoint
- 06-test_api.py: service tests
- 07-app_streamlit.py: UI (user interface)

install libraries: freeze
pip install pandas numpy matplotlib seaborn scikit-learn torch fastapi uvicorn streamlit requests
"""
import pandas as pd  # Data loading/manipulation
import numpy as np  # Numerical operations (NaN handling, arrays)
import matplotlib.pyplot as plt  # Plotting (time series, figures)
import seaborn as sns  # Statistical plotting (correlation heatmap)

# Load the CSV file  # Read the dataset from disk
# We use sep=';' because the file is semicolon-separated  # Dataset delimiter
# We use decimal=',' because decimals are represented with comma  # European decimal format
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",", encoding="latin1")  # Read raw data
print(df.head())  # Quick sanity check (first rows)

# Drop fully empty columns  # UCI file often contains trailing empty columns
df.dropna(axis=1, how="all", inplace=True)  # Remove columns where all values are NaN

# Merge Date and Time columns and convert to a datetime object  # Create a proper time index
df["datetime"] = pd.to_datetime(  # Parse combined date-time string
    df["Date"] + " " + df["Time"],  # Concatenate date and time text
    format="%d/%m/%Y %H.%M.%S",  # Expected date-time format in the dataset
    errors="coerce",  # Invalid parses become NaT (then we drop them)
)
print(df.head())  # View with datetime column
print(df.info())  # Inspect dtypes / missingness

# Drop rows with invalid datetime parsing  # Remove corrupted rows
df.dropna(subset=["datetime"], inplace=True)  # Keep only valid timestamps

# Drop original Date and Time columns  # We already have datetime
df.drop(["Date", "Time"], inplace=True, axis=1)  # Remove unused columns
print(df.head())  # Verify columns

# Set datetime as index  # Enable time-based interpolation and plotting
df.set_index("datetime", inplace=True)  # Make time the index
print(df.head())  # Confirm index set

# Sensor errors are coded as -200; convert them to NaN  # Treat as missing
df.replace(-200, np.nan, inplace=True)  # Replace invalid sentinel values

# Fill missing values via time-based interpolation  # Smoothly fill gaps along timeline
df.interpolate(method="time", inplace=True)  # Interpolate using the datetime index

# Feature engineering: add time components as new features  # Useful for seasonality patterns
df["hour"] = df.index.hour  # Hour of day (0-23)
df["month"] = df.index.month  # Month of year (1-12)
print(df.head())  # Confirm engineered features

# Select input and target columns (target: NO2(GT))  # Build model-ready dataframe
selected_columns = ["NO2(GT)", "T", "RH", "AH", "CO(GT)", "hour", "month"]  # Target + predictors
df = df[selected_columns]  # Keep only selected features
print(df.head())  # Verify selection

# Check remaining missing values (percentage)  # Basic missingness audit
print(f"Eksik deger: {df.isnull().mean()*100}")  # Missing ratio per column (in %)

# Build a correlation matrix  # Explore linear relationships
plt.figure()  # Create a new figure
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="YlGnBu")  # Plot correlation heatmap
plt.title("Korelasyon Matrisi")  # Title
plt.show()  # Display

# Plot the target time series (NO2)  # Visual inspection of dynamics and trends
plt.figure()  # New figure
df["NO2(GT)"].plot()  # Plot NO2 over time
plt.title("NO2(GT) zaman serisi")  # Title
plt.xlabel("Tarih")  # X label
plt.ylabel("NO2(GT)")  # Y label
plt.show()  # Display