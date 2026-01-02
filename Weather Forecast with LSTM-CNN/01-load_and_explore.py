"""
CNN-LSTM Weather Forecasting

Problem Definition:
- Predicting temperature values ​​by looking at historical weather data
- Extracting local patterns with 1D CNN and modeling time dependencies with LSTM
- Thus, we can make short-term temperature predictions

Data:

- link: https://www.kaggle.com/datasets/muthuj7/weather-dataset
- kaggle: weather history
- time interval: hourly
- value to be predicted: temperature
- humidity, wind_speed_km/h, visibility ...

Technologies and Tools:
- Tensorflow/keras: CNN and LSTM

Workflow/Plan:
    - data analysis (01-load_and_explore.py)
    - preprocessing (02-preprocessing.py)
    - model training (03-train.py)
    - testing and evaluation (04-test.py)
    - FastAPI deployment (05-main_api.py)
    - FastAPI test client (06-test_requests.py)
    - Streamlit UI (07-app_streamlit.py)

install libraries: freeze
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
"""
# Import libraries for data processing and visualization
import pandas as pd  # Used for data manipulation and analysis
import matplotlib.pyplot as plt  # Used for plotting charts
import seaborn as sns  # Used for statistical data visualization

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("weatherHistory.csv")  # Load historical weather dataset
print(df.head())  # Display first 5 rows
print(df.describe().T)  # Display statistical summary (transposed)

# Convert the "Formatted Date" column to datetime format
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])  # Ensure proper datetime parsing

# Set the datetime column as the index of the DataFrame
df.set_index("Formatted Date", inplace=True)  # Use date as index for time-series operations
print(df.head())  # Display first rows after indexing

# Display index metadata for debugging/verification
print("Index type:", type(df.index))  # Print index data type
print("Index dtype:", df.index.dtype)  # Print index dtype
print(df.index[:3])  # Show first 3 index values

# Ensure the index is datetime in UTC timezone, handle invalid entries
df.index = pd.to_datetime(df.index, errors="coerce", utc=True)  # Force UTC and coerce invalid values

# Remove rows where the index could not be parsed
df = df[~df.index.isna()]  # Drop invalid datetime index rows

# Sort index chronologically to maintain temporal order
df = df.sort_index()  # Guarantee ascending time order
print("Final Index type:", type(df.index))  # Confirm index type
print("Timezone:", df.index.tz)  # Print timezone
print(df.index[:3])  # Show first 3 values after timezone enforcement

# Display precipitation type counts
print(df["Precip Type"].value_counts())  # Count rain/snow occurrences

# Drop columns that are not needed for modeling
df.drop(["Summary", "Daily Summary", "Precip Type"], axis=1, inplace=True)  # Remove unused text and categorical fields
print(df.head())  # Display cleaned data

# Normalize column names for consistency and compatibility
df.columns = [
    col.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    .replace(" ", "_")  # Replace spaces with underscores
    .replace("(", "")  # Remove parentheses
    .replace(")", "")  # Remove parentheses
    .replace("/", "_")  # Replace slashes with underscores
    for col in df.columns
]
print(df.columns)  # Print updated column names

# Check missing values in the dataset
print(f"Missing value count per column:\n{df.isnull().sum()}")  # Print null counts

# Display DataFrame information
print(df.info())  # Print column types and memory usage

# Plot temperature distribution
plt.figure()  # Create a new figure
sns.histplot(df["temperature_c"], bins=60, kde=True)  # Plot histogram with kernel density estimate
plt.title("Temperature Distribution")  # Chart title
plt.xlabel("Temperature (°C)")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label
plt.tight_layout()  # Optimize layout spacing
plt.grid(True)  # Enable grid
plt.show()  # Render chart

# Scatter plot for last 1000 hours of temperature data
last_1000 = df["temperature_c"].iloc[-1000:]  # Select last 1000 entries
plt.figure()  # Create new figure
plt.scatter(last_1000.index, last_1000.values, s=10, color="blue", alpha=0.6)  # Scatter plot
plt.title("Temperature Change Over Last 1000 Hours")  # Chart title
plt.xlabel("Time")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.grid(True)  # Enable grid
plt.tight_layout()  # Optimize layout
plt.show()  # Render chart

# Daily resampling for temperature statistics
daily_temp = df["temperature_c"].resample("D").agg(["mean", "min", "max"]).dropna()  # Compute mean/min/max per day
plt.figure(figsize=(12, 5))  # Create new figure with size
plt.plot(daily_temp.index, daily_temp["mean"], label="Daily Mean", linewidth=2.2)  # Plot daily mean
plt.fill_between(daily_temp.index, daily_temp["min"], daily_temp["max"], alpha=0.35, label="Min–Max Range")  # Shade range
plt.title("Daily Temperature (Mean + Min/Max Range)")  # Chart title
plt.xlabel("Date")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.grid(True)  # Enable grid
plt.legend()  # Show legend
plt.tight_layout()  # Optimize layout
plt.show()  # Render chart

# Weekly resampling for temperature statistics
weekly_temp = df["temperature_c"].resample("W").agg(["mean", "min", "max"]).dropna()  # Compute mean/min/max per week
plt.figure(figsize=(12, 5))  # Create new figure
plt.plot(weekly_temp.index, weekly_temp["mean"], label="Weekly Mean", linewidth=2.2)  # Plot weekly mean
plt.fill_between(weekly_temp.index, weekly_temp["min"], weekly_temp["max"], alpha=0.35, label="Min/Max Range")  # Shade range
plt.title("Weekly Temperature (Mean + Min/Max Range)")  # Chart title
plt.xlabel("Date")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.grid(True)  # Enable grid
plt.legend()  # Show legend
plt.tight_layout()  # Optimize layout
plt.show()  # Render chart

# Rolling mean for daily trend (7 days)
daily_roll7 = daily_temp["mean"].rolling(window=7).mean()  # Compute 7-day rolling mean
plt.figure(figsize=(12, 5))  # Create new figure
plt.plot(daily_temp.index, daily_temp["mean"], alpha=0.4, label="Daily Mean")  # Plot original daily mean
plt.plot(daily_roll7.index, daily_roll7, label="7-Day Rolling Mean")  # Plot rolling trend
plt.title("Daily Temperature Trend (7-Day Rolling Mean)")  # Chart title
plt.xlabel("Date")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.grid(True)  # Enable grid
plt.legend()  # Show legend
plt.tight_layout()  # Optimize layout
plt.show()  # Render chart

# Rolling mean for weekly trend (4 weeks)
weekly_roll4 = weekly_temp["mean"].rolling(window=4).mean()  # Compute 4-week rolling mean
plt.figure(figsize=(12, 5))  # Create new figure
plt.plot(weekly_temp.index, weekly_temp["mean"], alpha=0.4, label="Weekly Mean")  # Plot original weekly mean
plt.plot(weekly_roll4.index, weekly_roll4, label="4-Week Rolling Mean")  # Plot rolling trend
plt.title("Weekly Temperature Trend (4-Week Rolling Mean)")  # Chart title
plt.xlabel("Date")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.grid(True)  # Enable grid
plt.legend()  # Show legend
plt.tight_layout()  # Optimize layout
plt.show()  # Render chart

