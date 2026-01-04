import pandas as pd  # Data loading/manipulation
import numpy as np  # Numerical operations
from sklearn.preprocessing import MinMaxScaler  # Feature scaling to [0, 1]
from sklearn.model_selection import train_test_split  # (Imported but not used; kept as-is)
import joblib  # Persist scaler to disk for inference (API/UI)

# Load CSV  # Read raw dataset
df = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',', encoding='latin1')  # Dataset delimiter/decimal config
df.dropna(axis=1, how='all', inplace=True)  # Drop fully empty columns (common in this dataset)

# Date-time processing and missing value cleanup  # Build time index and prepare interpolation
df["datetime"] = pd.to_datetime(  # Create datetime column
    df["Date"] + " " + df["Time"],  # Combine date and time text
    format="%d/%m/%Y %H.%M.%S",  # Expected input format
    errors="coerce",  # Invalid rows become NaT
)
df.dropna(subset=["datetime"], inplace=True)  # Drop rows with invalid timestamps
df.drop(["Date", "Time"], inplace=True, axis=1)  # Remove original text columns
df.set_index("datetime", inplace=True)  # Set datetime as index (needed for time interpolation)
df.replace(-200, np.nan, inplace=True)  # Replace sensor error sentinel with NaN
df.interpolate(method="time", inplace=True)  # Fill missing values using time-based interpolation

# Add time-based columns  # Simple seasonality-related features
df["hour"] = df.index.hour  # Hour of day
df["month"] = df.index.month  # Month of year

# Select columns (target + inputs)  # Build feature set for LSTM
selected_columns = ['NO2(GT)', 'T', 'RH', 'AH', 'CO(GT)', 'hour', 'month']  # Target first, then predictors
df = df[selected_columns]  # Keep only the selected features
df.dropna(inplace=True)  # Drop any remaining NaNs (safety net)

# Sequence creation function  # Convert tabular time series into sliding windows for LSTM
def create_sequences(data, window_size):  # data: (N, features), window_size: int
    X, y = [], []  # Containers for inputs (windows) and targets
    for i in range(len(data) - window_size):  # Slide over the data
        X_seq = data[i : i + window_size]  # X: past window [i, i+window_size)
        y_seq = data[i + window_size][0]  # y: future NO2(GT) at next step (column 0)
        X.append(X_seq)  # Append window
        y.append(y_seq)  # Append target
    return np.array(X), np.array(y)  # Return as NumPy arrays

# Window size: 72 (3 days x 24 hours)  # Model uses last 72 hours to predict next NO2
WINDOW_SIZE = 72  # Sequence length for LSTM

# Train/Val/Test split ratios  # Time-series split (no shuffling)
train_ratio = 0.70  # First 70% for training
val_ratio = 0.10  # Next 10% for validation
test_ratio = 0.20  # Last 20% for testing

assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "The sum of the split ratios must be 1.0."  # Sanity check

raw_values = df.values  # Convert to raw NumPy array (rows x features)
n_rows = len(raw_values)  # Total rows (time steps)
n_windows = n_rows - WINDOW_SIZE  # Total number of supervised windows we can create

train_end = int(n_windows * train_ratio)  # Index where training windows end (exclusive)
val_end = int(n_windows * (train_ratio + val_ratio))  # Index where validation windows end (exclusive)

scaler_fit_end_row = train_end + WINDOW_SIZE  # Last row index included by the final training window (exclusive)  # Fit boundary

# Fit MinMaxScaler on TRAIN-ONLY slice  # Prevent leakage from val/test ranges
scaler = MinMaxScaler()  # Initialize scaler
scaler.fit(raw_values[:scaler_fit_end_row])  # Fit only on train-covered rows

# Transform the entire dataset with the train-fitted scaler  # Apply consistent scaling to all splits
scaled_values = scaler.transform(raw_values)  # Scaled features in [0, 1] based on train statistics

# Save scaler object (used by API and Streamlit)  # Ensures consistent scaling at inference
joblib.dump(scaler, "scaler.pkl")  # Persist scaler to disk

# Create windowed datasets AFTER scaling  # Build supervised learning arrays
X, y = create_sequences(scaled_values, WINDOW_SIZE)  # X: (windows, 72, 7), y: (windows,)

# Split windows by time order  # No shuffling for temporal integrity
X_train, y_train = X[:train_end], y[:train_end]  # Training split (earliest chunk)
X_val, y_val = X[train_end : val_end], y[train_end : val_end]  # Validation split (middle chunk)
X_test, y_test = X[val_end:], y[val_end:]  # Test split (latest chunk)

# Shape checks  # Confirm LSTM-compatible tensor shapes
print("X_train shape:", X_train.shape)  # (samples, window, features)
print("y_train shape:", y_train.shape)  # (samples,)
print("X_val shape:", X_val.shape)  # (samples, window, features)
print("y_val shape:", y_val.shape)  # (samples,)
print("X_test shape:", X_test.shape)  # (samples, window, features)
print("y_test shape:", y_test.shape)  # (samples,)

# Save arrays to disk  # Used by training, testing, and inference demos
np.save("X_train.npy", X_train)  # Training inputs
np.save("y_train.npy", y_train)  # Training targets

np.save("X_val.npy", X_val)  # Validation inputs
np.save("y_val.npy", y_val)  # Validation targets

np.save("X_test.npy", X_test)  # Test inputs
np.save("y_test.npy", y_test)  # Test targets
