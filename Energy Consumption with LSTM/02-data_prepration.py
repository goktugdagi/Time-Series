import numpy as np  # Numerical operations
from sklearn.preprocessing import MinMaxScaler  # Scaling utility for normalization
import pandas as pd  # Data processing
import joblib  # Save and load scaler and models

# Load the hourly resampled dataset
df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col=0,  # First column is datetime index
    parse_dates=True  # Convert index to datetime format
)

# Drop missing values to ensure clean input
df_hourly.dropna(inplace=True)

# Convert DataFrame into NumPy array for model input
values = df_hourly.values.reshape(-1, 1)

# Normalize data to 0-1 range using MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Save scaler to reuse in validation and forecasting (must be identical)
joblib.dump(scaler, "scaler.save")

# Create sliding window samples for LSTM input
def create_sliding(data, window_size=24):
    # data: normalized time series
    # window_size: number of past time steps used as model input
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])  # Input window
        y.append(data[i + window_size])  # Next time step as target
    return np.array(X), np.array(y)

# Generate sliding window features and labels
window_size = 24
X, y = create_sliding(scaled, window_size)

# Perform chronological train / validation / test split (no shuffling for time series)
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Print dataset shapes to confirm split correctness
print(
    f"X_train shape: {X_train.shape}\n"
    f"y_train shape: {y_train.shape}\n"
    f"X_val shape:   {X_val.shape}\n"
    f"y_val shape:   {y_val.shape}\n"
    f"X_test shape:  {X_test.shape}\n"
    f"y_test shape:  {y_test.shape}"
)

# Save split datasets for training and evaluation
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
