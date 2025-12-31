import pandas as pd  # Import pandas for data loading and preprocessing
import numpy as np  # Import numpy for numerical operations
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for normalization
import joblib  # Import joblib to save/load scaler objects

# Load the CSV file
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")  # Read the dataset into a DataFrame

# Convert date_time from string to datetime
df["date_time"] = pd.to_datetime(df["date_time"])  # Parse date_time as datetime objects

# Set date_time as the DataFrame index
df.set_index("date_time", inplace=True)  # Use date_time as index for time-based features

# Create time-based features
df["hour"] = df.index.hour  # Extract hour of day
df["dayofweek"] = df.index.dayofweek  # Extract day of week (0=Monday, ..., 6=Sunday)
df["month"] = df.index.month  # Extract month (1-12)

# Define features and target
features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "dayofweek", "month"]  # Input features
target = "traffic_volume"  # Target variable
df = df[features + [target]].dropna()  # Keep only required columns and drop rows with missing values
print(df.head())  # Print sample rows to confirm preprocessing

# Normalize features and target to [0, 1]
scaler_X = MinMaxScaler()  # Scaler for input features
scaler_y = MinMaxScaler()  # Scaler for target

X_scaled = scaler_X.fit_transform(df[features])  # Fit and transform input features
y_scaled = scaler_y.fit_transform(df[[target]])  # Fit and transform target

# Save scaler objects for later inference
joblib.dump(scaler_X, "scaler_X.save")  # Save input scaler
joblib.dump(scaler_y, "scaler_y.save")  # Save target scaler

# Create sequences (sliding window) for time series modeling
def create_sequences(X, y, seq_length):  # Define a function to create sequences
    X_seq, y_seq = [], []  # Initialize containers for sequences and labels
    for i in range(len(X) - seq_length):  # Slide a window across the dataset
        X_seq.append(X[i : i + seq_length])  # Append window of length seq_length
        y_seq.append(y[i + seq_length])  # Append the next step as the label
    return np.array(X_seq), np.array(y_seq)  # Convert lists to numpy arrays

SEQ_LEN = 24  # Sequence length (24 hours)
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)  # Create input-output sequences

# Train / Validation / Test split (chronological split for time series)
train_ratio = 0.70  # Train set ratio
val_ratio = 0.15  # Validation set ratio
# test_ratio = 0.15  # Remaining part becomes the test set

n_total = len(X_seq)  # Total number of sequences
train_end = int(train_ratio * n_total)  # End index for train set
val_end = int((train_ratio + val_ratio) * n_total)  # End index for validation set

# Train split
X_train = X_seq[:train_end]  # Train inputs
y_train = y_seq[:train_end]  # Train labels

# Validation split
X_val = X_seq[train_end:val_end]  # Validation inputs
y_val = y_seq[train_end:val_end]  # Validation labels

# Test split
X_test = X_seq[val_end:]  # Test inputs
y_test = y_seq[val_end:]  # Test labels

# Save splits to disk
np.save("X_train.npy", X_train)  # Save train inputs
np.save("y_train.npy", y_train)  # Save train labels

np.save("X_val.npy", X_val)  # Save validation inputs
np.save("y_val.npy", y_val)  # Save validation labels

np.save("X_test.npy", X_test)  # Save test inputs
np.save("y_test.npy", y_test)  # Save test labels

print("Save completed.")  # Print completion message
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")  # Print split shapes
