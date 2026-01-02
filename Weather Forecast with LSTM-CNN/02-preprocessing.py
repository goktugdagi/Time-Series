# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import zscore
# import joblib

# # Read data
# df = pd.read_csv("weatherHistory.csv")

# # Parse datetime and set as index
# df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])
# df.set_index("Formatted Date", inplace=True)

# # Force UTC datetime index; drop invalid; sort chronologically
# df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
# df = df[~df.index.isna()]
# df = df.sort_index()

# # Normalize column names (IMPORTANT: "/" -> "_" so wind_speed_km/h becomes wind_speed_km_h)
# df.columns = [
#     col.strip().lower()
#     .replace(" ", "_")
#     .replace("(", "")
#     .replace(")", "")
#     .replace("/", "_")
#     for col in df.columns
# ]

# # Select columns
# feature_cols = [
#     "humidity",
#     "wind_speed_km_h",
#     "pressure_millibars",
#     "visibility_km",
#     "apparent_temperature_c",
# ]
# target_col = "temperature_c"

# # Keep only required columns and drop missing
# df_sub = df[feature_cols + [target_col]].copy()
# df_sub.dropna(inplace=True)

# # Outlier filtering using z-score ONLY on features (avoid target-based filtering)
# z = np.abs(zscore(df_sub[feature_cols]))
# df_clean = df_sub[(z < 3).all(axis=1)].copy()

# # --- Chronological split on ROW LEVEL (before scaling & before sequences) ---
# train_ratio = 0.70
# val_ratio = 0.15

# n_rows = len(df_clean)
# train_end_row = int(train_ratio * n_rows)
# val_end_row = int((train_ratio + val_ratio) * n_rows)

# df_train = df_clean.iloc[:train_end_row].copy()
# df_val = df_clean.iloc[train_end_row:val_end_row].copy()
# df_test = df_clean.iloc[val_end_row:].copy()

# X_train_raw = df_train[feature_cols].values
# y_train_raw = df_train[target_col].values

# X_val_raw = df_val[feature_cols].values
# y_val_raw = df_val[target_col].values

# X_test_raw = df_test[feature_cols].values
# y_test_raw = df_test[target_col].values

# # --- Scale features: FIT ONLY on TRAIN (leakage-free) ---
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_raw)
# X_val_scaled = scaler.transform(X_val_raw)
# X_test_scaled = scaler.transform(X_test_raw)

# joblib.dump(scaler, "scaler.pkl")

# # --- Create sequences INSIDE each split (no cross-boundary windows) ---
# sequence_length = 24

# def create_sequences_from_arrays(X_arr: np.ndarray, y_arr: np.ndarray, seq_length: int = 24):
#     X_seq, y_seq = [], []
#     for i in range(len(X_arr) - seq_length):
#         X_seq.append(X_arr[i : i + seq_length])
#         y_seq.append(y_arr[i + seq_length])
#     return np.array(X_seq), np.array(y_seq)

# X_train, y_train = create_sequences_from_arrays(X_train_scaled, y_train_raw, sequence_length)
# X_val, y_val = create_sequences_from_arrays(X_val_scaled, y_val_raw, sequence_length)
# X_test, y_test = create_sequences_from_arrays(X_test_scaled, y_test_raw, sequence_length)

# print("Leakage-free splits with per-split sequences:")
# print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
# print(f"X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
# print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

# # Save splits
# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)

# np.save("X_val.npy", X_val)
# np.save("y_val.npy", y_val)

# np.save("X_test.npy", X_test)
# np.save("y_test.npy", y_test)


# Import required libraries for preprocessing and saving artifacts
import pandas as pd  # Data handling
import numpy as np  # Numerical processing
from sklearn.preprocessing import StandardScaler  # Feature scaling
from scipy.stats import zscore  # Z-score outlier detection
import joblib  # Saving scaler
import os  # Environment variable control

# Read the dataset
df = pd.read_csv("weatherHistory.csv")  # Load weather dataset

# Parse datetime column and set it as index
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])  # Convert to datetime
df.set_index("Formatted Date", inplace=True)  # Set as index

# Enforce UTC timezone on index, drop invalid entries, and sort
df.index = pd.to_datetime(df.index, errors="coerce", utc=True)  # Force UTC and coerce errors
df = df[~df.index.isna()]  # Drop invalid dates
df = df.sort_index()  # Ensure chronological order

# Normalize column names to match expected format
df.columns = [
    col.strip().lower()  # Trim and lowercase
    .replace(" ", "_")  # Replace spaces
    .replace("(", "")  # Remove parentheses
    .replace(")", "")  # Remove parentheses
    .replace("/", "_")  # Replace slashes
    for col in df.columns
]

# Define feature columns and target column
feature_cols = [
    "humidity",
    "wind_speed_km_h",
    "pressure_millibars",
    "visibility_km",
    "apparent_temperature_c",
]
target_col = "temperature_c"

# Keep only relevant columns and drop missing rows
df_sub = df[feature_cols + [target_col]].copy()  # Subset selected columns
df_sub.dropna(inplace=True)  # Drop missing values

# Remove outliers using Z-score only on features (avoid filtering on target)
z = np.abs(zscore(df_sub[feature_cols]))  # Compute absolute Z-scores
df_clean = df_sub[(z < 3).all(axis=1)].copy()  # Keep rows where all feature Z-scores < 3

# Split data on row level (train, validation, test)
train_ratio = 0.70  # 70% for training
val_ratio = 0.15  # 15% for validation
test_ratio = 0.15  # 15% for testing

n_rows = len(df_clean)  # Total number of rows
train_end_row = int(train_ratio * n_rows)  # Index for end of training set
val_end_row = int((train_ratio + val_ratio) * n_rows)  # Index for end of validation set

df_train = df_clean.iloc[:train_end_row].copy()  # Training subset
df_val = df_clean.iloc[train_end_row:val_end_row].copy()  # Validation subset
df_test = df_clean.iloc[val_end_row:].copy()  # Test subset

# Extract raw arrays for each subset
X_train_raw = df_train[feature_cols].values  # Raw training features
y_train_raw = df_train[target_col].values  # Raw training target

X_val_raw = df_val[feature_cols].values  # Raw validation features
y_val_raw = df_val[target_col].values  # Raw validation target

X_test_raw = df_test[feature_cols].values  # Raw test features
y_test_raw = df_test[target_col].values  # Raw test target

# Fit scaler on train only (leakage-free), then transform all
scaler = StandardScaler()  # Initialize scaler
X_train_scaled = scaler.fit_transform(X_train_raw)  # Fit + transform train
X_val_scaled = scaler.transform(X_val_raw)  # Transform validation
X_test_scaled = scaler.transform(X_test_raw)  # Transform test

# Save scaler artifact
joblib.dump(scaler, "scaler.pkl")  # Persist scaler for future use

# Create sequences per split (no boundary leakage)
sequence_length = 24  # Number of time steps per sequence

def create_sequences_from_arrays(X_arr: np.ndarray, y_arr: np.ndarray, seq_length: int = 24):
    # Create sliding window sequences within each split
    X_seq, y_seq = [], []  # Initialize sequence lists
    for i in range(len(X_arr) - seq_length):  # Iterate until last valid window
        X_seq.append(X_arr[i : i + seq_length])  # Append feature window
        y_seq.append(y_arr[i + seq_length])  # Append next-step target
    return np.array(X_seq), np.array(y_seq)  # Return as NumPy arrays

# Generate sequences for each split
X_train, y_train = create_sequences_from_arrays(X_train_scaled, y_train_raw, sequence_length)
X_val, y_val = create_sequences_from_arrays(X_val_scaled, y_val_raw, sequence_length)
X_test, y_test = create_sequences_from_arrays(X_test_scaled, y_test_raw, sequence_length)

# Print shapes for verification
print("Leakage-free splits with per-split sequences:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

# Save sequence splits to disk
np.save("X_train.npy", X_train)  # Persist training sequences
np.save("y_train.npy", y_train)  # Persist training targets

np.save("X_val.npy", X_val)  # Persist validation sequences
np.save("y_val.npy", y_val)  # Persist validation targets

np.save("X_test.npy", X_test)  # Persist test sequences
np.save("y_test.npy", y_test)  # Persist test targets
np.save("y_test.npy", y_test)  # Persist test targets
