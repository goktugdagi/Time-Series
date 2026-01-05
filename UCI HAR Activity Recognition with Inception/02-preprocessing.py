import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # one hot encoding
import joblib

# Load segmented data
X_train = np.load("X_train_raw.npy") # (n_sample, 128, 9)
X_test = np.load("X_test_raw.npy")
y_train = np.load("y_train_raw.npy")
y_test = np.load("y_test_raw.npy")


VAL_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train)

# One hot encoding of labels
num_classes = len(np.unique(y_train)) # 6
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Normalization
scalers = {}
X_train_scaled = np.zeros_like(X_train)
X_val_scaled = np.zeros_like(X_val)
X_test_scaled = np.zeros_like(X_test)

# Channel Normalization
for i in range(X_train.shape[2]):
    scaler = StandardScaler()
    X_train_scaled[:, :, i] = scaler.fit_transform(X_train[:, :, i])
    X_val_scaled[:, :, i] = scaler.transform(X_val[:, :, i])
    X_test_scaled[:, :, i] = scaler.transform(X_test[:, :, i])
    scalers[i] = scaler

# Save
np.save("X_train.npy", X_train_scaled)
np.save("X_val.npy", X_val_scaled)
np.save("X_test.npy", X_test_scaled)

np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)

# Save Scaler
joblib.dump(scalers, "scalers.pkl")
print("Train/Val/Test data and scalers.pkl file were saved.")
print(f"Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Shapes -> y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")