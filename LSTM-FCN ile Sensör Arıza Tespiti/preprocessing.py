import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv("secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("secom_labels.data", sep=r"\s+", header=None)
df["label"] = labels[0].values

# Missing values
missing_per_col = df.isnull().sum()
total_missing = missing_per_col.sum()
missing_ratio = (missing_per_col / len(df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
# print(missing_ratio)
df["missing_ratio"] = missing_ratio
# print(df.head())
df = df[~(df["missing_ratio"] > 0.9)]
# print(df.shape) # (1464, 592)
df.drop("missing_ratio", inplace=True, axis=1)
print(df.shape)

# Fill missing values with column means
df.fillna(df.mean(), inplace=True)

# Convert labels to 0-1
df["label"] = df["label"].apply(lambda x: 0 if x == -1 else 1)  # 0: normal, 1: faulty

# Split into X and y
X = df.drop(columns=["label"])
y = df["label"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Numpy -> PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
