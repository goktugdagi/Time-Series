"""
Human Activity Recognition with Inception

Problem Definition:

- Sensor data (accelerometer, gyroscope, etc.) is generated in wearable devices (smartphones, bracelets, etc.).
- This data is time series data.
- Our goal: To classify different activities using the Inception architecture. - Activities: Walking, standing, lying down, climbing stairs

Dataset:

- https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Segmented sensor data from 30 different individuals
- 9 channels sampled with 128 time-step windows ((x,y,z) x (body_acc, body_gyro, total_acc))

Tools and Technologies: - Inception: TensorFlow/Keras

Plan/Program:
- 01-load_and_segment.py: data loading
- 02-preprocessing.py: one-hot encoding and normalization
- inception_model.py: defining CNN-based Inception architecture
- 04-train.py: model training, validation
- 05-test.py: tests with accuracy, F1-score and confusion matrix
- 06-main_api.py: FastAPI with /predict endpoint
- 07-test_api.py: service tests
- 08-app_streamlit.py: UI (user interface)

install libraries: freeze

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
"""

import numpy as np
import os 

# Read segmented inertial signals.
def load_inertial_signals(folder_path, subset = "train"):
    """
        folder_path: "UCI HAR Dataset/train/Inertial Signals
        subset: "train" veya "test"
    """
    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z"
    ]

    signals_data = []
    for signal in signal_names:
        filename = f"{signal}_{subset}.txt"
        file_path = os.path.join(folder_path, filename)
        data = np.loadtxt(file_path)
        signals_data.append(data)
    
    # (9, n_samples, 128) -> (n_samples, 128, 9)
    stacked = np.stack(signals_data, axis=0).transpose(1, 2, 0)
    return stacked

# Egitim seti
X_train = load_inertial_signals("UCI HAR Dataset/train/Inertial Signals", subset="train")
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int) - 1
print(X_train.shape)
print(y_train.shape)

# Test seti
X_test = load_inertial_signals("UCI HAR Dataset/test/Inertial Signals", subset="test")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int) - 1
print(X_test.shape)
print(y_test.shape)

# kaydet
np.save("X_train_raw.npy", X_train)
np.save("X_test_raw.npy", X_test)
np.save("y_train_raw.npy", y_train)
np.save("y_test_raw.npy", y_test)
print("The data has been recorded.")