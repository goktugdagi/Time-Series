
import requests
import numpy as np
import joblib
import random

API_URL = "http://127.0.0.1:8000/predict"

scaler = joblib.load("scaler.pkl")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

idx = random.randint(0, len(X_test) - 1)

sample_scaled = X_test[idx]
sample_raw = scaler.inverse_transform(sample_scaled)
sample_sequence = sample_raw.tolist()

response = requests.post(API_URL, json={"sequence": sample_sequence})

if response.status_code == 200:
    pred = response.json()["predicted_temperature_c"]
    actual = float(y_test[idx])
    print(f"idx={idx}")
    print(f"Predicted temperature_c: {pred}")
    print(f"Actual temperature_c:    {actual}")
    print(f"Absolute error:          {abs(pred - actual)}")
else:
    print("Error:", response.status_code, response.text)
