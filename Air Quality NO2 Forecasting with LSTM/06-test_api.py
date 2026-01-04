import requests  # HTTP client for calling the FastAPI service
import numpy as np  # Load saved test arrays

# Load test dataset  # Produced by 02-preprocessing.py
X_test = np.load("X_test.npy")  # Test inputs (N, 72, 7)

# Pick a sample from the test set for API testing  # Deterministic sample selection
sample_index = 2  # Which test window to use
sample = X_test[sample_index]  # Single sample window (72, 7)

print(sample.tolist())  # Print payload to inspect (optional)

# Build JSON payload  # FastAPI expects list[list[float]]
payload = {  # Request body
    "sequence": sample.tolist()  # Convert NumPy array to nested Python lists
}

# API address  # Localhost FastAPI server
url = "http://127.0.0.1:8000/predict"  # /predict endpoint

# Send POST request  # Call the model inference API
try:
    response = requests.post(url, json=payload)  # POST JSON payload

    if response.status_code == 200:  # Success
        result = response.json()  # Parse JSON response
        print(f"Estimated NO2 value: {result}")  # Print result
    else:
        print(f"Error code: {response.status_code}")  # Print error code
        print(f"Error message: {response.text}")  # Print error details

except requests.exceptions.ConnectionError:
    print("The API server is not running. Please start FastAPI.")  # Server not running
