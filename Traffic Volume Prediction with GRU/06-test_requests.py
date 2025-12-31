import requests  # Import requests to send HTTP calls
import random  # Import random to generate sample input

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict"  # Local prediction endpoint

# Generate a sample 24-hour input sequence
# Each time step has 7 features: [temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]
sample_sequence = []  # Initialize container for the input sequence
for i in range(24):  # Iterate over 24 hours
    temp = random.uniform(280, 300)  # Temperature in Kelvin
    rain = 0  # Rain amount
    snow = 0  # Snow amount
    clouds = random.randint(0, 100)  # Cloud coverage
    hour = i  # Hour feature (0..23)
    dayofweek = random.randint(0, 6)  # Day of week (0..6)
    month = 10  # Month (example fixed)
    sample_sequence.append([temp, rain, snow, clouds, hour, dayofweek, month])  # Append the feature vector

print(sample_sequence)  # Print the generated input for visibility

# Send POST request with JSON payload
response = requests.post(API_URL, json={"sequence": sample_sequence})  # Call the API with the required body format

if response.status_code == 200:  # If request succeeded
    result = response.json()  # Parse JSON response
    print(f"Prediction succeeded.\nPredicted traffic volume: {result['predicted_traffic_volume']}")  # Print prediction
else:
    print("An error occurred.")  # Print error header
    print(f"Status code: {response.status_code}")  # Print HTTP status code
    print(f"Response: {response.text}")  # Print response body
