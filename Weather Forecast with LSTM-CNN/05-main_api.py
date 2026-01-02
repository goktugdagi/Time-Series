# # python -m uvicorn 05-main_api:app --reload

# from fastapi import FastAPI
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import random
# import os
# import tensorflow as tf

# app = FastAPI(title="Weather Forecast API")

# # IMPORTANT: These must match training preprocessing exactly
# FEATURE_COLUMNS = [
#     "humidity",
#     "wind_speed_km_h",
#     "pressure_millibars",
#     "visibility_km",
#     "apparent_temperature_c",
# ]

# INPUT_SHAPE = (24, 5)

# class InputData(BaseModel):
#     # Nested list with shape 24x5 in FEATURE_COLUMNS order
#     sequence: list

# # Load model & scaler
# model = tf.keras.models.load_model("cnn_lstm_weather_model.h5")
# scaler_x = joblib.load("scaler.pkl")

# @app.post("/predict")
# def predict(data: InputData):
#     """
#     Predicts temperature_c from a 24-hour input sequence (24x5).
#     Input must follow FEATURE_COLUMNS order.
#     """
#     input_seq = np.array(data.sequence, dtype=np.float32)

#     if input_seq.shape != INPUT_SHAPE:
#         return {
#             "error": f"Invalid input shape. Expected {INPUT_SHAPE}, got {tuple(input_seq.shape)}.",
#             "expected_feature_order": FEATURE_COLUMNS,
#         }

#     # Scale using training scaler
#     input_scaled = scaler_x.transform(input_seq)

#     # Add batch dimension: (1, 24, 5)
#     input_batch = np.expand_dims(input_scaled, axis=0)

#     pred = model.predict(input_batch, verbose=0)
#     predicted_value = float(pred[0][0])

#     return {"predicted_temperature_c": predicted_value}

# # Optional sample generator (disabled by default to avoid uvicorn reload side-effects)
# # Enable by setting environment variable: PRINT_SAMPLE=1
# if os.getenv("PRINT_SAMPLE", "0") == "1":
#     sample_sequence = []
#     for i in range(24):
#         humidity = random.uniform(0, 100)
#         wind_speed_km_h = random.uniform(0, 60)
#         pressure_millibars = random.uniform(980, 1050)
#         visibility_km = random.uniform(0, 20)
#         apparent_temperature_c = random.uniform(-5, 40)

#         sample_sequence.append([
#             humidity,
#             wind_speed_km_h,
#             pressure_millibars,
#             visibility_km,
#             apparent_temperature_c
#         ])

#     print("FEATURE_COLUMNS:", FEATURE_COLUMNS)
#     print("Sample sequence (24x5):")
#     print(sample_sequence)

# Import FastAPI and validation utilities
from fastapi import FastAPI  # API framework
from pydantic import BaseModel  # Request body validation
import numpy as np  # Numerical operations
import joblib  # Load scaler
import random  # Sample generator
import os  # Environment variable control
import tensorflow as tf  # Load trained model

# Initialize FastAPI app with title
app = FastAPI(title="Weather Forecast API")  # Create API instance

# Feature columns must match training preprocessing order
FEATURE_COLUMNS = [
    "humidity",
    "wind_speed_km_h",
    "pressure_millibars",
    "visibility_km",
    "apparent_temperature_c",
]

# Expected input shape must match 24 hours x 5 features
INPUT_SHAPE = (24, 5)  # Timesteps x features

# Define request body model
class InputData(BaseModel):
    # Nested list with shape 24x5 following FEATURE_COLUMNS order
    sequence: list  # 24 x 5 input sequence

# Load trained model and scaler
model = tf.keras.models.load_model("cnn_lstm_weather_model.h5")  # Load CNN-LSTM model
scaler = joblib.load("scaler.pkl")  # Load pre-fitted scaler (fit on train only)

# Define prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    """
    Predicts temperature_c from a 24-hour input sequence (24x5).
    Input must follow FEATURE_COLUMNS order.
    """
    input_seq = np.array(data.sequence, dtype=np.float32)  # Convert input to NumPy array

    # Validate input shape
    if input_seq.shape != INPUT_SHAPE:  # If shape mismatch
        return {
            "error": f"Invalid input shape. Expected {INPUT_SHAPE}, got {tuple(input_seq.shape)}.",
            "expected_feature_order": FEATURE_COLUMNS,
        }

    # Scale input using scaler fitted on training data
    input_scaled = scaler.transform(input_seq)  # Transform test input

    # Add batch dimension: (1, 24, 5)
    input_batch = np.expand_dims(input_scaled, axis=0)  # Add batch axis

    # Run model prediction
    pred = model.predict(input_batch, verbose=0)  # Predict silently
    predicted_value = float(pred[0][0])  # Convert prediction to float

    # Return result
    return {"predicted_temperature_c": predicted_value}  # JSON response
