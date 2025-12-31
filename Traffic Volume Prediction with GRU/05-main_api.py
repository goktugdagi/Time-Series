# python -m uvicorn 05-main_api:app --reload  # Command to run the FastAPI app with auto-reload

from fastapi import FastAPI  # Import FastAPI framework
from pydantic import BaseModel  # Import BaseModel for request validation
import numpy as np  # Import numpy for array handling
import torch  # Import PyTorch
import torch.nn as nn  # Import neural network components
import joblib  # Import joblib for loading scalers
import random  # Import random for sample generation

# Initialize the FastAPI application
app = FastAPI(title="GRU Traffic Forecast API")  # Create FastAPI app instance 

class InputData(BaseModel):  # Define request schema for input data
    # Each time step is a list: [temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]
    sequence: list  # Nested list of shape 24 x 7

# Define GRU model architecture (must match the trained model)
class GRUNet(nn.Module):  # GRU-based model definition
    def __init__(self, input_size, hidden_size, num_layers, output_size):  # Initialize model
        super(GRUNet, self).__init__()  # Call parent constructor
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU layer
        self.fc = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):  # Forward pass
        out, _ = self.gru(x)  # GRU outputs for all time steps
        last_out = out[:, -1, :]  # Take the last time step output
        out = self.fc(last_out)  # Map to prediction
        return out  # Return output

# Load model and scaler at startup
INPUT_SIZE = 7  # Input feature size
HIDDEN_SIZE = 64  # GRU hidden size
NUM_LAYERS = 2  # Number of GRU layers
OUTPUT_SIZE = 1  # Output size

model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)  # Instantiate model
model.load_state_dict(torch.load("gru_model.pth"))  # Load trained model weights
model.eval()  # Set to evaluation mode for inference

scaler_y = joblib.load("scaler_y.save")  # Load target scaler (traffic volume)

# API Endpoint: /predict (POST)
@app.post("/predict")  # Define POST route
def predict(data: InputData):  # Endpoint function
    """
        Predicts traffic volume based on a 24-hour input sequence.
    """
    # Convert input sequence to a float32 numpy array
    input_seq = np.array(data.sequence).astype(np.float32)  # Ensure correct dtype

    # Validate input shape
    if input_seq.shape != (24, 7):  # Check expected shape
        return {"error": "Invalid input shape. Expected (24, 7)."}  # Return error message

    # Convert to torch tensor and add batch dimension
    input_tensor = torch.tensor(input_seq).unsqueeze(0)  # Shape: (1, 24, 7)

    # Generate prediction
    with torch.no_grad():  # Disable gradients for inference
        prediction = model(input_tensor)  # Predict in scaled space
        prediction = prediction.numpy()  # Convert to numpy

    # Inverse transform to original scale
    prediction_orig = scaler_y.inverse_transform(prediction)  # Convert back to original units

    predicted_value = float(prediction_orig[0][0])  # Extract scalar prediction

    return {"predicted_traffic_volume": predicted_value}  # Return English response key

# Sample input generator (for quick local testing)
sample_sequence = []  # Initialize sample sequence container
for i in range(24):  # Loop over 24 hours
    temp = random.uniform(280, 300)  # Temperature in Kelvin (random)
    rain = 0  # Rain amount
    snow = 0  # Snow amount
    clouds = random.randint(0, 100)  # Cloud coverage percentage
    hour = i  # Hour feature
    dayofweek = random.randint(0, 6)  # Day of week (0-6)
    month = 10  # Month (example fixed)
    sample_sequence.append([temp, rain, snow, clouds, hour, dayofweek, month])  # Append feature vector

print(sample_sequence)  # Print generated sample sequence
