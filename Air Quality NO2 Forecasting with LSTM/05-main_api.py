from fastapi import FastAPI, HTTPException  # FastAPI framework + HTTP error responses
from pydantic import BaseModel  # Request body schema validation

import torch  # PyTorch core
import torch.nn as nn  # Neural network layers (LSTM)
import numpy as np  # Numerical ops, array shaping
import joblib  # Load scaler used during preprocessing

app = FastAPI(  # Create FastAPI application instance
    title="NO2 Prediction API",  # API title (Swagger UI)
    description="It predicts NO2 with 72 time steps using the LSTM model."  # API description
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Load scaler (MinMaxScaler)  # Ensures consistent normalization between training and inference
scaler = joblib.load("scaler.pkl")  # Load persisted scaler
input_size = scaler.n_features_in_  # Number of input features (expected: 7)

# LSTM model hyperparameters  # Must match training configuration
hidden_size = 128  # Hidden size
num_layers = 3  # Number of LSTM layers
dropout_rate = 0.2  # Dropout rate
output_size = 1  # Single regression output

# LSTM model definition (PyTorch)  # Must match training architecture for weight loading
class LSTMModel(nn.Module):  # Define model class
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):  # Constructor
        super(LSTMModel, self).__init__()  # Initialize base class
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear output head

    def forward(self, x):  # Forward pass
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # Take last time step hidden state
        out = self.fc(out)  # Map to prediction
        return out  # Return (batch, 1)
    
# Create and load trained model  # Initialize model and load checkpoint weights
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)  # Instantiate model
model.load_state_dict(torch.load("lstm_model_best.pt", map_location=device))  # Load best trained weights
model.eval()  # Evaluation mode

# Request payload schema  # Defines expected JSON structure
class SequenceInput(BaseModel):  # Pydantic model for request body
    sequence: list[list[float]]  # 2D list: 72 time steps x 7 features

# Prediction endpoint  # Accepts POST request with sequence data
@app.post("/predict")  # Route decorator
def predict(data: SequenceInput):  # Endpoint function
    
    seq = data.sequence  # Extract sequence from request body

    # Validate input shape  # Ensure it matches model expectation
    if not seq:  # Reject empty input early
        raise HTTPException(status_code=400, detail="The sequence field cannot be empty.")  # Bad request

    if len(seq) != 72:  # Must be exactly 72 time steps
        raise HTTPException(status_code=400, detail="Data size is invalid. Expected size is 72x7.")  # Bad request

    if any((not isinstance(row, list)) or (len(row) != input_size) for row in seq):  # Validate each row length
        raise HTTPException(status_code=400, detail="Data size is invalid. Expected size is 72x7.")  # Bad request
    
    try:
        # Convert input to NumPy then Torch tensor  # Model expects (batch, seq_len, features)
        X = np.array(seq).reshape(1, 72, input_size)  # Add batch dimension
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Torch tensor on device

        # Predict with model  # Inference-only
        with torch.inference_mode():  # Disable gradients
            prediction = model(X_tensor).cpu().numpy()[0][0]  # Extract scalar prediction (normalized)

        # Inverse scaling (NO2 only)  # Convert normalized target back to original units without dummy arrays
        # MinMaxScaler stores per-feature min/max; NO2(GT) is column 0  # Target column index
        no2_min = float(scaler.data_min_[0])  # NO2 minimum from TRAIN-fitted scaler
        no2_max = float(scaler.data_max_[0])  # NO2 maximum from TRAIN-fitted scaler
        inv_pred = prediction * (no2_max - no2_min) + no2_min  # Inverse MinMax scaling

        # Return JSON response  # API result
        return {"estimated_NO2": round(float(inv_pred), 2)}  # Rounded prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Internal server error with details
