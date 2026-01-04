import torch  # PyTorch core
import torch.nn as nn  # Neural network layers (LSTM model definition)
import numpy as np  # Array loading and transformations
import joblib  # Load scaler object
import matplotlib.pyplot as plt  # Plot predictions vs ground truth
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Regression metrics

# Select CUDA/CPU device  # Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device selection
print(f"Device: {device}")  # Print device

# Load test data  # Produced by 02-preprocessing.py
X_test = np.load("X_test.npy")  # Test windows (N, 72, 7)
y_test = np.load("y_test.npy")  # Test targets (N,)

# Convert NumPy -> Torch tensors and move to device  # Required for model inference
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Inputs
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)  # Targets as (N, 1)

# Load scaler for inverse transform  # Needed to interpret NO2 in original units
scaler = joblib.load("scaler.pkl")  # MinMaxScaler fitted during preprocessing

# LSTM model class  # Must match training architecture to load weights correctly
class LSTMModel(nn.Module):  # Define same model as in training
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):  # Constructor
        super(LSTMModel, self).__init__()  # Initialize base class
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear output layer (last hidden -> prediction)

    def forward(self, x):  # Forward pass
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)  # Map to output
        return out  # (batch, 1)

# Model hyperparameters  # Must match training settings
input_size = X_test.shape[2]  # Number of features (7)
hidden_size = 128  # Hidden size
num_layers = 3  # Number of LSTM layers
dropout_rate = 0.2  # Dropout between layers
output_size = 1  # Single output

# Create model and move to device  # Instantiate same architecture
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)  # Model instance

# Load trained weights with device compatibility  # map_location handles CPU/GPU differences
model.load_state_dict(torch.load("lstm_model_best.pt", map_location=device))  # Load best checkpoint
model.eval()  # Set evaluation mode (disables dropout)

# Run predictions  # Inference on test set
with torch.no_grad():  # Disable gradients
    predictions = model(X_test)  # Forward pass

# Move tensors to CPU and convert to NumPy  # NumPy cannot handle CUDA tensors directly
y_pred = predictions.detach().cpu().numpy()  # Predicted normalized NO2 (N, 1)
y_true = y_test.detach().cpu().numpy()  # True normalized NO2 (N, 1)

# Inverse scaling (NO2 only)  # Convert normalized NO2 back to original units without dummy arrays
no2_min = float(scaler.data_min_[0])  # NO2 minimum from TRAIN-fitted scaler
no2_max = float(scaler.data_max_[0])  # NO2 maximum from TRAIN-fitted scaler

inv_y_pred = y_pred[:, 0] * (no2_max - no2_min) + no2_min  # Inverse MinMax scaling for predictions
inv_y_true = y_true[:, 0] * (no2_max - no2_min) + no2_min  # Inverse MinMax scaling for ground truth

# Plot ground truth vs predictions  # Visual comparison
plt.figure()  # New figure
plt.plot(inv_y_true, label="Real NO2", color="blue")  # True series
plt.plot(inv_y_pred, label="Prediction NO2", color="red", alpha=0.5)  # Predicted series
plt.title("Fact vs Guess NO2")  # Title
plt.xlabel("Time Step")  # X label
plt.ylabel("NO2")  # Y label
plt.legend()  # Legend
plt.show()  # Display

# Evaluate performance metrics  # Compute MAE and MSE in original scale
mae = mean_absolute_error(inv_y_pred, inv_y_true)  # Mean Absolute Error
mse = mean_squared_error(inv_y_pred, inv_y_true)  # Mean Squared Error

print(f"MAE: {mae}")  # Print MAE
print(f"MSE: {mse}")  # Print MSE
