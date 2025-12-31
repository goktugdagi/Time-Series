import numpy as np  # Import numpy for array operations
import torch  # Import PyTorch
import torch.nn as nn  # Import neural network components
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import joblib  # Import joblib for loading scalers
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # Import evaluation metrics

# Define the GRU model class (must match the training architecture)
class GRUNet(nn.Module):  # GRU-based model definition
    def __init__(self, input_size, hidden_size, num_layers, output_size):  # Initialize model parameters
        super(GRUNet, self).__init__()  # Call parent constructor
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU layer
        self.fc = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):  # Forward pass
        out, _ = self.gru(x)  # GRU outputs for all time steps
        last_out = out[:, -1, :]  # Take output from the last time step
        out = self.fc(last_out)  # Map to final prediction
        return out  # Return prediction

# Hyperparameters (architecture must match training)
SEQ_LEN = 24  # Sequence length (informational)
INPUT_SIZE = 7  # Number of input features
HIDDEN_SIZE = 64  # Hidden size
NUM_LAYERS = 2  # Number of layers
OUTPUT_SIZE = 1  # Output size

# Load test dataset
X_test = np.load("X_test.npy")  # Test inputs (N, 24, 7)
y_test = np.load("y_test.npy")  # Test labels (N, 1)

# Convert test inputs to torch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # Convert test inputs to float32 tensor

# Create model instance and load trained weights
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)  # Instantiate model
model.load_state_dict(torch.load("gru_model.pth"))  # Load trained model weights
model.eval()  # Set model to evaluation mode

# Generate predictions
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X_test_tensor)  # Predict on test set

predictions = predictions.numpy()  # Convert predictions to numpy array

# Load the target scaler to inverse-transform predictions
scaler_y = joblib.load("scaler_y.save")  # Load target scaler

# Inverse-transform predictions and true labels back to original scale
predictions_orig = scaler_y.inverse_transform(predictions)  # Convert predictions to original scale
y_test_orig = scaler_y.inverse_transform(y_test)  # Convert test labels to original scale

# Compute evaluation metrics
rmse = root_mean_squared_error(y_test_orig, predictions_orig)  # Compute RMSE
mae = mean_absolute_error(y_test_orig, predictions_orig)  # Compute MAE

print(f"rmse: {rmse}")  # Print RMSE
print(f"mae: {mae}")  # Print MAE

# Plot a sample of predictions vs. actual values
plt.figure()  # Create a new figure
plt.plot(y_test_orig[:200], label="Actual Values", color="blue")  # Plot actual values
plt.plot(predictions_orig[:200], label="Predictions", color="orange")  # Plot predictions
plt.title("GRU Model - Traffic Volume Prediction")  # Set plot title
plt.xlabel("Time")  # Set x-axis label
plt.ylabel("Traffic Volume")  # Set y-axis label
plt.legend()  # Show legend
plt.show()  # Display plot
