import torch  # PyTorch core
import torch.nn as nn  # Neural network layers
import numpy as np  # Load NumPy arrays saved from preprocessing
import matplotlib.pyplot as plt  # Plot training curves
from torch.utils.data import DataLoader, TensorDataset  # Mini-batching utilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, else CPU
print(f"Device: {device}")  # Print selected device

# Load Train / Val data  # Produced by 02-preprocessing.py
X_train = np.load("X_train.npy")  # Training input windows
y_train = np.load("y_train.npy")  # Training targets

X_val = np.load("X_val.npy")  # Validation input windows
y_val = np.load("y_val.npy")  # Validation targets

# Convert NumPy -> Torch tensors  # Required for PyTorch model training
X_train = torch.tensor(X_train, dtype=torch.float32)  # Inputs as float32 tensor
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Targets as (N, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)  # Validation inputs
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)  # Validation targets as (N, 1)

# Model hyperparameters  # Configuration for LSTM network and training
input_size = X_train.shape[2]  # Number of features (e.g., 7)
hidden_size = 128  # Hidden state size
num_layers = 3  # Number of stacked LSTM layers
dropout_rate = 0.2  # Dropout applied between LSTM layers (when num_layers > 1)
output_size = 1  # Single regression output (NO2)
num_epochs = 100  # Maximum training epochs
learning_rate = 0.001  # Adam learning rate
patience = 10  # Early stopping patience (epochs without improvement)
min_delta = 1e-6  # Minimum improvement threshold to reset early stopping
best_val_loss = float("inf")  # Track best validation loss
early_stop_counter = 0  # Counter for early stopping
batch_size = 64  # Mini-batch size
num_workers = 0  # DataLoader subprocesses (0 recommended on Windows)
pin_memory = torch.cuda.is_available()  # Pin memory if using CUDA for faster host->device copies

train_loader = DataLoader(  # DataLoader for training data
    TensorDataset(X_train, y_train),  # Dataset of (X, y)
    batch_size=batch_size,  # Batch size
    shuffle=True,  # Shuffle batches for better generalization (windows remain fixed; temporal order inside windows is preserved)
    num_workers=num_workers,  # Worker processes
    pin_memory=pin_memory,  # Pin memory on CUDA
    drop_last=False,  # Keep last smaller batch if exists
)

val_loader = DataLoader(  # DataLoader for validation data
    TensorDataset(X_val, y_val),  # Dataset of (X, y)
    batch_size=batch_size,  # Batch size
    shuffle=False,  # No shuffling
    num_workers=num_workers,  # Worker processes
    pin_memory=pin_memory,  # Pin memory on CUDA
    drop_last=False,  # Keep last smaller batch if exists
)

# LSTM-based model definition  # Stacked LSTM + linear head
class LSTMModel(nn.Module):  # Define model class
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):  # Constructor
        super(LSTMModel, self).__init__()  # Initialize base nn.Module
        self.lstm = nn.LSTM(  # LSTM layer
            input_size,  # Number of input features
            hidden_size,  # Hidden size
            num_layers,  # Number of stacked LSTM layers
            batch_first=True,  # Input format: (batch, seq, features)
            dropout=dropout,  # Dropout between layers (if num_layers > 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer mapping hidden -> prediction

    def forward(self, x):  # Forward pass
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # Take last time step representation
        out = self.fc(out)  # Map to output
        return out  # Return prediction tensor (batch, 1)

# Instantiate model  # Create model and move to device
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)  # Model on GPU/CPU

# Loss function and optimizer  # Regression objective
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training + validation loop  # Track loss curves
train_loss_list = []  # Store training loss per epoch
val_loss_list = []  # Store validation loss per epoch

for epoch in range(num_epochs):  # Epoch loop
    model.train()  # Enable training mode (dropout, etc.)
    running_train_loss = 0.0  # Accumulate batch losses (sum)

    for xb, yb in train_loader:  # Iterate over mini-batches
        xb, yb = xb.to(device), yb.to(device)  # Move batch to device

        output = model(xb)  # Forward pass
        train_loss = criterion(output, yb)  # Compute loss

        optimizer.zero_grad()  # Clear gradients
        train_loss.backward()  # Backpropagate
        optimizer.step()  # Update parameters

        running_train_loss += train_loss.item() * xb.size(0)  # Sum loss over samples

    epoch_train_loss = running_train_loss / len(train_loader.dataset)  # Average training loss
    train_loss_list.append(epoch_train_loss)  # Store

    # Validation phase  # Evaluate model on validation set
    model.eval()  # Evaluation mode
    running_val_loss = 0.0  # Accumulate validation loss

    with torch.inference_mode():  # Disable grad tracking (faster + less memory)
        for xb, yb in val_loader:  # Iterate validation batches
            xb, yb = xb.to(device), yb.to(device)  # Move to device

            val_output = model(xb)  # Forward
            val_loss = criterion(val_output, yb)  # Compute validation loss

            running_val_loss += val_loss.item() * xb.size(0)  # Sum over samples

    epoch_val_loss = running_val_loss / len(val_loader.dataset)  # Average validation loss
    val_loss_list.append(epoch_val_loss)  # Store

    # Early stopping  # Save best model by validation loss
    if epoch_val_loss < best_val_loss - min_delta:  # Improvement check
        best_val_loss = epoch_val_loss  # Update best
        early_stop_counter = 0  # Reset counter
        torch.save(model.state_dict(), "lstm_model_best.pt")  # Save best model weights
    else:
        early_stop_counter += 1  # No improvement -> increment counter

    if (epoch + 1) % 10 == 0:  # Log every 10 epochs
        print(  # Print progress
            f"Epoch [{epoch + 1}/{num_epochs}],"
            f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"  # Print correct validation loss
        )
    
    if early_stop_counter >= patience:  # If no improvement for 'patience' epochs
        print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")  # Report stopping
        break  # Exit training loop

# Plot loss curves # Visualize training dynamics
plt.figure(figsize=(10, 4)) #Figure size
plt.plot(train_loss_list, label="Train Loss") # Training loss line
plt.plot(val_loss_list, label="Val Loss") # Validation loss line
plt.title("Training / Validation Loss Graph (MSE)") # Title
plt.xlabel("Epoch") # X-axis label
plt.ylabel("Loss") # Y-axis label
plt.grid() # Grid
plt.legend() # Legend
plt.tight_layout() # Layout adjustment
plt.show() # Display

# Save final model # Also persist last-epoch weights (best weights already saved separately)
torch.save(model.state_dict(), "lstm_model.pt") # Save final weights
print("Model Successfully saved as 'lstm_model.pt'.") # Confirmation
print("According to the validation, the best model was saved as 'lstm_model_best.pt'.") # Confirmation