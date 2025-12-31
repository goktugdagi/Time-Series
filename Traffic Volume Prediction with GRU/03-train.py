import numpy as np  # Import numpy for array operations
import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch neural network module
import torch.optim as optim  # Import optimizers
import matplotlib.pyplot as plt  # Import matplotlib for plotting losses
from torch.utils.data import DataLoader, TensorDataset  # Import dataset/loader utilities

# Hyperparameters
INPUT_SIZE = 7  # Number of input features
HIDDEN_SIZE = 64  # Hidden size of GRU
NUM_LAYERS = 2  # Number of GRU layers
OUTPUT_SIZE = 1  # Output size (single regression value)
BATCH_SIZE = 64  # Batch size
LEARNING_RATE = 0.001  # Learning rate
NUM_EPOCHS = 15  # Maximum number of epochs

# Early stopping settings
PATIENCE = 3  # Stop if validation loss does not improve for 3 epochs
MIN_DELTA = 1e-4  # Minimum improvement threshold to be considered as "improved"

# Load train/validation data
X_train = np.load("X_train.npy")  # Train inputs (N, 24, 7)
y_train = np.load("y_train.npy")  # Train labels (N, 1)

X_val = np.load("X_val.npy")  # Validation inputs
y_val = np.load("y_val.npy")  # Validation labels

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert train inputs to float32 tensor
y_train = torch.tensor(y_train, dtype=torch.float32)  # Convert train labels to float32 tensor

X_val = torch.tensor(X_val, dtype=torch.float32)  # Convert val inputs to float32 tensor
y_val = torch.tensor(y_val, dtype=torch.float32)  # Convert val labels to float32 tensor

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)  # Wrap train tensors into a dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Train loader (shuffled)

val_dataset = TensorDataset(X_val, y_val)  # Wrap validation tensors into a dataset
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Validation loader (no shuffle)

# Define the GRU model
class GRUNet(nn.Module):  # Define a GRU-based neural network
    def __init__(self, input_size, hidden_size, num_layers, output_size):  # Initialize model params
        super(GRUNet, self).__init__()  # Call parent constructor
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected output layer

    def forward(self, x):  # Forward pass
        out, _ = self.gru(x)  # GRU outputs for all time steps
        last_out = out[:, -1, :]  # Take the last time step output
        out = self.fc(last_out)  # Map to final prediction
        return out  # Return predictions

# Instantiate the model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)  # Create model instance

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

# Track losses
train_loss_list = []  # Store average train loss per epoch
val_loss_list = []  # Store average validation loss per epoch
best_val_loss = float("inf")  # Initialize best validation loss

# Early stopping variables
patience_counter = 0  # Counts epochs without improvement

for epoch in range(NUM_EPOCHS):  # Loop over epochs
    model.train()  # Set model to training mode
    epoch_train_loss = 0  # Accumulate training loss for this epoch

    for X_batch, y_batch in train_loader:  # Iterate over training batches
        outputs = model(X_batch)  # Forward pass to get predictions
        loss = criterion(outputs, y_batch)  # Compute training loss

        optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update model parameters

        epoch_train_loss += loss.item()  # Add batch loss to epoch total

    avg_train_loss = epoch_train_loss / len(train_loader)  # Compute average train loss
    train_loss_list.append(avg_train_loss)  # Store it

    # Validation step
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0  # Accumulate validation loss
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in val_loader:  # Iterate over validation batches
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute validation loss
            epoch_val_loss += loss.item()  # Add batch loss to epoch total

    avg_val_loss = epoch_val_loss / len(val_loader)  # Compute average validation loss
    val_loss_list.append(avg_val_loss)  # Store it

    # Save the best model checkpoint based on validation loss
    if (best_val_loss - avg_val_loss) > MIN_DELTA:  # Check if validation improved meaningfully
        best_val_loss = avg_val_loss  # Update best validation loss
        torch.save(model.state_dict(), "gru_model.pth")  # Save best model weights
        patience_counter = 0  # Reset patience counter
        improved = True  # Improvement flag
    else:
        patience_counter += 1  # Increase patience counter
        improved = False  # No improvement flag

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Improved: {improved}, "
        f"Patience: {patience_counter}/{PATIENCE}"
    )  # Print training progress

    # Early stopping condition
    if patience_counter >= PATIENCE:  # If no improvement for PATIENCE epochs
        print(
            f"Early stopping triggered. "
            f"Best Val Loss: {best_val_loss:.4f} -> saved to gru_model.pth"
        )  # Print early stopping message
        break  # Stop training

# Plot loss curves
plt.figure()  # Create a new figure
plt.plot(train_loss_list, marker="o", label="Train Loss")  # Plot training loss
plt.plot(val_loss_list, marker="o", label="Val Loss")  # Plot validation loss
plt.title("Training and Validation Loss Curve")  # Set plot title
plt.xlabel("Epoch")  # Set x-axis label
plt.ylabel("Loss")  # Set y-axis label
plt.grid(True)  # Show grid
plt.legend()  # Show legend
plt.tight_layout()  # Tight layout
plt.show()  # Display plot
