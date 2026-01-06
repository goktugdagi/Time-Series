import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import train_loader, test_loader
from tqdm import tqdm

# Device selection: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = LSTMFCN().to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20

# Training loop
for epoch in range(num_epochs):

    model.train()  # Set model to training mode
    train_loss = 0  # Initialize training loss
    correct = 0
    total = 0

    # Loop over training data
    for inputs, labels in tqdm(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Make prediction
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Compute average loss
    avg_loss = train_loss / total

    # Compute accuracy
    accuracy = correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

# Save model weights
torch.save(model.state_dict(), "lstmfcn_secom.pth")
print("Model saved")
