import numpy as np  # Numerical operations
from tensorflow.keras.models import Sequential  # Model container for layer stacking
from tensorflow.keras.layers import LSTM, Dense  # LSTM for sequence modeling, Dense for output
from tensorflow.keras.callbacks import EarlyStopping  # Stop training when validation loss stalls
from tensorflow.keras.losses import MeanSquaredError  # Regression loss function (MSE)
import matplotlib.pyplot as plt  # Plot training curves

# Load training and validation datasets
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Initialize sequential LSTM model
model = Sequential()

# Add LSTM layer with 64 memory cells and tanh activation
model.add(LSTM(
    64,  # Number of LSTM units (memory cells)
    activation="tanh",
    input_shape=(X_train.shape[1], X_train.shape[2])  # (24, 1) → (time steps, features)
))

# Add Dense output layer for 1-step energy consumption prediction
model.add(Dense(1))

# Compile model with Adam optimizer and MSE loss
model.compile(
    optimizer="adam",
    loss=MeanSquaredError()
)

# Apply early stopping to avoid overfitting and recover best weights
early_stop = EarlyStopping(
    monitor="val_loss",  # Watch validation loss instead of train loss
    patience=5,  # Stop if no improvement for 5 consecutive epochs
    restore_best_weights=True  # Roll back to best model weights
)

# Train model using train and validation sets
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # Maximum training epochs
    batch_size=32,  # Number of samples processed per gradient update
    callbacks=[early_stop],
    verbose=1  # Print epoch-level logs
)

# Plot training and validation loss curves
plt.figure()
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("LSTM Model Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Save the trained LSTM model
model.save("lstm_model.h5")
