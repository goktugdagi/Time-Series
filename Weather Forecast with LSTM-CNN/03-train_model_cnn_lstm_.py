# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# X_train = np.load("X_train.npy")
# y_train = np.load("y_train.npy")
# X_val = np.load("X_val.npy")
# y_val = np.load("y_val.npy")
# X_test = np.load("X_test.npy")
# y_test = np.load("y_test.npy")

# print("X_train:", X_train.shape, "y_train:", y_train.shape)
# input_shape = X_train.shape[1:]  # (24, 5)

# # Build CNN-LSTM model
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(LSTM(units=64, return_sequences=False))
# model.add(Dense(units=1))

# model.compile(
#     optimizer="adam",
#     loss="mean_squared_error",
#     metrics=["mae"],
# )

# model.summary()

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=5,
#     restore_best_weights=True,
# )

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=15,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=1,
# )

# # Plot loss
# plt.figure()
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Val Loss")
# plt.title("Training Curve (MSE)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot MAE
# plt.figure()
# plt.plot(history.history["mae"], label="Train MAE")
# plt.plot(history.history["val_mae"], label="Val MAE")
# plt.title("Training Curve (MAE)")
# plt.xlabel("Epoch")
# plt.ylabel("MAE")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Quick evaluation on test
# test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Loss (MSE): {test_loss}")
# print(f"Test MAE: {test_mae}")

# model.save("cnn_lstm_weather_model.h5")
# print("Saved model: cnn_lstm_weather_model.h5")

# Import TensorFlow and model building utilities
import tensorflow as tf  # Deep learning framework
from tensorflow.keras.models import Sequential  # Linear model container
from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Dense, Dropout  # Model layers
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping callback
import numpy as np  # Data loading
import matplotlib.pyplot as plt  # Training visualization

# Load preprocessed train/validation/test splits
X_train = np.load("X_train.npy")  # Training sequences
y_train = np.load("y_train.npy")  # Training labels
X_val = np.load("X_val.npy")  # Validation sequences
y_val = np.load("y_val.npy")  # Validation labels
X_test = np.load("X_test.npy")  # Test sequences
y_test = np.load("y_test.npy")  # Test labels

# Print shapes for sanity check
print("X_train:", X_train.shape, "y_train:", y_train.shape)  # Confirm loaded shapes

# Derive model input shape (24, 5) from training data
input_shape = X_train.shape[1:]  # Use timesteps x features shape

# Build the CNN-LSTM model architecture
model = Sequential()  # Initialize model
model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))  # 1D CNN for local pattern extraction
model.add(BatchNormalization())  # Normalize CNN output for stable training
model.add(Dropout(0.2))  # Reduce overfitting
model.add(LSTM(units=64, return_sequences=False))  # LSTM for temporal dependency modeling
model.add(Dense(units=1))  # Output temperature prediction

# Compile model with optimizer, loss, and metrics
model.compile(
    optimizer="adam",  # Adaptive learning rate optimizer
    loss="mean_squared_error",  # Regression loss function
    metrics=["mae"],  # Mean Absolute Error metric for interpretability
)

# Print model summary
model.summary()  # Display architecture

# Configure early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss",  # Watch validation loss
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True,  # Keep best performing weights
)

# Train model with validation set
history = model.fit(
    X_train, y_train,  # Training data
    validation_data=(X_val, y_val),  # Validation data
    epochs=15,  # Maximum epoch count
    batch_size=32,  # Batch size for gradient updates
    callbacks=[early_stopping],  # Early stopping callback
    verbose=1,  # Logging verbosity
)

# Visualize training and validation loss
plt.figure()  # Create new figure
plt.plot(history.history["loss"], label="Train Loss")  # Training loss curve
plt.plot(history.history["val_loss"], label="Validation Loss")  # Validation loss curve
plt.title("Training Curve (MSE Loss)")  # Chart title
plt.xlabel("Epoch")  # X-axis label
plt.ylabel("Loss")  # Y-axis label
plt.legend()  # Display legend
plt.grid(True)  # Enable grid
plt.show()  # Render plot

# Visualize training and validation MAE
plt.figure()  # Create new figure
plt.plot(history.history["mae"], label="Train MAE")  # Training MAE curve
plt.plot(history.history["val_mae"], label="Validation MAE")  # Validation MAE curve
plt.title("Training Curve (MAE)")  # Chart title
plt.xlabel("Epoch")  # X-axis label
plt.ylabel("MAE")  # Y-axis label
plt.legend()  # Display legend
plt.grid(True)  # Enable grid
plt.show()  # Render plot

# Evaluate model on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)  # Compute test metrics silently
print(f"Test Loss (MSE): {test_loss}")  # Print MSE loss
print(f"Test MAE: {test_mae}")  # Print MAE

# Save model artifact to disk
model.save("cnn_lstm_weather_model.h5")  # Persist trained model
print("Saved model: cnn_lstm_weather_model.h5")  # Confirmation log
