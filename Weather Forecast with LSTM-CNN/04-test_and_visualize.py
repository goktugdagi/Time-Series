# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# # Load test data
# X_test = np.load("X_test.npy")
# y_test = np.load("y_test.npy")

# # Load trained model
# model = tf.keras.models.load_model("cnn_lstm_weather_model.h5")

# # Predict
# y_pred = model.predict(X_test, verbose=0).flatten()

# # Metrics
# mae = mean_absolute_error(y_test, y_pred)
# rmse = root_mean_squared_error(y_test, y_pred)

# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")

# # Plot actual vs predicted
# plt.figure()
# plt.plot(y_test[:200], label="Actual", marker="o")
# plt.plot(y_pred[:200], label="Predicted", marker="x")
# plt.title("Actual vs Predicted Temperature (first 200 samples)")
# plt.xlabel("Time")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Error distribution
# errors = y_test - y_pred
# plt.figure()
# plt.hist(errors, bins=30, edgecolor="k")
# plt.title("Prediction Error Distribution")
# plt.xlabel("Error (Actual - Predicted)")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# Import evaluation and visualization utilities
import numpy as np  # Data loading
import matplotlib.pyplot as plt  # Plotting
import tensorflow as tf  # Load model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # Regression metrics

# Load test split sequences and labels
X_test = np.load("X_test.npy")  # Test sequences
y_test = np.load("y_test.npy")  # True labels

# Load the trained CNN-LSTM model
model = tf.keras.models.load_model("cnn_lstm_weather_model.h5")  # Load trained model

# Run predictions on test set
y_pred = model.predict(X_test, verbose=0).flatten()  # Predict and flatten to 1D array

# Compute metrics
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
rmse = root_mean_squared_error(y_test, y_pred)  # Root Mean Squared Error

# Print metric results
print(f"MAE: {mae}")  # Print MAE
print(f"RMSE: {rmse}")  # Print RMSE

# Plot first 200 actual vs predicted values
plt.figure()  # New figure
plt.plot(y_test[:200], label="Actual", marker="o")  # Plot true values
plt.plot(y_pred[:200], label="Predicted", marker="x")  # Plot predictions
plt.title("Actual vs Predicted Temperature (first 200 samples)")  # Chart title
plt.xlabel("Sample Index")  # X-axis label
plt.ylabel("Temperature (°C)")  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Enable grid
plt.show()  # Render plot

# Plot prediction error distribution
errors = y_test - y_pred  # Compute residual errors
plt.figure()  # New figure
plt.hist(errors, bins=30, edgecolor="k")  # Histogram of errors
plt.title("Prediction Error Distribution")  # Chart title
plt.xlabel("Error (Actual - Predicted)")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label
plt.grid(True)  # Enable grid
plt.show()  # Render plot
