import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization for results
from tensorflow.keras.models import load_model  # Load trained LSTM model
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Evaluation metrics
import joblib  # Load scaler

# Load the test dataset and model
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

# Predict on test set (scaled 0-1 output)
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions and ground truth back to original scale (kW)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# Compute final evaluation metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Print model performance
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Plot first 200 hours of test predictions vs real values
plt.figure()
plt.plot(y_true[:200], label="Actual Energy (kW)", linewidth=2)
plt.plot(y_pred[:200], label="Predicted Energy (kW)", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Energy Consumption (Test Set)")
plt.xlabel("Hours")
plt.ylabel("kW")
plt.legend()
plt.grid(True)
plt.show()
