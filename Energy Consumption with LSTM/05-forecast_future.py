import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization for forecast
from tensorflow.keras.models import load_model  # Load trained LSTM model
import joblib  # Load scaler
import pandas as pd  # Data loading

# Load model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

# Load the original hourly time series
df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col=0,
    parse_dates=True
)

# Take the last 48 hours → first 24 is model input, next 24 is ground truth comparison
last_48 = df_hourly.iloc[-48:].copy()
real_next_24 = last_48.iloc[24:].values.flatten()

# Use last test window as the initial forecast input
X_test = np.load("X_test.npy")
forecast_input = X_test[-1].copy()

# Generate 24-hour future forecast
future_predictions = []
for _ in range(24):
    input_3d = forecast_input.reshape(1, forecast_input.shape[0], 1)
    next_scaled = model.predict(input_3d, verbose=0)[0]
    next_value = scaler.inverse_transform(next_scaled.reshape(1, -1))[0][0]
    future_predictions.append(next_value)
    forecast_input = np.vstack((forecast_input[1:], next_scaled.reshape(1, 1)))

# Plot forecast vs actual next 24 hours
plt.figure()
plt.plot(real_next_24, label="Actual Next 24 Hours (kW)", linewidth=2)
plt.plot(future_predictions, label="Forecast Next 24 Hours (kW)", linestyle="--", linewidth=2)
plt.title("Next 24-Hour Actual vs LSTM Forecast")
plt.xlabel("Hours Ahead")
plt.ylabel("kW")
plt.legend()
plt.grid(True)
plt.show()
