# Energy Consumption Forecasting with LSTM

## Overview
This project implements an LSTM-based time-series forecasting pipeline to predict household electricity consumption using the UCI *Individual Household Electric Power Consumption* dataset. The model learns from historical hourly power usage and forecasts future energy demand while ensuring a strict chronological split across **train**, **validation**, and **test** sets to prevent data leakage.

## Project Structure
- `01-load_and_clean.py` → Loads raw UCI data, cleans missing values, creates a datetime index, and exports hourly aggregates
- `02-data_prepration.py` → Normalizes the hourly data, generates sliding-window sequences, and performs chronological train/val/test splits
- `03-lstm_train.py` → Builds and trains the LSTM model using the validation set for monitoring
- `04-lstm_test.py` → Evaluates the trained model on the test set using MAE and RMSE
- `05-forecast_future.py` → Uses the final test window to forecast the next 24 hours and visualizes predictions

## Dataset
- **Source:** UCI Individual Household Electric Power Consumption Dataset
- **Target Feature:** `global_active_power` (Total active power, kW)
- **Sampling:** Resampled into hourly averages
- **Missing Values:** Represented by `?`, converted to NaN and removed

## Methodology
1. **Load & Clean:** Parse raw data, combine date-time fields, and build a datetime index
2. **Resample:** Convert power consumption to hourly means
3. **Sliding Window:** Create 24-hour input sequences (`X`) to predict the next hour (`y`)
4. **Chronological Split:** 70% Train / 15% Validation / 15% Test (no shuffle)
5. **Scaling:** MinMax normalization (0–1 range), saved for reuse in test/forecast
6. **Training:** LSTM with `tanh` activation, optimized with `Adam`, loss = `MSE`, monitored via `EarlyStopping`
7. **Evaluation:** MAE & RMSE on test set only
8. **Forecasting:** Iteratively predict next 24 hours from last test window

## Model Details
- LSTM (64 units, activation=`tanh`)
- Dense (1 output unit)

## Results
Performance metrics are computed in the test phase only:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

The 24-hour forecast is compared visually against actual values when available.

## Usage

### Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn joblib tensorflow
