# CNN‑LSTM Weather Forecast API

## Overview
This repository provides an end‑to‑end, **reproducible** pipeline for **hourly temperature forecasting** using a **CNN‑LSTM** model (TensorFlow/Keras), exposed as a **FastAPI** inference service and demonstrated via a **Streamlit** UI.

The model predicts **next‑hour temperature (°C)** given the **last 24 hours** of 5 weather signals.

## Problem Definition
**Input (24×5 window):**
- `humidity`
- `wind_speed_km_h`
- `pressure_millibars`
- `visibility_km`
- `apparent_temperature_c`

**Output:** `temperature_c` (single scalar, next hour)

**Important:** The API expects the input feature order **exactly as above** and a strict shape of **(24, 5)**.

## Dataset
This project expects the Kaggle “Weather History” dataset with the file:
- `weatherHistory.csv`

Place `weatherHistory.csv` in the **project root** (same folder as the scripts).

Source:
- Kaggle dataset: https://www.kaggle.com/datasets/muthuj7/weather-dataset

## Repository Structure
Core scripts:
- `01-load_and_explore.py` — Load CSV, basic cleaning/column normalization, and exploratory plots.
- `02-preprocessing.py` — Feature selection, outlier filtering, **chronological** train/val/test split, leakage‑safe scaling, and 24‑hour windowing to `*.npy`.
- `03-train_model_cnn_lstm_.py` — Build and train the CNN‑LSTM model using train/val, save the trained model.
- `04-test_and_visualize.py` — Evaluate on test set, metrics + plots.

Serving & demo:
- `05-main_api.py` — FastAPI app exposing `POST /predict` for 24×5 sequences.
- `06-test_requests.py` — Quick local API test (random sample from test set).
- `07-app_streamlit.py` — Streamlit UI for manual 24‑hour input and prediction.

Generated artifacts (created after running the pipeline):
- `scaler.pkl` — `StandardScaler` fit on train only (leakage‑safe).
- `cnn_lstm_weather_model.h5` — Trained Keras model.
- `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`

## Installation (Windows)
Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow fastapi uvicorn streamlit requests joblib scipy
```

Notes:
- `02-preprocessing.py` uses `scipy` for z‑score outlier filtering.
- `04-test_and_visualize.py` uses `root_mean_squared_error`; for this function, use a recent `scikit-learn` (recommended: **>= 1.4**).

## Quickstart (End‑to‑End)
1) **EDA / sanity checks**
```bash
python 01-load_and_explore.py
```

2) **Preprocess + create train/val/test sequences**
```bash
python 02-preprocessing.py
```
This generates `scaler.pkl` and the `*.npy` split files.

3) **Train model**
```bash
python 03-train_model_cnn_lstm_.py
```
This generates `cnn_lstm_weather_model.h5`.

4) **Evaluate on test and visualize**
```bash
python 04-test_and_visualize.py
```

## Run the API (FastAPI)
Start the server:
```bash
python -m uvicorn 05-main_api:app --reload
```

Open Swagger UI:
- `http://127.0.0.1:8000/docs`

### API Contract
**Endpoint:** `POST /predict`

**Request body:**
```json
{
  "sequence": [
    [humidity, wind_speed_km_h, pressure_millibars, visibility_km, apparent_temperature_c],
    ...
    (24 rows total)
  ]
}
```

**Response:**
```json
{ "predicted_temperature_c": 18.61 }
```

### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{"sequence": [[50,10,1010,10,15],[... 24 rows total ...]]}"
```

## Local API Test (Python)
After starting the API, run:
```bash
python 06-test_requests.py
```
This script picks a random sample from `X_test.npy`, converts it back to raw feature space via `scaler.pkl`, calls the API, and prints predicted vs actual.


## Streamlit Demo UI
In a separate terminal (while FastAPI is running):
```bash
streamlit run 07-app_streamlit.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Implementation Notes
- **Leakage‑safe preprocessing:** scaling is fit on **train only**; validation and test are transformed with the same scaler.
- **Chronological split:** the dataset is split by time order before creating sequences, so windows never cross split boundaries.
- **Column normalization:** the preprocessing normalizes `/` to `_` so `wind_speed_km/h` becomes `wind_speed_km_h`.

## Common Issues / Troubleshooting
- **`FileNotFoundError: weatherHistory.csv`**  
  Ensure `weatherHistory.csv` exists in the project root (same directory where you run the scripts).
- **API fails on startup (missing model/scaler)**  
  Run `02-preprocessing.py` and `03-train_model_cnn_lstm_.py` first to generate `scaler.pkl` and `cnn_lstm_weather_model.h5`.
- **Input shape errors (`Expected (24, 5)`)**  
  Ensure your request includes exactly 24 rows and 5 columns in the correct feature order.

## License
MIT. (Add a `LICENSE` file if it is not present in the repository.)

