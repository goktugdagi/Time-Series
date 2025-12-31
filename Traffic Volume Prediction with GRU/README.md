# GRU Traffic Forecasting API

An end-to-end deep learning project for **hourly traffic volume forecasting** using a **Gated Recurrent Unit (GRU)** neural network.  
The trained model is deployed as a REST API via **FastAPI (Uvicorn)** and accessed interactively through a **Streamlit web interface**.

---

## Dataset

- **Metro Interstate Traffic Volume (2012–2018)**
- Hourly measurements (~48K samples)
- Target column: `traffic_volume`
- Model input: 24-hour sequences, each timestep containing 7 features:

```
[temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]
```

- Source: UCI Machine Learning Repository

---

## Project Pipeline

```
Load Data → EDA → MinMax Scaling → Sliding Window (24h) →  
Chronological Train/Validation/Test Split → GRU Training (MSE + Adam) →  
Early Stopping → Save Model (.pth) → Serve with FastAPI → UI with Streamlit
```

---

## Repository Structure

```
repo-root/
│── 01-load_and_explore.py      # Exploratory Data Analysis
│── 02-preprocessing.py         # Scaling & train/val/test split
│── 03-train.py                 # GRU model training + early stopping
│── 04-test.py                  # Model evaluation (RMSE, MAE)
│── 05-main_api.py              # FastAPI model serving endpoint
│── 06-test_requests.py         # API request testing
│── 07-app_streamlit.py         # Streamlit UI for predictions
│── gru_model.pth               # Trained GRU weights
│── scaler_X.save               # Feature scaler
│── scaler_y.save               # Target scaler
│── X_train.npy, y_train.npy    # Training sequences
│── X_val.npy, y_val.npy        # Validation sequences
│── X_test.npy, y_test.npy      # Test sequences
│── README.md                   # Documentation
```

---

## Installation (Windows + VS Code Recommended)

```powershell
git clone https://github.com/goktugdagi/Computer-Vision.git
cd GRU-Traffic-API
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Make sure Git is installed and added to system PATH before cloning.

---

## Run the Project

### Train the model
```powershell
python 03-train.py
```

### Test the model
```powershell
python 04-test.py
```

### Start FastAPI server
```powershell
uvicorn 05-main_api:app --reload
```

### Start Streamlit UI
```powershell
streamlit run 07-app_streamlit.py
```

### Test API manually
```powershell
python 06-test_requests.py
```

---

## API Usage

### POST `/predict`

Example request body (24 timesteps, 7 features each):

```json
{
  "sequence": [
    [290.2, 0.0, 0.0, 40, 0, 1, 10],
    [291.1, 0.0, 0.0, 55, 1, 2, 10],
    ...
    (24 rows total)
  ]
}
```

Example response:

```json
{
  "predicted_traffic_volume": 1328.5
}
```

---

## Model Details

| Parameter | Value |
|---|---|
| Model Type | 2-layer GRU + FC |
| Task | Univariate regression |
| Input Window | 24 hours |
| Split Type | Chronological 70/15/15 |
| Loss | MSE (Mean Squared Error) |
| Optimizer | Adam (lr=0.001) |
| Overfitting Control | Early Stopping |
| Deployment | FastAPI + Streamlit |

---

## Important Notes

- Train batches are shuffled, **validation and test are not** (time-series safe)
- The same scalers are reused during inference
- Model runs on CPU/GPU depending on PyTorch environment
- All documentation and comments are in English for global readability
- Designed as a portfolio project and API-ready for deployment

---

## Possible Improvements (Not applied, future ideas)

- Multi-step forecasting support
- Cloud deployment with Docker (Render, Railway, GCP, AWS, etc.)
- Prediction logging & monitoring dashboard
- Automated model retraining pipeline
- Additional lag features or temporal decay feature engineering

---