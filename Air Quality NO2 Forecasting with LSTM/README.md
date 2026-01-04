# Air Quality NO2 Forecasting with LSTM (PyTorch) + FastAPI + Streamlit

This repository provides a leakage‑free multivariate time‑series forecasting pipeline to predict NO2(GT) (Nitrogen Dioxide concentration) using the UCI Air Quality dataset.

## Implemented Fixes & Improvements
- Leakage‑free MinMaxScaler fit (train‑only covered rows)
- Correct validation loss logging
- Safe JSON parsing in Streamlit (no eval)
- Target‑only inverse scaling for NO2
- API input validation edge‑case checks

## Project Structure
01‑Load_and_explore.py → EDA + missing handling + plots  
02‑preprocessing.py → time split + scaler fit(train‑only) + 72‑step windows + save .npy + scaler.pkl  
03‑train.py → stacked LSTM training + validation + early stopping  
04‑test.py → offline evaluation + inverse scaling + MAE/MSE + plot  
05‑main_api.py → FastAPI POST /predict  
06‑test_api.py → API client test  
07‑app_streamlit.py → Streamlit UI calling API

## Installation (Windows / VS Code)
```bash
python ‑m venv .venv
.venv\Scripts\activate
pip install ‑r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib scikit‑learn torch fastapi uvicorn streamlit requests joblib
```

## How to Run
1) Preprocess → `python 02‑preprocessing.py`  
2) Train → `python 03‑train.py`  
3) Test offline → `python 04‑test.py`  
4) Run API → `uvicorn 05‑main_api:app ‑‑reload`  
5) Test API → `python 06‑test_api.py`  
6) Run UI → `streamlit run 07‑app_streamlit.py`

## Request Format (API)
POST /predict
```json
{ "sequence": [[7 floats], ... 72 rows] }
```
Response:
```json
{ "tahmin_edilen_NO2": 128.75 }
```

## License
Educational and portfolio use.


