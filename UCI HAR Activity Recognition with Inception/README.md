# UCI HAR Activity Recognition (Inception1D + FastAPI + Streamlit)

End-to-end **Human Activity Recognition (HAR)** project built on the **UCI HAR Dataset**.  
It trains an **Inception-style 1D CNN** (TensorFlow/Keras) on **128×9 inertial windows**, serves predictions via **FastAPI**, and provides a **Streamlit** demo UI for interactive testing.

---

## What this project does

- Loads and stacks **9 inertial sensor channels** into windows of shape **(128 time steps, 9 channels)**
- Creates **Train / Validation / Test** splits (stratified)
- Applies **channel-wise StandardScaler** normalization (fit on train, transform on val/test)
- Trains an **Inception1D** classifier (softmax over **6 activities**)
- Exposes a production-like inference endpoint:
  - `POST /predict` → returns predicted label, confidence, top-k, and full probability distribution
- Provides a **Streamlit** interface for:
  - sample prediction from local `.npy` files
  - manual JSON input (advanced)
  - `.npy` upload
  - API health checks

---

## Activities (6 classes)

1. `WALKING`
2. `WALKING_UPSTAIRS`
3. `WALKING_DOWNSTAIRS`
4. `SITTING`
5. `STANDING`
6. `LAYING`

---

## Dataset

This project uses the **UCI HAR (Human Activity Recognition Using Smartphones) Dataset**.

Expected folder structure (relative to project root):

```
UCI HAR Dataset/
  train/
    Inertial Signals/
      body_acc_x_train.txt
      ...
    y_train.txt
  test/
    Inertial Signals/
      body_acc_x_test.txt
      ...
    y_test.txt
```

---

## Repository structure

```
.
├── 01-load_and_segment.py     # Loads inertial signals, creates X_*_raw.npy / y_*_raw.npy
├── 02-preprocessing.py        # Train/Val split + scaling + one-hot, saves X_train/val/test.npy
├── inception_model.py         # Inception1D model definition
├── 04-train.py                # Training loop (train + val) with callbacks, saves best_model.h5
├── 05-test.py                 # Evaluation: classification report + confusion matrix
├── 06-main_api.py             # FastAPI inference service (loads best_model.h5 + scalers.pkl)
├── 07-test_api.py             # Simple client script to test /predict endpoint
└── 08-app_streamlit.py        # Streamlit demo UI (API health + inference + visualization)
```

Artifacts produced during the workflow:

- `X_train_raw.npy`, `X_test_raw.npy`, `y_train_raw.npy`, `y_test_raw.npy`
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `scalers.pkl`
- `best_model.h5`

---

## Installation

Create a virtual environment (Windows):

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy scikit-learn joblib matplotlib seaborn
pip install tensorflow
pip install fastapi uvicorn streamlit requests pydantic
```

> Notes
- TensorFlow installation may differ depending on CPU/GPU setup.
- If you are using a GPU, ensure your CUDA/cuDNN stack matches your TensorFlow build.

---

## How to run (step-by-step)

### 1) Load & export raw windows
```bash
python 01-load_and_segment.py
```
Output:
- `X_train_raw.npy`, `X_test_raw.npy`
- `y_train_raw.npy`, `y_test_raw.npy`

### 2) Preprocess (split + scaling + one-hot)
```bash
python 02-preprocessing.py
```
Output:
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `scalers.pkl`

### 3) Train Inception1D
```bash
python 04-train.py
```
Output:
- `best_model.h5` (best checkpoint by validation accuracy)

### 4) Evaluate
```bash
python 05-test.py
```
Outputs:
- `classification_report` in console
- confusion matrix heatmaps for Validation and Test

---

## FastAPI inference service

Start the API:

```bash
uvicorn 06-main_api:app --host 127.0.0.1 --port 8000 --reload
```

### Endpoints

#### `GET /health`
Returns a simple health status.

#### `POST /predict`
**Request body**
```json
{
  "sequence": [[... 9 floats ...], "... total 128 rows ..."],
  "top_k": 3
}
```

- `sequence` must be **128×9**
- Returns predicted label, confidence, top-k list, and full probability distribution

**Response (example)**
```json
{
  "predicted_class": 4,
  "predicted_label": "STANDING",
  "confidence": 0.998712,
  "top_k": [
    {"class": 4, "label": "STANDING", "prob": 0.998712},
    {"class": 3, "label": "SITTING", "prob": 0.000812}
  ],
  "probabilities": {
    "WALKING": 0.0,
    "WALKING_UPSTAIRS": 0.0,
    "WALKING_DOWNSTAIRS": 0.0,
    "SITTING": 0.000812,
    "STANDING": 0.998712,
    "LAYING": 0.000476
  }
}
```

### API scaling behavior (important)

`06-main_api.py` loads `scalers.pkl` and applies **channel-wise scaling inside the API**.

- If you send **RAW** windows (recommended): send from `X_test_raw.npy` or original inertial signals  
- If you send **SCALED** windows: do **not** scale again in the API (otherwise you double-transform the input)

The Streamlit UI includes a RAW/SCALED selection to avoid this confusion.

---

## Test the API (client script)

With the API running:

```bash
python 07-test_api.py
```

---

## Streamlit demo UI

Start Streamlit:

```bash
streamlit run 08-app_streamlit.py
```

What you can do in the UI:
- Health check the API
- Load a sample from `.npy` and run predictions
- Upload an array (`(128, 9)` or `(N, 128, 9)`)
- View top-k predictions and probability distributions

---

## Troubleshooting

### 1) `name 'x' is not defined` (FastAPI 500)
In `06-main_api.py`, ensure you scale `X` not an undefined `x`:
```python
X_scaled = apply_channel_scaling(X)
```

### 2) “Prediction looks wrong”
Most common cause: **double scaling**
- Sending `X_test.npy` (already scaled) to an API that also scales internally
- Fix: send `X_test_raw.npy` to the API, or disable API scaling when sending scaled input

### 3) Streamlit cannot connect to API
Start API first:
```bash
uvicorn 06-main_api:app --reload
```

---

## Author
Göktuğ (GitHub: `goktugdagi`)
