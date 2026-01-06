# Sensor Fault Classification with LSTM-FCN (SECOM)

This project performs **fault detection and classification** using the **UCI SECOM Manufacturing Dataset**, which contains high-dimensional time-series sensor readings from semiconductor manufacturing processes. The goal is to classify each sample as **Normal** or **Faulty** using a leakage-free, imbalance-aware pipeline implemented in **PyTorch**.

The model architecture follows the **LSTM-FCN (Long Short-Term Memory + Fully Convolutional Network)** design for multivariate time-series classification, combining sequential learning with 1D convolutional feature extraction.

---

## Dataset

- **Source:** UCI SECOM Manufacturing Dataset  
- Contains:
  - `secom.data` → 590 sequential sensor measurements per sample
  - `secom_labels.data` → Labels (`-1` = Normal, `1` = Faulty)

### How to Obtain the Dataset

1. Visit the dataset page on UCI.
2. Download the archive and extract:
   - `secom.data`
   - `secom_labels.data`
3. Copy both files into the project root directory.
4. Ensure the filenames match exactly before running the pipeline.

---

## Project Structure

```
.
├── 01-load_data.py        # Loads dataset and performs initial EDA
├── preprocessing.py       # Missing-value handling, scaling, SMOTE, DataLoaders
├── model.py               # LSTM-FCN architecture definition
├── 04-train.py            # Model training and weight saving
├── 05-test.py             # Evaluation on test set (accuracy, F1, confusion matrix)
└── README.md              # Project documentation
```

---

## Installation (Windows, VS Code / PyCharm Compatible)

### Requirements

- Python 3.9+
- NVIDIA GPU recommended (CUDA support enabled automatically with PyTorch)
- Libraries:
  - `torch`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib`
  - `tqdm`

### Setup

```bash
pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib tqdm imbalanced-learn
```

---

## Usage

### 1. Exploratory Data Analysis (Optional)

```bash
python 01-load_data.py
```

This script:
- Prints class distribution
- Analyzes missing values
- Generates basic visualizations

### 2. Training

```bash
python 04-train.py
```

After training, model weights are saved as:

```
lstmfcn_secom.pth
```

### 3. Testing / Evaluation

```bash
python 05-test.py
```

Outputs include:
- Accuracy
- F1 score
- Classification report
- Confusion matrix heatmap

---

## Key Technical Notes

- The dataset is highly imbalanced (~93% Normal, ~7% Faulty). **SMOTE oversampling** is applied before training.
- The pipeline is **future-data leakage-free**.
- Recommended production-level improvements (not implemented in this learning repo):
  - Persisting `StandardScaler` for future inference
  - Adding validation split and early stopping
  - Including `ROC-AUC` and `PR-AUC` metrics for critical positive class focus
  - Setting deterministic seeds for full reproducibility

---

## Model Architecture Summary (`model.py`)

- **Input:** `[Batch, 590]` → reshaped to `[Batch, 590, 1]` for LSTM branch
- **LSTM branch:** Captures temporal dependencies, returns last hidden state
- **FCN branch:** 1D convolutional blocks + Global Average Pooling
- **Fusion:** Concatenation of LSTM and CNN features → Fully connected classifier

---

## License

This repository is a **learning and portfolio project**. The dataset is subject to UCI's original licensing terms. Ensure compliance before using the data outside educational or research purposes.

---