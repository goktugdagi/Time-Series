"""
Sensor Defect Detection with LSTM/FCN

Problem definition:

- Automatically predict whether a product is manufactured defectively or normally by looking at sensor data on a production line.
- This allows for quick detection of defects, reduction of maintenance costs, and improvement of the quality control process.

Dataset:
- UCI SECOM Manufacturing Dataset
- https://archive.ics.uci.edu/ml/datasets/secom
- 1567 samples, each consisting of 590 sensor measurements
- Labels:
- "-1": normal production
- "1": defective production
- Features consist of numerical values, and some sensors have missing values
- Problem: The dataset is severely unbalanced (93% normal, 7% defective)

Tools/Technologies:
- LSTM-FCN: pytorch
- imbalanced-learn (smote): use to balance the dataset (oversampling)

Plan/Schedule:
- 1_load_data.py: Load data and perform initial analysis
- preprocessing.py: Fill in missing data, normalize, and balance with Smote
- model.py: Create LSTM + 1D CNN (FCN) architecture
- 4_train.py: Train the model
- 5_test.py: Test the model

install libraries: freeze
pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn tqdm imbalanced-learn

"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load sensor data
df = pd.read_csv("secom.data", sep=r"\s+", header=None)
print(df.head())

# Load label data
labels = pd.read_csv("secom_labels.data", sep=r"\s+", header=None)

# Append label column to sensor dataset
df["label"] = labels[0].values

print(df.shape)  # (1567, 591)

print(f"Class distribution: \n {df['label'].value_counts()}")
# -1 → Normal (1463)
#  1 → Faulty (104)

# Missing value analysis
missing_per_column = df.isnull().sum()
total_missing = missing_per_column.sum()
print(f"Total missing values: {total_missing}")  # 41951

# Compute missing ratio and sort descending
missing_ratio = (missing_per_column / len(df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
print(missing_ratio)

# Visualize missing data columns
plt.figure()
sns.barplot(x=missing_ratio.index, y=missing_ratio.values)
plt.title("Columns with Missing Data (%)")
plt.xlabel("Feature Index")
plt.ylabel("Missing Ratio (%)")
plt.xticks([])
plt.tight_layout()
plt.show()

# Scatter plot using first 2 features
plt.figure()
sns.scatterplot(data=df, x=0, y=1, hue="label", alpha=0.6, palette="Set1")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.tight_layout()
plt.show()
