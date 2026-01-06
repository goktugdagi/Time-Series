import torch
import torch.nn as nn
from model import LSTMFCN
from preprocessing import test_loader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model and load weights
model = LSTMFCN().to(device)
model.load_state_dict(torch.load("lstmfcn_secom.pth"))  # Load model parameters
model.eval()  # Switch to evaluation mode

# Define empty lists for predictions and ground-truth labels
all_preds = []
all_labels = []

with torch.inference_mode():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute scores
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_preds, all_labels)
print(f"Test accuracy: {acc:.4f}")
print(f"F1 score: {f1:.4f}")

print(f"Classification Report: \n {classification_report(all_labels, all_preds)}")

cm = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix: \n {cm}")

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Faulty"], yticklabels=["Normal", "Faulty"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
