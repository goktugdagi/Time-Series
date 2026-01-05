import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Modeli yukle
model = load_model("best_model.h5")

activity_labels = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS","SITTING", "STANDING", "LAYING"
]

def evaluate_split(split_name: str, X_path: str, y_path: str):
    X = np.load(X_path)
    y = np.load(y_path)

    # Tahmin
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y, axis=1)

    print("\n" + "=" * 60)
    print(f"{split_name} RESULTS")
    print("=" * 60)

    # Rapor
    print(classification_report(y_true, y_pred, target_names=activity_labels))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=activity_labels, yticklabels=activity_labels
    )
    plt.title(f"Confusion Matrix - {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

evaluate_split("VALIDATION", "X_val.npy", "y_val.npy")
evaluate_split("TEST", "X_test.npy", "y_test.npy")