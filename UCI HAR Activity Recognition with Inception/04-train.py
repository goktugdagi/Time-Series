import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from inception_model import build_inception_model

# Upload data
X_train = np.load("X_train.npy") # (num_samples, 128, 9(features))
y_train = np.load("y_train.npy") # (num_samples, 6(num_class))

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Create an Inception model.
input_shape = (X_train.shape[1], X_train.shape[2]) # (128, 9)
num_classes = y_train.shape[1]

model = build_inception_model(input_shape, num_classes)
model.compile(
    loss = "categorical_crossentropy",
    optimizer = Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# Training + Validation
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs = 10,
    batch_size=64,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


plt.figure(figsize=(10, 4))

# Loss graph
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy graph
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()