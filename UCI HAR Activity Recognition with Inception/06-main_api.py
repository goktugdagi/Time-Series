from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(
    title="UCI HAR Activity Recognition API",
    description="Predicts human activity from a 128x9 inertial signal window using a Keras Inception1D model."
)

# Constants 
SEQ_LEN = 128
N_CHANNELS = 9

activity_labels = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

try:
    model = load_model("best_model.h5")
except Exception as e:
    raise (f"Model could not be loaded (best_model.h5). Error: {e}")

try:
    scalers = joblib.load("scalers.pkl")
except Exception as e:
    raise RuntimeError(f"Scalers could not be loaded (scalers.pkl). Error: {e}")

# Basic sanity check for scaler
if not isinstance(scalers, dict) or any(i not in scalers for i in range(N_CHANNELS)):
    raise RuntimeError("scalers.pkl must be a dict containing keys 0..8 for 9 channels.")

# Request Schema
class SequenceInput(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="2D list with shape [128][9] representing one inertial window."
    )
    top_k: int = Field(
        3,
        ge=1,
        le=6,
        description="How many top classes to return (1..6)."
    )


def validate_and_prepare(sequence: list[list[float]]) -> np.ndarray:
    """ 
        Validate shape and convert to np.array(1, 128, 9)
    """
    if sequence is None or len(sequence) == 0:
        raise HTTPException(status_code=400, detail="The 'sequence' field cannot be empty.")
    
    if len(sequence) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sequence length. Expected {SEQ_LEN} rows (time steps)."
        )
    

    # Validate each row
    for idx, row in enumerate(sequence):
        if not isinstance(row, list) or len(row) != N_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid row at index {idx}. Expected a list of length {N_CHANNELS}."
            )
    
    # Convert 
    X = np.asarray(sequence, dtype=np.float32) # (128, 9)
    X = X.reshape(1, SEQ_LEN, N_CHANNELS) # (1, 128, 9)
    return X

def apply_channel_scaling(X: np.ndarray) -> np.ndarray:
    """
        Apply channel-wise StandardScaler exactly like training code:
        scaler.fit_transform was donw on train[:, :, i] (2D)
        so here we transform X[:, :, i] with the saved scaler.
    """

    X_scaled = np.zeros_like(X, dtype=np.float32)

    for i in range(N_CHANNELS):
        scaler = scalers[i]
        X_scaled[:, :, i] = scaler.transform(X[:, :, i])

    return X_scaled

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "scalers_loaded": True}

@app.post("/predict")
def predict(payload: SequenceInput):
    try:
        # Validate + shape
        X = validate_and_prepare(payload.sequence) #(1, 128, 9)

        # Scale with saved scalers
        X_scaled = apply_channel_scaling(X) # (1, 128, 9)

        # Predict
        probs = model.predict(X_scaled,verbose=0)[0] # (6, )
        pred_class = int(np.argmax(probs))
        pred_label = activity_labels[pred_class]
        confidence = float(probs[pred_class])

        # 4) Top-k
        top_k = int(payload.top_k)
        top_idx = np.argsort(probs)[::-1][:top_k]
        top = [
            {"class": int(i), "label": activity_labels[int(i)], "prob": float(probs[int(i)])}
            for i in top_idx
        ]

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence, 6),
            "top_k": top,
            "probabilities": {activity_labels[i]: float(probs[i]) for i in range(len(activity_labels))}
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))