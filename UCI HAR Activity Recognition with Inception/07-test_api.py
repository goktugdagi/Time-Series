import requests # HTTP client for calling the FastAPI service
import numpy as np # Load saved arrays

API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_INDEX = 2

# If your API applies scalers internally, send RAW data.
# If your API expects already scaled input, set USE_RAW = False.
USE_RAW  = True

X_TEST_PATH = "X_test_raw.npy" if USE_RAW else "X_test.npy"

def main():
    # Load test dataset
    try:
        X_test = np.load(X_TEST_PATH) # Expected shape: (N, 128, 9)
    except FileNotFoundError:
        print(f"Could not find '{X_TEST_PATH}'.")
        print("If you don't have raw arrays, set USE_RAW = False to use X_test.npy")
        return
    
    # Basic shape check
    if X_test.ndim !=3 or X_test.shape[1] != 128 or X_test.shape[2] != 9:
        print(f"Unexpected shape: {X_test.shape}. Expected (N, 128, 9).")
        return
    
    # Pick a sample
    if not (0 <= SAMPLE_INDEX < X_test.shape[0]):
        print(f"SAMPLE_INDEX out of range. Must be in [0, {X_test.shape[0]-1}].")
        return
    
    sample = X_test[SAMPLE_INDEX] # (128, 9)

    # Build JSON payload
    payload = {
        "sequence": sample.tolist(), # FastAPI expects list[list[float]] with shape (128, 9)
        "top_k": 3
    }

    # Send POST Request
    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("\n=== PREDICTION RESULT ===")
            print(f"Predicted label : {result.get('predicted_label')}")
            print(f"Predicted class : {result.get('predicted_class')}")
            print(f"Confidence      : {result.get('confidence')}")
            print("\nTop-K:")
            for item in result.get("top_k", []):
                print(f"  - {item['label']} (class={item['class']}): prob={item['prob']:.6f}")
        
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("The API server is not running. Start FastAPI with:")
        print("        uvicorn api:app --host 127.0.0.1 --port 8000 --reload")
    except requests.exceptions.Timeout:
        print("Request timed out. The server may be slow or unresponsive.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

