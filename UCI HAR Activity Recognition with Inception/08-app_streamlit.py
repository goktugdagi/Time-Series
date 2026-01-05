import json
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="UCI HAR Activity Recognition",
    page_icon="🧭",
    layout="wide"
)

# -------------------------
# Constants
# -------------------------
SEQ_LEN = 128
N_CHANNELS = 9

ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

DEFAULT_API_BASE = "http://127.0.0.1:8000"

# -------------------------
# Helpers
# -------------------------
def health_check(api_base: str):
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        if r.status_code == 200:
            return True, r.json()
        return False, {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return False, {"error": str(e)}

def validate_window(window):
    """Validate window is [128][9] list-like."""
    if not isinstance(window, list) or len(window) != SEQ_LEN:
        return False, f"Expected outer list length {SEQ_LEN}."
    for i, row in enumerate(window):
        if not isinstance(row, list) or len(row) != N_CHANNELS:
            return False, f"Row {i} must be a list of length {N_CHANNELS}."
        for v in row:
            if not isinstance(v, (int, float)):
                return False, f"All values must be numeric. Found {type(v)}."
    return True, ""

def call_predict(api_base: str, window_128x9, top_k: int):
    payload = {"sequence": window_128x9, "top_k": int(top_k)}
    r = requests.post(f"{api_base}/predict", json=payload, timeout=30)
    return r

def plot_probabilities(prob_dict):
    labels = list(prob_dict.keys())
    values = [prob_dict[k] for k in labels]

    fig = plt.figure(figsize=(8, 3.5))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Probability")
    plt.title("Class Probabilities")
    plt.tight_layout()
    st.pyplot(fig)

def topk_table(topk_list):
    rows = []
    for item in topk_list:
        rows.append(
            {"Class ID": item.get("class"), "Label": item.get("label"), "Probability": item.get("prob")}
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

def preview_window(window_128x9, n_rows=5):
    """Show a small preview of the window for sanity-check."""
    arr = np.asarray(window_128x9, dtype=np.float32)
    st.write(f"Window shape: {arr.shape}")
    st.caption("Preview (first rows):")
    st.dataframe(arr[:n_rows], use_container_width=True)

# -------------------------
# Session state init
# -------------------------
if "window" not in st.session_state:
    st.session_state["window"] = None
if "result" not in st.session_state:
    st.session_state["result"] = None

# -------------------------
# Sidebar - Controls
# -------------------------
with st.sidebar:
    st.header("Settings")

    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    top_k = st.slider("Top-K", min_value=1, max_value=6, value=3, step=1)

    st.divider()
    st.subheader("Connection")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        do_health = st.button("Health Check")
    with col_h2:
        clear_state = st.button("Clear UI State")

    if clear_state:
        st.session_state["window"] = None
        st.session_state["result"] = None
        st.success("UI state cleared.")

    if do_health:
        ok, info = health_check(api_base)
        if ok:
            st.success("API is healthy.")
            st.json(info)
        else:
            st.error("API is not reachable or unhealthy.")
            st.json(info)

    st.divider()
    st.subheader("Input Source")

    # Focus on demo inputs
    mode = st.radio("Choose input method", ["Sample (.npy)", "Upload .npy"], index=0)
    sample_index = st.number_input("Sample index", min_value=0, value=3, step=1)

    # Explicit data type to prevent double scaling confusion
    data_kind = st.radio(
        "Data sent to API is",
        ["RAW (recommended if API scales internally)", "SCALED (only if API does NOT scale)"],
        index=0
    )

    st.caption(
        "Guidance:\n"
        "- If your FastAPI applies scalers.pkl inside /predict, send RAW windows.\n"
        "- If your FastAPI expects already scaled windows, send SCALED."
    )

# -------------------------
# Main - Layout
# -------------------------
st.title("UCI HAR Activity Recognition")
st.caption(
    "Professional demo UI for a FastAPI-served Keras Inception1D model. "
    "Input: 128 time steps × 9 channels."
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("1) Load / Provide Input")
    st.write("Provide a single inertial window shaped **[128][9]**.")

    if mode == "Sample (.npy)":
        # Select correct file based on RAW/SCALED selection
        use_raw = data_kind.startswith("RAW")
        path = "X_test_raw.npy" if use_raw else "X_test.npy"

        st.info("Loads a local dataset file and selects one window by index.")
        st.code(f"Loading: {path}\nUsing SAMPLE_INDEX: {int(sample_index)}", language="text")

        if st.button("Load Sample Window"):
            try:
                X = np.load(path)  # expected (N, 128, 9)
                idx = int(sample_index)

                if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, N_CHANNELS):
                    st.error(f"Unexpected shape in {path}: {X.shape}. Expected (N, 128, 9).")
                elif idx < 0 or idx >= X.shape[0]:
                    st.error(f"Sample index out of range. Must be 0..{X.shape[0]-1}")
                else:
                    window = X[idx].tolist()
                    ok, msg = validate_window(window)
                    if not ok:
                        st.error(f"Loaded window is invalid: {msg}")
                    else:
                        st.session_state["window"] = window
                        st.session_state["result"] = None
                        st.success("Sample window loaded and ready.")
                        preview_window(window)

            except FileNotFoundError:
                st.error(f"File not found: {path}. Run preprocessing first.")
            except Exception as e:
                st.error(f"Failed to load sample: {e}")

    elif mode == "Upload .npy":
        st.info("Upload a NumPy array: either (128,9) or (N,128,9).")
        up = st.file_uploader("Upload .npy", type=["npy"])

        if up is not None:
            try:
                arr = np.load(up, allow_pickle=False)
                st.write(f"Uploaded shape: {arr.shape}")

                # Accept (128,9)
                if arr.ndim == 2 and arr.shape == (SEQ_LEN, N_CHANNELS):
                    window = arr.tolist()

                # Accept (N,128,9)
                elif arr.ndim == 3 and arr.shape[1:] == (SEQ_LEN, N_CHANNELS):
                    idx = int(sample_index)
                    if idx < 0 or idx >= arr.shape[0]:
                        st.error(f"Sample index out of range. Must be 0..{arr.shape[0]-1}")
                        window = None
                    else:
                        window = arr[idx].tolist()
                else:
                    st.error("Unsupported shape. Expected (128,9) or (N,128,9).")
                    window = None

                if window is not None:
                    ok, msg = validate_window(window)
                    if not ok:
                        st.error(f"Uploaded window is invalid: {msg}")
                    else:
                        st.session_state["window"] = window
                        st.session_state["result"] = None
                        st.success("Window loaded from upload and ready.")
                        preview_window(window)

            except Exception as e:
                st.error(f"Failed to read .npy: {e}")

    st.divider()

    st.subheader("2) Predict")
    st.write("Calls FastAPI `/predict` and displays top-k results and probability distribution.")

    # Preflight health check before prediction (optional but professional)
    preflight = st.checkbox("Run health check before predicting", value=True)

    predict_clicked = st.button("Predict Activity", type="primary")

    if predict_clicked:
        if st.session_state["window"] is None:
            st.warning("No valid window available. Load a sample or upload a file first.")
        else:
            ok, msg = validate_window(st.session_state["window"])
            if not ok:
                st.error(f"Stored input is invalid: {msg}")
            else:
                # Preflight
                if preflight:
                    ok_h, info = health_check(api_base)
                    if not ok_h:
                        st.error("API health check failed. Prediction cancelled.")
                        st.json(info)
                        st.stop()

                try:
                    resp = call_predict(api_base, st.session_state["window"], top_k=top_k)

                    if resp.status_code == 200:
                        st.session_state["result"] = resp.json()
                        st.success("Prediction completed.")
                    else:
                        st.error(f"API Error: {resp.status_code}")
                        st.code(resp.text, language="json")

                except requests.exceptions.ConnectionError:
                    st.error("Connection error. Is FastAPI running?")
                    st.caption("Run: uvicorn api:app --host 127.0.0.1 --port 8000 --reload")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The API may be slow or unresponsive.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    st.divider()

    # Advanced manual JSON (optional, not primary UX)
    with st.expander("Advanced: Manual JSON input (128x9)"):
        st.caption("Paste a JSON array shaped [128][9]. This is intended for advanced/testing use.")
        txt = st.text_area("Window JSON", height=220, placeholder="Paste JSON here...")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Validate JSON (Advanced)"):
                try:
                    parsed = json.loads(txt)
                    ok, msg = validate_window(parsed)
                    if ok:
                        st.session_state["window"] = parsed
                        st.session_state["result"] = None
                        st.success("Valid input. Ready to predict.")
                        preview_window(parsed)
                    else:
                        st.error(f"Invalid shape/content: {msg}")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

        with col_b:
            if st.button("Load Example JSON Template"):
                template = [[0.0] * N_CHANNELS for _ in range(SEQ_LEN)]
                st.text_area("Template (copy & edit)", value=json.dumps(template), height=220)

with right:
    st.subheader("Result")

    if st.session_state["result"] is None:
        st.write("Run a prediction to view results.")
        st.caption("Tip: Load a sample or upload an array, then click Predict.")
    else:
        result = st.session_state["result"]

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted Label", result.get("predicted_label", "-"))
        k2.metric("Class ID", str(result.get("predicted_class", "-")))
        k3.metric("Confidence", str(result.get("confidence", "-")))

        st.divider()

        st.markdown("**Top-K**")
        topk_table(result.get("top_k", []))

        st.divider()

        st.markdown("**Probabilities**")
        probs = result.get("probabilities", {})
        if isinstance(probs, dict) and len(probs) > 0:
            plot_probabilities(probs)
        else:
            st.write("No probability dictionary found in response.")

        st.divider()

        st.markdown("**Raw JSON Response**")
        st.json(result)
