import streamlit as st  # Streamlit UI framework
import numpy as np  # Load saved NumPy arrays for example input
import requests  # Call FastAPI endpoint from the UI

# Title and description  # Basic UI header
st.title("NO2 Prediction Interface")  # App title
st.markdown("This application estimates NO2 levels using environmental data from the past 72 hours.")  # App description

if st.button("🧪 Make predictions using sample data."):  # Button to run an example prediction
    try:
        X_test = np.load("X_test.npy")  # Load test windows
        example = X_test[3].tolist()  # Pick an example window and convert to lists

        # Send request to API  # Call /predict endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json={"sequence": example})  # POST JSON payload

        if response.status_code == 200:  # Success response
            result = response.json()  # Parse JSON
            st.success(f"Estimated NO2: {result['estimated_NO2']}")  # Show predicted NO2
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")  # Show API error response
    except FileNotFoundError:
        st.error("The file x_test.npy was not found..")  # Missing local file
    except requests.exceptions.ConnectionError:
        st.error("API connection error")  # API not reachable

# UI for manual input  # Advanced users can paste a custom 72x7 JSON-like array
with st.expander("Create your own inputs."):  # Collapsible section
    st.markdown("Data format: 72 time steps, 7 features per step.")  # Input format guidance

    custom_input = st.text_area("Enter the data here in JSON format. Example: [[0.2, 0.3, ...], [...], [...]]")  # Text area

    if st.button("Estimate data manually"):  # Button to submit manual input
        try:
            import json  # Safe JSON parsing
            parsed = json.loads(custom_input)  # Parse input safely (expects valid JSON)

            if len(parsed) == 72 and len(parsed[0]) == 7:  # Basic shape check
                response = requests.post("http://127.0.0.1:8000/predict", json={"sequence": parsed})  # Call API

                if response.status_code == 200:  # Success
                    result = response.json()  # Parse JSON result
                    st.success(f"Estimated NO2: {result['estimated_NO2']}")  # Show prediction
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")  # Show API error details
            else:
                st.warning("Please enter the data with 72 time steps and 7 features.")  # Shape warning
        except Exception as e:
            st.error(f"Input could not be parsed. {e}")  # Parsing error display
