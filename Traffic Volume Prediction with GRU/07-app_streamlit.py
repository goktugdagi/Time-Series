# streamlit run 07-app_streamlit.py  # Command to run the Streamlit app

import streamlit as st  # Import streamlit to build a web UI
import requests  # Import requests to call the FastAPI backend

# Title and description
st.title("Traffic Volume Prediction")  # Set app title
st.markdown("""
    Enter the last 24 hours of temperature, rain, snow, cloud coverage, hour, day of week, and month values.
    Default values are provided for each hour. You can modify them if needed. Click **Predict** to see the result.
        """)  # Provide user guidance text

# Prepare 24-hour input sequence
# Each hour: [temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]
# User inputs are collected into the sequence_input list.
sequence_input = []  # Will be a 24 x 7 nested list

# Start a form block (inputs are submitted together)
with st.form("manual_input_form"):  # Create a form
    st.subheader("24-Hour Input Data")  # Form subtitle

    # Collect 7 features for each of 24 hours
    for i in range(24):  # Loop for 24 hours
        st.markdown(f"**Hour {i}**")  # Hour label

        # Create a 7-column grid layout
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)  # Layout columns

        # Get user inputs for each feature
        temp = col1.number_input("Temp (K)", value=290.0, min_value=270.0, max_value=330.0, step=0.1, key=f"temp_{i}")  # Temperature
        rain = col2.number_input("Rain (mm)", value=0.0, min_value=0.0, max_value=100.0, step=0.1, key=f"rain_{i}")  # Rain
        snow = col3.number_input("Snow (mm)", value=0.0, min_value=0.0, max_value=100.0, step=0.1, key=f"snow_{i}")  # Snow
        clouds = col4.slider("Clouds (%)", value=40, min_value=0, max_value=100, key=f"clouds_{i}")  # Cloud coverage
        hour = i  # Hour feature (automatically set 0..23)
        dayofweek = col6.selectbox("Day of Week", options=list(range(7)), index=1, key=f"day_{i}")  # Day-of-week selection
        month = col7.selectbox("Month", options=list(range(1, 13)), index=1, key=f"month_{i}")  # Month selection

        # Append the 7-feature vector for this hour
        sequence_input.append([temp, rain, snow, clouds, hour, dayofweek, month])  # Add to the sequence list

    # Submit button
    submitted = st.form_submit_button("Predict")  # Submit inputs

    # When submitted, call FastAPI and display the prediction
    if submitted:  # Check if form is submitted
        try:  # Catch connection and request errors
            API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint URL

            response = requests.post(API_URL, json={"sequence": sequence_input})  # Send POST request

            if response.status_code == 200:  # If success
                result = response.json()  # Parse JSON response
                prediction = result["predicted_traffic_volume"]  # Extract predicted value

                # Show result to the user
                st.success(f"Predicted Traffic Volume: {int(prediction)} vehicles/hour")  # Display prediction
            else:
                st.error(f"API request failed. Code: {response.status_code}")  # Display HTTP status
                st.text(response.text)  # Display response text
        except Exception as e:
            st.error("Could not connect to the FastAPI server.")  # Connection error message
            st.exception(e)  # Show exception details
