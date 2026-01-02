import streamlit as st
import requests
import random

st.title("Weather Forecast API")
st.markdown(""" 
Enter the last 24 hours of humidity, wind_speed_km_h, pressure_millibars, visibility_km, and apparent_temperature_c values.
Default values are provided for each hour. You can modify them if needed. Click **Predict** to see the result.
""")

API_URL = "http://127.0.0.1:8000/predict"

# IMPORTANT: Create the sequence INSIDE the form submission flow
with st.form("manual_input_form"):
    st.subheader("24-Hour Input Data")

    sequence_input = []  # 24 x 5

    for i in range(24):
        st.markdown(f"**Hour {i}**")
        col1, col2, col3, col4, col5 = st.columns(5)

        # Use realistic defaults and vary them per hour (breaks constant inputs)
        humidity_default = float(random.randint(30, 90))
        wind_default = float(random.uniform(0, 30))
        pressure_default = float(random.uniform(980, 1050))
        visibility_default = float(random.uniform(1, 20))
        apparent_default = float(random.uniform(-5, 35))

        humidity = col1.number_input(
            "humidity",
            value=humidity_default, min_value=0.0, max_value=100.0, step=1.0,
            key=f"humidity_{i}"
        )
        wind_speed_km_h = col2.number_input(
            "wind_speed_km_h",
            value=wind_default, min_value=0.0, max_value=100.0, step=0.5,
            key=f"wind_speed_km_h_{i}"
        )
        pressure_millibars = col3.number_input(
            "pressure_millibars",
            value=pressure_default, min_value=800.0, max_value=1100.0, step=0.5,
            key=f"pressure_millibars_{i}"
        )
        visibility_km = col4.number_input(
            "visibility_km",
            value=visibility_default, min_value=0.0, max_value=50.0, step=0.5,
            key=f"visibility_km_{i}"
        )
        apparent_temperature_c = col5.number_input(
            "apparent_temperature_c",
            value=apparent_default, min_value=-50.0, max_value=60.0, step=0.5,
            key=f"apparent_temperature_c_{i}"
        )

        sequence_input.append([
            float(humidity),
            float(wind_speed_km_h),
            float(pressure_millibars),
            float(visibility_km),
            float(apparent_temperature_c),
        ])

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(API_URL, json={"sequence": sequence_input}, timeout=30)

        if response.status_code == 200:
            result = response.json()
            prediction = float(result["predicted_temperature_c"])
            st.success(f"Predicted temperature_c: {prediction:.2f} °C")
        else:
            st.error(f"API request failed. Code: {response.status_code}")
            st.text(response.text)

    except Exception as e:
        st.error("Could not connect to the FastAPI server.")
        st.exception(e)
