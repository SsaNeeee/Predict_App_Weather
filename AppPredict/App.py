import streamlit as st
import pandas as pd
import joblib

# Load the models
rf_model_path = 'AppPredict/random_forest_model.pkl'
logistic_model_path = 'AppPredict/logistic_regression_model.pkl'

rf_model = joblib.load(rf_model_path)
logistic_model = joblib.load(logistic_model_path)

# Define the input form
st.title("Dự báo thời tiết: Mưa/Không Mưa")

# User selects the model
model_choice = st.selectbox(
    "Chọn mô hình bạn muốn sử dụng:",
    ["Random Forest", "Logistic Regression"]
)

# Input fields for weather data
st.header("Nhập dữ liệu thời tiết của ngày cần dự báo")
temperature = st.number_input("Nhiệt độ trung bình (°C):", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Độ ẩm (%):", min_value=0.0, max_value=100.0, step=1.0)
wind_speed = st.number_input("Vận tốc gió (km/h):", min_value=0.0, max_value=150.0, step=0.1)

# Predict button
if st.button("Dự báo"):
    # Prepare input data
    input_data = pd.DataFrame([{
        "NhietDoTB": temperature,
        "DoAm": humidity,
        "VanTocGio": wind_speed
    }])

    # Choose the selected model
    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = logistic_model

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Map prediction to readable output
    result = "Mưa" if prediction == 1 else "Không Mưa"

    # Display result
    st.success(f"Kết quả dự báo: {result}")
