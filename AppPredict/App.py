import streamlit as st
import pandas as pd
import joblib

# Function to load models with error handling
def load_model(file_path, model_name):
    try:
        model = joblib.load(file_path)
        st.success(f"Đã tải thành công mô hình: {model_name}")
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình {model_name}: {str(e)}")
        return None

# File paths to models
rf_model_path = 'AppPredict/random_forest_model.pkl'
logistic_model_path = 'AppPredict/logistic_regression_model.pkl'

# Load the models
rf_model = load_model(rf_model_path, "Random Forest")
logistic_model = load_model(logistic_model_path, "Logistic Regression")

# App Title
st.title("Dự báo thời tiết: Mưa/Không Mưa")

# User selects the model
st.header("Chọn mô hình dự báo")
model_choice = st.radio(
    "Vui lòng chọn mô hình bạn muốn sử dụng:",
    ["Chưa chọn", "Random Forest", "Logistic Regression"]
)

# Validate model selection
if model_choice == "Chưa chọn":
    st.warning("Bạn cần chọn một mô hình để tiến hành dự báo!")

# Input fields for weather data
st.header("Nhập dữ liệu thời tiết của ngày cần dự báo")
temperature = st.number_input("Nhiệt độ trung bình (°C):", value=25.0, min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Độ ẩm (%):", value=50.0, min_value=0.0, max_value=100.0, step=1.0)
wind_speed = st.number_input("Vận tốc gió (km/h):", value=10.0, min_value=0.0, max_value=150.0, step=0.1)

# Button for prediction
if st.button("Dự báo"):
    # Check for model selection
    if model_choice == "Chưa chọn":
        st.error("Vui lòng chọn mô hình để tiến hành dự báo!")
    else:
        # Ensure models are loaded
        if (model_choice == "Random Forest" and rf_model is None) or \
           (model_choice == "Logistic Regression" and logistic_model is None):
            st.error("Mô hình không được tải thành công. Vui lòng kiểm tra lại.")
        else:
            # Prepare input data
            input_data = pd.DataFrame([{
                "NhietDoTB": temperature,
                "DoAm": humidity,
                "VanTocGio": wind_speed
            }])

            # Choose the selected model
            model = rf_model if model_choice == "Random Forest" else logistic_model

            # Check prediction probabilities (if available)
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_data)
                st.write(f"Xác suất dự đoán: {prediction_proba}")
                # Adjust threshold if needed
                if prediction_proba[0][1] > 0.4:  # Thay đổi ngưỡng tại đây
                    result = "Mưa"
                else:
                    result = "Không Mưa"
            else:
                # Predict without probabilities
                prediction = model.predict(input_data)[0]
                result = "Mưa" if prediction == 1 else "Không Mưa"

            # Display result
            st.success(f"Kết quả dự báo: {result}")
