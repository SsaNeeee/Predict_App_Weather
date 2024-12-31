import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Read the data
file_path = 'Cleaned_Weather_HCMC.csv'
data = pd.read_csv(file_path)

# Select the necessary columns
features = ['NhietDoTB', 'DoAm', 'VanTocGio']
target = 'Mua/KhngMua'
data = data[features + [target]]

# Kiểm tra và xử lý NaN trong cột target
if data[target].isnull().any():
    print(f"NaN values found in '{target}': {data[target].isnull().sum()}")
    data = data.dropna(subset=[target])  # Loại bỏ các hàng chứa NaN trong cột target
    print(f"Removed rows with NaN values in '{target}'.")
# Check class distribution
class_distribution = data[target].value_counts()
print("Class Distribution:")
print(class_distribution)

# Encode the target column
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Kiểm tra và loại bỏ các lớp có ít hơn 2 mẫu
min_class_samples = 2
class_counts = data[target].value_counts()
rare_classes = class_counts[class_counts < min_class_samples].index

if len(rare_classes) > 0:
    print(f"Removing rare classes: {rare_classes}")
    data = data[~data[target].isin(rare_classes)]
# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Logistic Regression
print("\n=== Logistic Regression ===")
logistic_model = LogisticRegression(class_weight='balanced', random_state=42)

# 10-Fold Cross-Validation for Logistic Regression
logistic_cv_scores = cross_val_score(logistic_model, X, y, cv=10, scoring='accuracy')
logistic_mean_cv = np.mean(logistic_cv_scores)
print(f"10-Fold Cross-Validation Scores (Logistic Regression): {logistic_cv_scores}")
print(f"Mean Accuracy (Logistic Regression): {logistic_mean_cv:.2f}")

# Train the Logistic Regression model
logistic_model.fit(X_train, y_train)

# Save the Logistic Regression model
logistic_model_file = 'AppPredict/logistic_regression_model.pkl'
joblib.dump(logistic_model, logistic_model_file)
print(f"Logistic Regression Model saved to {logistic_model_file}")

# Evaluate Logistic Regression on the test set
y_pred_logistic = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_report = classification_report(
    y_test, y_pred_logistic, labels=np.unique(y_test), target_names=label_encoder.inverse_transform(np.unique(y_test)), output_dict=True
)

classes_in_y_test = np.unique(y_test)
logistic_f1_score = sum(
    logistic_report[label_encoder.inverse_transform([cls])[0]]['f1-score']
    for cls in classes_in_y_test
) / len(classes_in_y_test)
print(f"\nAccuracy (Logistic Regression): {logistic_accuracy:.2f}")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_logistic, labels=np.unique(y_test), target_names=label_encoder.inverse_transform(np.unique(y_test))))

# 2. Random Forest
print("\n=== Random Forest ===")
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100, max_depth=10)

# 10-Fold Cross-Validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy')
rf_mean_cv = np.mean(rf_cv_scores)
print(f"10-Fold Cross-Validation Scores (Random Forest): {rf_cv_scores}")
print(f"Mean Accuracy (Random Forest): {rf_mean_cv:.2f}")

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Save the Random Forest model
rf_model_file = 'AppPredict/random_forest_model.pkl'
joblib.dump(rf_model, rf_model_file)
print(f"Random Forest Model saved to {rf_model_file}")

# Evaluate Random Forest on the test set
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, target_names=label_encoder.inverse_transform(np.unique(y_test)), output_dict=True)
rf_f1_score = sum(
    rf_report[class_name]['f1-score']
    for class_name in rf_report.keys() if class_name != 'accuracy'
) / (len(rf_report) - 1)
print(f"\nAccuracy (Random Forest): {rf_accuracy:.2f}")
print("Classification Report (Random Forest):")
# In ra báo cáo phân loại với đúng số lớp
print(classification_report(y_test, y_pred_rf, labels=np.unique(y_test), target_names=label_encoder.classes_))

# Predictions for custom data
custom_data = pd.DataFrame([
    {'NhietDoTB': 30.0, 'DoAm': 85.0, 'VanTocGio': 10.0},  # Prediction Rain
    {'NhietDoTB': 25.0, 'DoAm': 50.0, 'VanTocGio': 5.0}   # Prediction No Rain
])

# Logistic Regression prediction
custom_predictions_logistic = logistic_model.predict(custom_data)
custom_results_logistic = custom_data.copy()
custom_results_logistic['Predicted_Logistic'] = label_encoder.inverse_transform(custom_predictions_logistic)

# Random Forest prediction
custom_predictions_rf = rf_model.predict(custom_data)
custom_results_rf = custom_data.copy()
custom_results_rf['Predicted_RF'] = label_encoder.inverse_transform(custom_predictions_rf)

# Display prediction results from both models
print("\nPredictions for custom data (Logistic Regression):")
for index, row in custom_results_logistic.iterrows():
    print(f"{row['Predicted_Logistic']}: Avg Temperature: {row['NhietDoTB']:.2f}, Humidity: {row['DoAm']:.2f}, Wind Speed: {row['VanTocGio']:.2f}")

print("\nPredictions for custom data (Random Forest):")
for index, row in custom_results_rf.iterrows():
    print(f"{row['Predicted_RF']}: Avg Temperature: {row['NhietDoTB']:.2f}, Humidity: {row['DoAm']:.2f}, Wind Speed: {row['VanTocGio']:.2f}")

# Model comparison
print("\n=== Model Comparison ===")
print(f"Logistic Regression: Accuracy = {logistic_accuracy:.2f}, F1-Score = {logistic_f1_score:.2f}, Cross-Validation Mean = {logistic_mean_cv:.2f}")
print(f"Random Forest: Accuracy = {rf_accuracy:.2f}, F1-Score = {rf_f1_score:.2f}, Cross-Validation Mean = {rf_mean_cv:.2f}")

if logistic_f1_score > rf_f1_score and logistic_mean_cv >= rf_mean_cv:
    print("\nLogistic Regression is recommended due to better overall performance.")
elif rf_f1_score > logistic_f1_score and rf_mean_cv >= logistic_mean_cv:
    print("\nRandom Forest is recommended due to better overall performance.")
else:
    print("\nBoth models have similar performance, use the simpler model like Logistic Regression if speed is prioritized.")
