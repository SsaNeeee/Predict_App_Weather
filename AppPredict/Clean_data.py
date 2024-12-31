import pandas as pd

# Read the CSV file
file_path = 'AppPredict/Weather HCMC2020To8-2024 Full.csv'  # Replace with the correct path to the data file
try:
   data = pd.read_csv(file_path, encoding='latin1')
except UnicodeDecodeError:
   data  = pd.read_csv(file_path, encoding='ISO-8859-1')

# Drop irrelevant columns 'Unnamed: 21' and 'Mô tả'
irrelevant_columns = ['Unnamed: 21']  # Add other columns if not relevant
for col in irrelevant_columns:
   if col in data.columns:
       data.drop(columns=[col], inplace=True)

# Fill missing values in the 'Mua/KhngMua' column with 'no_rain'
data['Mua/KhngMua'].fillna('no_rain', inplace=True)

# Step 3: Convert columns to numeric where applicable
numeric_columns = [col for col in data.columns if col not in [
   'Ngày', 'TinhTrang', 'Mua/KhngMua', 'MoTa'
]]
for col in numeric_columns:
   data[col] = pd.to_numeric(data[col], errors='coerce')

# Save the cleaned data to a new file
data.to_csv('AppPredict/Cleaned_Weather_HCMC.csv', index=False, encoding='utf-8-sig')
