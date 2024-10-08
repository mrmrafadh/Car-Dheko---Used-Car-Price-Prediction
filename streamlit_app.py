import streamlit as st
import pandas as pd
import numpy as np
import pickle


df1 = pd.read_csv('csv/ready_for_ml.csv')
list_bt = list(df1['bt'].unique())
list_oem = list(df1['oem'].unique())
list_ft = list(df1['ft'].unique())
list_city = list(df1['City'].unique())

# Load the saved encoders, scaler, and trained model from pickle
with open('test2_scale_encoder_.pkl', 'rb') as file:
    one_hot_encoder, label_encoder, scaler = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Car Dheko - Used Car Price Prediction")
# Sidebar inputs for the user to input the car features
st.sidebar.header("Car Dheko - Used Car Price Prediction")
st.sidebar.header("By Rafadh Rafeek")
st.sidebar.header("Enter Car Details")

bt = st.sidebar.selectbox("Body Type", list_bt)
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
oem = st.sidebar.selectbox("OEM", list_oem)
ft = st.sidebar.selectbox("Fuel Type", list_ft)
km = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=100)
ownerNo = st.sidebar.number_input("Number of Owners", min_value=1, max_value=10, step=1)
Seats = st.sidebar.number_input("Seats", min_value=2, max_value=10, step=1)
year_of_manufacture = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2024, step=1)
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=1.0, max_value=50.0, step=0.1)
engine = st.sidebar.number_input("Engine Capacity (cc)", min_value=500.0, max_value=5000.0, step=0.1)
city = st.sidebar.selectbox("City", list_city)

# Predict button
if st.sidebar.button("Predict Price"):
    # 1. Create a DataFrame with the user inputs
    new_data = pd.DataFrame([{
        'bt': bt,
        'transmission': transmission,
        'oem': oem,
        'ft': ft,
        'km': km,
        'ownerNo': ownerNo,
        'Seats': Seats,
        'Year of Manufacture': year_of_manufacture,
        'Mileage': mileage,
        'Engine': engine,
        'City': city
    }])

    # 2. One-hot encode the categorical columns
    one_hot_columns = ['bt', 'transmission', 'ft', 'City']
    new_data_one_hot_encoded = one_hot_encoder.transform(new_data[one_hot_columns])
    one_hot_encoded_columns = one_hot_encoder.get_feature_names_out(one_hot_columns)
    new_data_one_hot_encoded_df = pd.DataFrame(new_data_one_hot_encoded, columns=one_hot_encoded_columns)

    # 3. Label encode the 'oem' column
    new_data['oem_encoded'] = label_encoder.transform(new_data['oem'])

    # 4. Drop the original categorical columns and combine features
    new_data_numeric = new_data.drop(one_hot_columns + ['oem'], axis=1)
    new_data_final = pd.concat([new_data_numeric.reset_index(drop=True), new_data_one_hot_encoded_df.reset_index(drop=True)], axis=1)

    # 5. Scale the numeric features (km, Year of Manufacture, Engine)
    numeric_columns = ['km', 'Year of Manufacture', 'Engine']
    new_data_final[numeric_columns] = scaler.transform(new_data_final[numeric_columns])

    # 6. Predict the car price
    predicted_price = model.predict(new_data_final)

    # 7. Display the prediction result
    st.write(f"### Predicted Price for the car: ₹{predicted_price[0]:,.2f}")

