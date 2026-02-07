import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("🖥️ Laptop Price Predictor")

# Define all options
company_values = ['Asus', 'Dell', 'HP', 'Lenovo', 'Apple', 'MSI', 'Acer', 'Vaio', 'Razer', 'Medion']
type_values = ['Ultrabook', 'Notebook', '2 in 1 Convertible', 'Gaming', 'Netbook']
cpu_values = ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Other Intel Processor', 'AMD Processor']
gpu_values = ['None', 'Nvidia', 'Intel', 'AMD']
os_values = ['Windows', 'mac', 'Others / linux /no OS']

# Create 2-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Specs")
    company = st.selectbox('Brand', company_values)
    type_laptop = st.selectbox('Type', type_values)
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])

with col2:
    st.subheader("Advanced Specs")
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU', cpu_values)
    hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', gpu_values)
    os_type = st.selectbox('OS', os_values)

# Prediction
if st.button('🔮 Predict Price', use_container_width=True):
    try:
        # Convert Yes/No to 1/0
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0
        
        # Calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
        
        # Create query
        query = np.array([company, type_laptop, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os_type], dtype=object)
        query = query.reshape(1, 12)
        
        # Make prediction
        log_price = pipe.predict(query)[0]
        predicted_price = np.exp(log_price)
        
        st.success(f" **Predicted Price: ₹{predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
