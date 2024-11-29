import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Laptop Price Predictor",  
    page_icon="💻",  
    layout="centered",  
    initial_sidebar_state="collapsed"  
)


# Load model and preprocessed data
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop', value=1.5, min_value=0.5, max_value=5.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

image_url = "https://c4.wallpaperflare.com/wallpaper/689/847/592/apple-wallpaper-preview.jpg"  # Replace with actual hosted image
page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        text-align: center;
    }}
    </style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

if st.button('Predict Price'):
    # Encode categorical features
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    # Query array creation
    try:
        query = np.array([[
            company, type, ram, weight, touchscreen, ips, ppi,
            cpu, hdd, ssd, gpu, os
        ]], dtype=object)  # Use object dtype for mixed data types
    except Exception as e:
        st.error(f"Error creating input array: {str(e)}")
        st.stop()

    # Ensure feature order matches model training
    try:
        # Model prediction
        predicted_price = np.exp(pipe.predict(query)[0])
        st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
