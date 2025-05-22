import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps
import io

# Load model and scaler
@st.cache_resource
def load_model():
    with open("logistic_mnist_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Title
st.title("MNIST Digit Classifier (Logistic Regression)")

# File uploader
uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("L")  
    image = ImageOps.invert(image)                  
    image = image.resize((28, 28))                  

    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Preprocess image
    img_array = np.array(image).astype(np.float64).reshape(1, -1)
    img_scaled = scaler.transform(img_array)

    # Prediction
    prediction = model.predict(img_scaled)[0]
    st.markdown(f"### 🔢 Predicted Digit: `{prediction}`")
