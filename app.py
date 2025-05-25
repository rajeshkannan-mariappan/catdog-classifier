import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Choose a cat or dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 4:
        # Remove alpha channel if present
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"The image is a **Dog** 🐶 (Confidence: {prediction:.2f})")
    else:
        st.success(f"The image is a **Cat** 🐱 (Confidence: {(1 - prediction):.2f})")
