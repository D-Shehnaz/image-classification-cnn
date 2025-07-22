import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

st.title("Image Classification with CNN")

model = load_model('../models/cnn_model.h5')
with open('../models/labels.txt') as f:
    labels = f.read().splitlines()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    prediction = model.predict(image)
    pred_label = labels[np.argmax(prediction)]

    st.write(f"### Prediction: **{pred_label}**")
