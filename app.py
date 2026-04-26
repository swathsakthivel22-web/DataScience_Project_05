import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Face Mask Detection 😷")

# Load model
model = load_model("face_mask_model.h5")

IMG_SIZE = 128

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    # Prediction
    prediction = model.predict(img_resized)

    if prediction < 0.5:
        st.success("😷 Mask Detected")
    else:
        st.error("❌ No Mask Detected")
