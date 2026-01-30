import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("👤 Face Detection AI App")
st.write("Upload an image and detect faces automatically.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

    st.image(img_array, caption="Detected Faces", use_column_width=True)
    st.success(f"✅ Faces detected: {len(faces)}")
