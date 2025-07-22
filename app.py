import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Config
st.set_page_config(page_title='Face Mask Detector 😷', page_icon='😷', layout='centered')

# Load models only once
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(
        os.path.join("face_detector", "deploy.prototxt"),
        os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    )
    mask_model = load_model("mask_detector.model")
    return face_net, mask_model

face_net, mask_model = load_models()


# Utility: Apply mask detection on an image
def detect_mask_on_image(image):
    image = np.array(image.convert('RGB'))
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = image[startY:endY, startX:endX]
        if face.size == 0:
            continue 

        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, withoutMask) = mask_model.predict(face, verbose=0)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        cv2.putText(orig, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2)

    return orig


# UI Components
st.title("😷 Face Mask Detection")
option = st.sidebar.radio("Choose mode", ["Detect on Image", "Detect on Webcam (Coming Soon)"])

if option == "Detect on Image":
    st.markdown("### Upload an image (.jpg or .jpeg):")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            processed_image = detect_mask_on_image(image)
            st.image(processed_image, caption="Processed Output", use_column_width=True)

elif option == "Detect on Webcam (Coming Soon)":
    st.info("Webcam detection feature will be available soon. Stay tuned!")

