import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Skin Disease Classification",
    layout="centered"
)

st.title("🩺 Skin Disease Classification")
st.write("Upload a skin image to get the predicted disease")

# ----------------------------------
# Model file check
# ----------------------------------
MODEL_PATH = "dermnet_model.keras"
CLASS_INDICES_PATH = "class_indices.json"  # Optional: saved from training

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

# ----------------------------------
# Load model safely
# ----------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("❌ Failed to load model")
        st.exception(e)
        st.stop()

model = load_model()

# ----------------------------------
# Real class names (replace generic names)
# ----------------------------------
REAL_CLASSES = [
    "Acne and Rosacea Photos",
    "Eczema Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Warts Molluscum and other Viral Infections",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections"
]

# Load class names dynamically if JSON exists, otherwise use real class names
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_names_dict = json.load(f)
    # Ensure proper order: sort by value
    class_names = [k for k,v in sorted(class_names_dict.items(), key=lambda item: item[1])]
else:
    class_names = REAL_CLASSES
    st.info("Using real class names as fallback")

# ----------------------------------
# Image Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])

        # Safety check
        if predicted_index >= len(class_names):
            st.error("❌ Class index mismatch between model output and class names")
            st.stop()

        predicted_class = class_names[predicted_index]

        # Display results
        st.success(f"🧠 Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

        # Show all probabilities
        st.subheader("📊 Class Probabilities")
        for cls, prob in zip(class_names, predictions[0]):
            st.write(f"{cls}: {prob * 100:.2f}%")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("⚠️ This app is for educational purposes only.")
