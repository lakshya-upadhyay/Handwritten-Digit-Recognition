import streamlit as st
import os
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from utils import preprocess_image
from train import train_and_save_model

# Page configuration
st.set_page_config(page_title="AI Digit Recognizer", layout="centered")

st.title("🖊️ Handwritten Digit Recognition")
st.markdown("""
Draw a digit from **0 to 9** in the box below. 
The CNN model will try to predict what you've drawn!
""")

# Ensure model exists
MODEL_PATH = 'model.h5'

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found. Training version 1.0 on MNIST..."):
            train_and_save_model()
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Draw a single digit (0-9).
2. Use the 'Predict' button.
3. Use the 'Trash' icon to clear the canvas.
""")

# Create canvas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    predict_button = st.button('Predict Digit')
    
    if canvas_result.image_data is not None:
        processed_img = preprocess_image(canvas_result.image_data)
        
        if predict_button:
            if processed_img is not None:
                prediction = model.predict(processed_img)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                st.success(f"### Result: {predicted_digit}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Confidence chart
                st.bar_chart(prediction[0])
            else:
                st.warning("Please draw something on the canvas first!")

# Display small preview of what the model sees
if canvas_result.image_data is not None:
    with st.expander("Show AI Input View"):
        raw_img = preprocess_image(canvas_result.image_data)
        if raw_img is not None:
            # Reshape back to 2d for display
            preview = raw_img.reshape(28, 28)
            st.image(preview, width=100, caption="28x28 Rescaled Input")
