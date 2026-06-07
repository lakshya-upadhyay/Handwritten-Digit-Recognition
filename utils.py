import cv2
import numpy as np

def preprocess_image(canvas_data):
    """
    Converts canvas RGBA image data to MNIST compatible format.
    
    Steps:
    1. Convert RGBA to Grayscale
    2. Resize to 28x28
    3. Normalize pixel values to 0-1
    4. Reshape for model input (1, 28, 28, 1)
    """
    if canvas_data is None:
        return None
        
    # canvas_data is a numpy array (H, W, 4) from streamlit-drawable-canvas
    # Convert RGBA to grayscale
    img = cv2.cvtColor(canvas_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    
    # Check if canvas is empty (mostly black)
    if np.max(img) < 10:
        return None

    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to 0-1
    img = img.astype('float32') / 255.0
    
    # Reshape for the model: (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=(0, -1))
    
    return img
