import tensorflow as tf
from tensorflow.keras import layers, models

def create_digit_model():
    """
    Defines a Convolutional Neural Network (CNN) architecture optimized for 
    the MNIST handwritten digit dataset (28x28 grayscale images).
    """
    model = models.Sequential([
        # First Convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Regularization to prevent overfitting
        layers.Dense(10, activation='softmax') # 10 output classes for digits 0-9
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
