import os
import tensorflow as tf
from model_architecture import create_digit_model

def train_and_save_model():
    """
    Trains the CNN model on the MNIST dataset and saves it to a file.
    """
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    print("Building model...")
    model = create_digit_model()

    print("Starting training (5 epochs)...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    print("Saving model to model.h5...")
    model.save('model.h5')
    print("Training complete!")

if __name__ == "__main__":
    train_and_save_model()
