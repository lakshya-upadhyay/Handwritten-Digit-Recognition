# Handwritten Digit Recognition with CNN

This project is a complete end-to-end application for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) and a Streamlit web interface. It uses the classic MNIST dataset for training.

## Features
-   **Drawing Canvas**: Interactive web-based canvas to draw digits.
-   **Real-time Recognition**: Instant prediction of the drawn digit.
-   **CNN Model**: A robust architecture built with TensorFlow/Keras.
-   **Data Visualization**: Displays probability distribution (confidence) for each digit.
-   **Auto-Training**: Automatically trains the model on first launch if no pre-trained model is found.

## Prerequisites
-   Python 3.9, 3.10, or 3.11 installed.
-   Internet connection (to download the MNIST dataset and libraries on first run).

## Installation and Setup

### Automatic Launch (One-Click)
1.  **Windows**: Double-click `run_app.bat`.
2.  **macOS/Linux**: Run `bash run_app.sh` in your terminal.

The script will automatically create a virtual environment, install requirements, train the model, and launch the browser.

### Manual Installation
If you prefer to run steps manually:

1.  Create a virtual environment:
    `python -m venv venv`

2.  Activate the environment:
    -   Windows: `venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`

3.  Install dependencies:
    `pip install -r requirements.txt`

4.  Train the model (optional if you want to run it separately):
    `python train.py`

5.  Run the application:
    `streamlit run app.py`

## Project Structure
-   `app.py`: The main Streamlit web application.
-   `model_architecture.py`: Definition of the CNN layers and compilation settings.
-   `train.py`: Logic for loading MNIST data and training the model.
-   `utils.py`: Image processing functions to convert canvas drawings to model input.
-   `requirements.txt`: Python package dependencies.
-   `model.h5`: The saved trained model (generated after training).

## Troubleshooting
-   **Low Accuracy**: Digits should be drawn large and centered in the canvas for best results. Since MNIST training data is centered and normalized, drawing very small or very off-center digits may result in incorrect predictions.
-   **OpenCV Errors**: If you encounter issues with `cv2`, ensure you are using the `opencv-python-headless` package included in the requirements to avoid GUI library conflicts in server environments.
-   **Python Version**: This project is tested on Python 3.9+. Ensure your Python version is compatible.

## Usage Example
1.  Launch the app.
2.  Use your mouse or touch screen to draw a '7'.
3.  Click the "Predict Digit" button.
4.  The application will highlight '7' as the result and show a bar chart where the bar for 7 is significantly higher than others.
