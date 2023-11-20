# Real-time Emotion Detection using OpenCV and CNN

## Overview

This project demonstrates real-time emotion detection using OpenCV and a Convolutional Neural Network (CNN) model. The model predicts emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral from a live video feed captured by your camera.

## Prerequisites

- Python 3.x
- OpenCV
- TensorFlow
- Keras

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/zargon01/Real-Time-Emotion-Detector
    cd Real-Time-Emotion-Detector
    ```

2. **Install the required dependencies:**

    ```bash
    pip install opencv-python tensorflow keras
    ```

3. **Download the pre-trained CNN model:**

   [Download Model](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

4. **Run the emotion detection script:**

    ```bash
    python emotion_detection.py
    ```

## Usage

1. Ensure your camera is connected and accessible.
2. Run the script, and the application will open a window displaying real-time video feed with emotion predictions.

## Customization

- **Model:** You can experiment with different pre-trained models for emotion detection. Ensure to update the model file path in the script.

- **Fine-tuning:** If needed, fine-tune the model on a dataset more representative of your specific use case.

- **Parameters:** Adjust face detection parameters in the script to improve face recognition.


## License

This project is licensed under the [MIT License](LICENSE).
