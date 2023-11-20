import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained CNN model for emotion detection
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open the camera
cap = cv2.VideoCapture(0)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    try:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for emotion detection
            roi = gray[y:y + h, x:x + w]

            # Resize the ROI to match the input size of the model
            roi = cv2.resize(roi, (64, 64))

            # Normalize the pixel values
            roi = roi / 255.0

            # Reshape the image to match the model's expected shape
            roi = np.reshape(roi, (1, 64, 64, 1))

            # Make a prediction using the model
            prediction = model.predict(roi)

            # Get the index of the predicted emotion
            emotion_index = np.argmax(prediction)

            # Get the corresponding emotion label
            emotion = emotion_labels[emotion_index]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the emotion on the frame with confidence score
            emotion_text = f'{emotion} ({prediction[0][emotion_index]:.2f})'
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
