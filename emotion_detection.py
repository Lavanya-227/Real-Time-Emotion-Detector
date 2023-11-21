import cv2
from keras.models import load_model
import numpy as np

model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]

            roi = cv2.resize(roi, (64, 64))

            roi = roi / 255.0

            roi = np.reshape(roi, (1, 64, 64, 1))

            prediction = model.predict(roi)

            emotion_index = np.argmax(prediction)

            emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            emotion_text = f'{emotion} ({prediction[0][emotion_index]:.2f})'
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

cap.release()
cv2.destroyAllWindows()
