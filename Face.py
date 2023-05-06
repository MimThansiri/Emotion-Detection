import cv2
import numpy as np
# import tensorflow as tf
from keras.models import load_model

face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_detection_model = load_model('emotion_detection_model.h5')

emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        face_roi = cv2.resize(face_roi, (48, 48))

        face_roi = np.reshape(face_roi, [1, face_roi.shape[0], face_roi.shape[1], 1])

        face_roi = face_roi / 255.0

        emotion_label = np.argmax(emotion_detection_model.predict(face_roi), axis=1)[0]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion_labels[emotion_label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()