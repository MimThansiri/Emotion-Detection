import cv2
import numpy as np
from keras.models import load_model
import os
import whisper
import nltk
import text2emotion as te
from collections import Counter

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

file_path = os.path.abspath('C:/Users/HP/PycharmProjects/AI_Technology/test/KeyandPeele.mp4')  # get absolute file path

# import os
os.environ['PATH'] += ';C:\\ffmpeg\\bin'

model = whisper.load_model("base")
result = model.transcribe(file_path)
print(result['text'])

sentences = sent_tokenize(result['text'])

wduration = 53.0  # total duration of the audio or video file in seconds - Right baby end at 5:35
num_words = sum(len(sentence.split()) for sentence in sentences)
words_per_second = num_words / wduration

modifier = np.empty((len(sentences), 7)) # 2D matrix to store emotions for each sentence

with open("transcription.txt", "w") as file:
    start_time = 0.0
    for i, sentence in enumerate(sentences):
        num_words = len(sentence.split())
        end_time = start_time + num_words / words_per_second
        print(f"{sentence} ({start_time:.2f}-{end_time:.2f})")

        # Get the emotions for the sentence using text2emotion
        emotions = te.get_emotion(sentence)
        if sum(emotions.values()) == 0:
            emotions["Disgust"] = 0
            emotions["Neutral"] = 1
        else:
            emotions["Disgust"] = 0.0
            emotions["Neutral"] = 0.0

        # Print the emotions for the sentence
        emotions_counter = Counter(emotions)
        print("Emotions for the sentence: ")
        for j, (emotion, count) in enumerate(emotions_counter.items()):
            print(f"{emotion}: {count}")
            modifier[i][j] = count  # store the emotion in the ith row and jth column of the matrix

        print("\n")
        start_time = end_time

face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_detection_model = load_model('emotion_detection_model.h5')
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Capture img from video and save in folder
absolute_path = "C:/Users/HP/PycharmProjects/AI_Technology/test"
num_vid = 1
# Load the video file - absolute path
video_path = f"C:/Users/HP/PycharmProjects/AI_Technology/test/KeyandPeele.mp4"
cap = cv2.VideoCapture(video_path)

# Create a folder to save the images - absolute path
fold_cap = f"C:/Users/HP/PycharmProjects/AI_Technology/test/img_from_vid"
if not os.path.exists(fold_cap):
    os.makedirs(fold_cap)
# Create a set to store emotions for each detected face
facing = np.empty((len(sentences), 7))
a = 0
b = 0

for sentence in sentences:
    sentence_words = sentence.split()
    sentence_duration = len(sentence_words) / words_per_second
    start_time = 0
    end_time = 0
    sentence_image_folder = os.path.join(fold_cap, f'{sentence_words[0]}_{sentence_words[-1]}')

    # Replace invalid characters with a valid character, such as "_"
    sentence_image_folder = sentence_image_folder.replace("?", "_")
    if not os.path.exists(sentence_image_folder):
        os.makedirs(sentence_image_folder)
    for i in range(10):
        interval = sentence_duration / 10
        start_time = end_time
        end_time += interval
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) * interval)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        for j in range(frame_interval):
            ret, frame = cap.read()
            if ret:
                image_path = os.path.join(sentence_image_folder, f"frame_{i}_{j}.jpg")
                cv2.imwrite(image_path, frame)
                print(image_path)
            else:
                break

    # Define the counters for each emotion
    emotions_count = {label: 0 for label in emotion_labels.values()}

    for image in os.listdir(sentence_image_folder):
        # Load image
        img_path = os.path.join(sentence_image_folder, image)
        # print("P1", image_path)
        img = cv2.imread(img_path)
        # Detect faces in image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detection_model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(150, 150))

        # For each detected face, predict the emotion
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.reshape(face_img, [1, face_img.shape[0], face_img.shape[1], 1]) / 255.0
            # predict
            emotion_probs = emotion_detection_model.predict(face_img)[0]
            emotion_index = np.argmax(emotion_probs)
            emotion = emotion_labels[emotion_index]
            emotions_count[emotion] += 1
        #     # Draw the bounding box and label on the frame
        #     print("P2",image_path)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     cv2.putText(frame, emotion_labels[emotion_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        #
        # # Show the frame
        # cv2.imshow('Face Emotion Recognition', frame)
        #
        # # Exit on 'q' key press
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Print emotion count for each image folder
    print(f"Emotion count for sentence '{sentence}':")
    for emotion, count in emotions_count.items():
        if emotion == 'Happy':
            a = 0
        elif emotion == 'Angry':
            a = 1
        elif emotion == 'Surprise':
            a = 2
        elif emotion == 'Sad':
            a = 3
        elif emotion == 'Fear':
            a = 4
        elif emotion == 'Disgust':
            a = 5
        elif emotion == 'Neutral':
            a = 6
        facing[b][a] = count
        print(f"{emotion}: {count}")
    b += 1

print(modifier)
print(facing)

emotional = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear', 'Disgust', 'Neutral']

for i in range(len(modifier)):
    for j in range(len(emotional)):
        if modifier[i][j] > 0:
            facing[i][j] *= 8

for i in range(len(facing)):
    max_emotion = emotional[0]
    max_value = facing[i][0]
    for j in range(1, len(emotional)):
        if facing[i][j] > max_value:
            max_value = facing[i][j]
            max_emotion = emotional[j]
    print(f'"{sentences[i]}", the highest emotion is {max_emotion}')