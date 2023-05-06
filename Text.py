import os
# from pytube import YouTube
import whisper
import nltk
import text2emotion as te

# import emoji
# import subprocess
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

with open("transcription.txt", "w") as file:
    start_time = 0.0
    for sentence in sentences:
        num_words = len(sentence.split())
        end_time = start_time + num_words / words_per_second
        file.write(f"{sentence} ({start_time:.2f}-{end_time:.2f})\n")
        start_time = end_time

start_time = 0.0
for sentence in sentences:
    num_words = len(sentence.split())
    end_time = start_time + num_words / words_per_second
    print(f"{sentence} ({start_time:.2f}-{end_time:.2f})")
    emotions = te.get_emotion(sentence)
    if sum(emotions.values()) == 0:
        emotions["Neutral"] = 1
    else:
        emotions["Neutral"] = 0.0
    print(emotions)
    start_time = end_time
    print("\n")

from collections import defaultdict

# Create a dictionary to store emotion counts
emotion_counts = defaultdict(int)

for sentence in sentences:
    emotions = te.get_emotion(sentence)
    for emotion, count in emotions.items():
        if count > 0:
            emotion_counts[emotion] += 1

# Print the emotion counts
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")
