# Emotion Detection based on Facial Recognition and Sentiment Analysis

## Overview

This project focuses on detecting emotions through facial recognition and sentiment analysis. It combines the analysis of facial expressions with text-based sentiment analysis to provide comprehensive insights into emotional states.

## Applications

### Marketing
- **Customer Sentiment Analysis:** Analyze customer reviews, comments, and feedback to understand their sentiment towards products or services.
- **Emotion Recognition in Advertising:** Use facial emotion recognition to analyze emotional responses to advertisements.

### Healthcare
- **Patient Care Improvement:** Analyze patient feedback and comments to understand emotional states and enhance care.
- **Mental Health Diagnosis:** Use facial emotion recognition for diagnosing and treating mental health disorders.

### Education
- **Student Support:** Analyze student feedback and comments to understand emotional states and provide targeted support.
- **Classroom Engagement:** Use facial emotion recognition to monitor student engagement and attention in classrooms.

### Human Resources
- **Employee Feedback Analysis:** Analyze employee feedback and comments to understand emotional states and identify areas for improvement.
- **Job Interviews:** Use facial emotion recognition in job interviews to analyze candidates' emotional responses.

## Tools Used

- **Code Editor:** PyCharm, Google Colab
- **Libraries & Models:**
  - **Text Analysis:** whisper, Model: text2emotion
  - **Facial Recognition:** cv2, numpy, keras.models, tensorflow, Model: haarcascade_frontalface_default.xml, emotion_detection_model.h5

## Methodology

Our program is divided into two parts: facial expression analysis and text-based sentiment analysis. This dual approach aims to enhance emotion detection accuracy by combining results from both methods.

### Facial Expression Analysis
We analyze subjects' faces for key expressions indicative of the following emotions:
- Angry, Fear, Happy, Neutral, Sad, Surprise, Disgust

### Text-based Sentiment Analysis
We analyze subjects' speech using pre-trained algorithms to detect keywords indicative of emotional states such as anxiety, happiness, or disgust. Neutral tones are identified when no specific emotional keywords are detected. If multiple emotions are detected within a sentence, percentages are assigned to indicate the prevalence of each emotion.

## Limitations

Despite the integration of text2emotion and facial recognition, the system may encounter errors. For instance, transcription errors like "tool" being recognized as "too" or "then" as "bend" may occur, affecting accuracy to some extent.
