import cv2  # OpenCV for real-time image/video processing and face detection
import numpy as np  # NumPy for handling numerical data and arrays
import librosa  # Librosa for audio processing and feature extraction (like MFCCs)
import pyaudio  # PyAudio for capturing audio from the microphone
import wave  # Wave for saving audio data in WAV file format
import threading  # Threading for running audio recording and processing simultaneously
import imutils  # Imutils for simplifying OpenCV functions (e.g., resizing frames)

from keras.models import load_model  # Keras method to load pre-trained deep learning models
from tensorflow.keras.preprocessing.image import img_to_array  # Converts images to numpy arrays for model input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Preprocessing for MobileNetV2 model input

from sklearn.preprocessing import LabelEncoder  # Converts string labels to numerical form and back

import time  # Time module for delays and timing operations


# Load pre-trained models
audio_model = load_model('speech_emotion_recognition_model.h5')
video_model = load_model('emotion_model.h5')

# LabelEncoder for emotion labels
video_emotion_labels = ['Positive', 'Negtive', 'Neutral']
label_encoder = LabelEncoder()
label_encoder.fit(video_emotion_labels)

# Audio recording settings
FORMAT = pyaudio.paInt16 # Audio format (16-bit PCM)
CHANNELS = 1             # Mono audio
RATE = 16000             # 16 kHz sampling rate
CHUNK = 1024             # Buffer size
p = pyaudio.PyAudio()    # Initialize PyAudio
audio_frames = []        # Buffer to store audio stream
predicted_audio_emotion = "Neutral"  # Default emotion

# Load face detector model
faceProto = "/Users/praveenmahan/Library/CloudStorage/OneDrive-BVRajuInstituteofTechnology/sunny/face_detector/deploy.prototxt"
faceModel = "/Users/praveenmahan/Library/CloudStorage/OneDrive-BVRajuInstituteofTechnology/sunny/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(faceProto, faceModel)

# Function to record audio
def record_audio():
    global audio_frames
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)

# Function to extract features and predict emotion from audio
def save_audio_and_predict_emotion():
    global predicted_audio_emotion
    while True:
        if len(audio_frames) >= int(RATE / CHUNK * 3):  # every 3 seconds
            frames = audio_frames[:int(RATE / CHUNK * 3)]
            del audio_frames[:int(RATE / CHUNK * 3)]

            # Save to WAV
            with wave.open("temp_audio.wav", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

            try:
                y, sr = librosa.load("temp_audio.wav", duration=3, offset=0.5)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512)
                mfccs = np.mean(mfccs.T, axis=0)
                mfccs = np.expand_dims(mfccs, axis=-1)
                mfccs = np.expand_dims(mfccs, axis=0)
                emotion_pred = audio_model.predict(mfccs)
                emotion_label = np.argmax(emotion_pred, axis=1)
                predicted_audio_emotion = label_encoder.inverse_transform(emotion_label)[0]
            except Exception as e:
                print(f"[AUDIO ERROR]: {e}")

# Function to detect face and predict emotion
def detect_and_predict_emotion(frame, faceNet, model, threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (48, 48))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]
                return video_emotion_labels[np.argmax(preds)], preds
            except:
                return "Neutral", np.array([0, 0, 1])
    return "Neutral", np.array([0, 0, 1])

# Start audio threads
threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=save_audio_and_predict_emotion, daemon=True).start()

# Start webcam
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=400)

    # Get video emotion
    video_emotion, video_prob = detect_and_predict_emotion(frame, faceNet, video_model)

    # Convert audio emotion to one-hot prob
    audio_prob = np.zeros_like(video_prob)
    if predicted_audio_emotion in video_emotion_labels:
        idx = video_emotion_labels.index(predicted_audio_emotion)
        audio_prob[idx] = 1.0

    # Average both predictions
    final_prob = (audio_prob + video_prob) / 2
    final_label = video_emotion_labels[np.argmax(final_prob)]

    # Print to terminal
    print(f"[Audio]: {predicted_audio_emotion} | [Video]: {video_emotion} | [Final]: {final_label}")

        # Show on frame with updated colors
    cv2.putText(frame, f"Audio: {predicted_audio_emotion}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)      # Green

    cv2.putText(frame, f"Video: {video_emotion}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)    # Yellow

    cv2.putText(frame, f"Final: {final_label}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)      # Blue


    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.terminate()
