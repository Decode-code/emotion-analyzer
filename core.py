import cv2
import numpy as np
import torch
import torchaudio
import threading
import time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import pyaudio
import wave

# Load XLS-R model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
xlsr_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
xlsr_model.eval()

# Your fine-tuned XLS-R classifier (should be a torch.nn.Module)
# Assume you've saved and loaded it:
class XLSREmotionClassifier(torch.nn.Module):
    def _init_(self):
        super(XLSREmotionClassifier, self)._init_()
        self.fc = torch.nn.Linear(1024, 3)  # 3 emotion classes

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

emotion_classifier = XLSREmotionClassifier()
emotion_classifier.load_state_dict(torch.load("xlsr_emotion_classifier.pth"))
emotion_classifier.eval()

# Load ResNet50 model (trained on facial expressions)
video_model = load_model("resnet50_facial_emotion.h5")
emotion_labels = ['Positive', 'Negative', 'Neutral']

# Face detector (OpenCV DNN)
faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(faceProto, faceModel)

# Audio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio_frames = []
p = pyaudio.PyAudio()
audio_probs = np.zeros(3)
predicted_audio_emotion = "Neutral"

def record_audio():
    global audio_frames
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)

def process_audio():
    global predicted_audio_emotion, audio_probs
    while True:
        if len(audio_frames) >= int(RATE / CHUNK * 3):  # 3 seconds
            frames = audio_frames[:int(RATE / CHUNK * 3)]
            del audio_frames[:int(RATE / CHUNK * 3)]
            with wave.open("temp_audio.wav", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

            waveform, sample_rate = torchaudio.load("temp_audio.wav")
            if sample_rate != RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, RATE)

            input_values = processor(waveform.squeeze(), sampling_rate=RATE, return_tensors="pt").input_values
            with torch.no_grad():
                features = xlsr_model(input_values).last_hidden_state
                pooled = torch.mean(features, dim=1)
                emotion_logits = emotion_classifier(pooled)
                emotion_probs = emotion_logits.detach().cpu().numpy()[0]
                predicted_audio_emotion = emotion_labels[np.argmax(emotion_probs)]
                audio_probs[:] = emotion_probs

# Video-based emotion detection
def detect_emotion(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = preprocess_input(img_to_array(face))
            face = np.expand_dims(face, axis=0)

            preds = video_model.predict(face)[0]
            return emotion_labels[np.argmax(preds)], preds

    return "Neutral", np.zeros(3)

# Start threads
threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=process_audio, daemon=True).start()

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    video_emotion, video_probs = detect_emotion(frame)

    # Fusion
    if np.any(video_probs) and np.any(audio_probs):
        fused = (video_probs + audio_probs) / 2
        final_emotion = emotion_labels[np.argmax(fused)]
    else:
        final_emotion = video_emotion if np.any(video_probs) else predicted_audio_emotion

    # Display
    cv2.putText(frame, f"Audio: {predicted_audio_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Video: {video_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Final: {final_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Multimodal Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.terminate()