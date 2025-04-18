# emotion-analyzer
EAS uses a microphone and camera to analyze the user's voice and facial expressions and then processes the results through a neural network to determine the user's emotions accurately.
# 🎙️ Emotion Analyzer - Speech Emotion Recognition using Wav2Vec2 + Keras

This repository provides a speech emotion recognition system that classifies emotions from Telugu audio clips using Wav2Vec2 XLS-R for feature extraction and a custom Keras classifier. The model is trained to detect emotions such as **Positive**, **Negitive**, and **Neutral**.

## 🚀 Features

- Utilizes [Facebook's Wav2Vec2 XLS-R](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) for powerful multilingual speech representations.
- Designed for **Telugu** audio emotion classification.
- Fast and lightweight classifier using Keras dense layers.
- Easily extendable to other Indian languages with minimal changes.

## 🧠 Model Architecture

1. **Wav2Vec2 XLS-R** (pretrained): Extracts robust audio features.
2. **Keras Dense Classifier**: Custom neural network to classify emotion states.

---

## 📁 Files

| File | Description |
|------|-------------|
| `speech emotion.py` | Main script for recording, processing, and predicting emotion from audio. |
| `train_emotion_detector.py` | Script to train the emotion classifier using precomputed Wav2Vec2 embeddings. |
| `emotion_model.h5` | Trained Keras model for emotion classification. |
| `temp_audio.wav` | Sample Telugu audio file used for testing. |

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Decode-code/emotion-analyzer.git
   cd emotion-analyzer


📦 Requirements
Python 3.7+

TensorFlow

Torchaudio

HuggingFace Transformers

Librosa

NumPy

Sounddevice

## Final OutPut

<p align="center">
  <img src="https://github.com/user-attachments/assets/569859b4-7275-4d3a-a747-f1633d0b29bb" alt="positive" width="200"/>
  <img src="https://github.com/user-attachments/assets/5747a9fe-09d6-4d11-8983-b3053ff9da35" alt="neutral+" width="200"/>
  <img src="https://github.com/user-attachments/assets/0a33c4bb-c6d6-400e-8393-ab3d1dc7455d" alt="negative" width="200"/>
  <img src="https://github.com/user-attachments/assets/f5f7457e-d72d-4729-b3c9-8081a2192d44" alt="neutral-" width="200"/>
</p>



## Scan Me For More Details 

<img src="https://github.com/user-attachments/assets/06912cc6-28a8-4a05-baab-9591ffec3e20" alt="URL QR Code" width="150"/>
