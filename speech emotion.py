import os
import numpy as np
import librosa
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Path to dataset (change as per your directory)
dataset_path = '/Users/praveenmahan/Library/CloudStorage/OneDrive-BVRajuInstituteofTechnology/telugu'  # Update with your path (containing subdirectories happy, sad, neutral)

# Function to extract MFCC features
def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Load dataset (this assumes your dataset is structured in subdirectories for each emotion)
def load_data(dataset_path):
    paths = []
    labels = []
    for emotion in os.listdir(dataset_path):
        emotion_folder = os.path.join(dataset_path, emotion)
        if os.path.isdir(emotion_folder):  # Only process directories
            for filename in os.listdir(emotion_folder):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    file_path = os.path.join(emotion_folder, filename)
                    paths.append(file_path)
                    labels.append(emotion)  # Label is the folder name (emotion)
    
    print(f"Dataset loaded. Total files: {len(paths)}")
    return paths, labels

# Load and process data
paths, labels = load_data(dataset_path)
df = pd.DataFrame({'speech': paths, 'label': labels})

# Extract MFCC features for all files
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc = X_mfcc.dropna()  # Drop any failed extractions

# Convert to numpy array
X = np.array([x for x in X_mfcc])
X = np.expand_dims(X, -1)  # Reshaping to match LSTM input

# Encode labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
y = np.expand_dims(y, axis=1)

# Print the number of classes
num_classes = len(np.unique(y))
print(f"Number of emotion classes: {num_classes}")

# Build the LSTM model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # Output layer: number of classes
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

# Plot training and validation accuracy
epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='Train accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
model.save('speech_emotion_recognition_model.h5')
print("Model saved!")

# Test accuracy evaluation (if you have separate test data)
# Assuming you have a test dataset or you can split the data into training and testing sets.
# Example: 
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_accuracy:.2f}")
