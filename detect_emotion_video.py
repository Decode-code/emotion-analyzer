import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import imutils
import time

# Load face detector model
faceProto = "/Users/praveenmahan/Library/CloudStorage/OneDrive-BVRajuInstituteofTechnology/sunny/face_detector/deploy.prototxt"  # path to the prototxt file
faceModel = "/Users/praveenmahan/Library/CloudStorage/OneDrive-BVRajuInstituteofTechnology/sunny/face_detector/res10_300x300_ssd_iter_140000.caffemodel"  # path to the caffemodel file
faceNet = cv2.dnn.readNet(faceProto, faceModel)

# Load emotion detection model
emotion_model_path = 'emotion_model.h5'  # Path to the emotion detection model
maskNet = load_model(emotion_model_path)

# Emotion labels (you can adjust this based on your model's output)
emotion_labels = [ 'Happy', 'Sad', 'Neutral']

# Initialize video stream (webcam)
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# Function to detect faces and predict emotions
def detect_and_predict_emotion(frame, faceNet, maskNet, threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the frame dimensions
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (48, 48))  # Adjust size based on model requirements
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            locs.append((startX, startY, endX, endY))
            preds.append(maskNet.predict(face)[0])

    return (locs, preds)

# Main loop to process the video stream
while True:
    ret, frame = vs.read()
    if not ret:
        print("[INFO] Failed to grab frame")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and predict emotions
    (locs, preds) = detect_and_predict_emotion(frame, faceNet, maskNet, threshold=0.5)

    # Loop over the detected faces and emotions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        label = emotion_labels[np.argmax(pred)]  # Get the emotion with the highest probability
        color = (0, 255, 0) if label == 'Happy' else (0, 0, 255)  # Color based on emotion

        # Draw bounding box and label on the frame
        cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)

    # Show the frame
    cv2.imshow("Emotion Recognition", original_frame)

    # Press 'q' to exit the video stream
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup and close the video window
cv2.destroyAllWindows()
vs.release()
