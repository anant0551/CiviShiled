import cv2
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.models import load_model
import cvlib as cv
import tensorflow as tf

# Load the violence detection model
violence_model = tf.keras.models.load_model('violence_classifier.h5')

# Load the gender detection model
gender_model = load_model('gender_detection.h5')

# Load the MobileNetV2 model for feature extraction (same as during training for violence model)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Define the classes for gender prediction
gender_classes = ['man', 'woman']

# Function to preprocess the frame for violence detection
def preprocess_frame(frame, target_size=(128, 128)):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Function to extract features using MobileNetV2 for violence detection
def extract_features(frame, model):
    frame_features = model.predict(frame, verbose=0)
    frame_features = np.mean(frame_features, axis=0)  # Average over spatial dimensions
    return frame_features

# Open webcam
webcam = cv2.VideoCapture(0)

# Loop through frames from the webcam
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection using cvlib
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop_resized = cv2.resize(face_crop, (96, 96))
        face_crop_resized = face_crop_resized.astype("float") / 255.0
        face_crop_resized = img_to_array(face_crop_resized)
        face_crop_resized = np.expand_dims(face_crop_resized, axis=0)

        # Apply gender detection on face
        gender_conf = gender_model.predict(face_crop_resized)[0]
        gender_idx = np.argmax(gender_conf)
        gender_label = gender_classes[gender_idx]
        gender_confidence = gender_conf[gender_idx] * 100

        # Preprocessing for violence detection model (using the whole frame)
        frame_resized = preprocess_frame(frame)

        # Extract features for violence detection
        features = extract_features(frame_resized, base_model)

        # Predict violence (0: Violence, 1: Non-Violence)
        violence_prediction = violence_model.predict(np.expand_dims(features, axis=0))  # Add batch dimension
        violence_label = "Violence" if np.argmax(violence_prediction) == 0 else "Non-Violence"
        violence_confidence = np.max(violence_prediction) * 100

        # Display results on the frame
        # Gender label
        cv2.putText(frame, f"Gender: {gender_label} ({gender_confidence:.2f}%)", 
                    (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Violence label
        cv2.putText(frame, f"Violence: {violence_label} ({violence_confidence:.2f}%)", 
                    (startX, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display output
    cv2.imshow("Gender & Violence Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
