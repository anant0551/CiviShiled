import os
import cv2
import numpy as np
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
FRAME_SIZE = (128, 128)  # Resize frames to 128x128
FRAMES_PER_VIDEO = 30  # Number of frames to use per video
DATASET_PATH = r"F:\1605\archive\Dataset2"  # Path to dataset
LABELS = {"violence": 0, "non_violence": 1}

# Function to extract frames from videos
def extract_frames(video_path, frame_size, frames_per_video):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // frames_per_video)
    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Prepare dataset
def prepare_dataset(dataset_path, labels, frame_size, frames_per_video):
    X, y = [], []
    for label, class_idx in labels.items():
        folder_path = os.path.join(dataset_path, label)
        print(f"Accessing folder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
        else:
            print(f"Folder exists: {folder_path}")
        for video_name in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_name)
            frames = extract_frames(video_path, frame_size, frames_per_video)
            if len(frames) == frames_per_video:  # Ensure consistent frame count
                X.append(frames)
                y.append(class_idx)
    return np.array(X), np.array(y)

# Load dataset
print("Loading dataset...")
X, y = prepare_dataset(DATASET_PATH, LABELS, FRAME_SIZE, FRAMES_PER_VIDEO)
y = to_categorical(y, num_classes=len(LABELS))
X = X / 255.0  # Normalize pixel values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('bm')
# Feature extraction with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3))

def extract_features(data, model):
    features = []
    for video in data:
        video_features = model.predict(video, verbose=0)
        video_features = np.mean(video_features, axis=0)  # Average over frames
        features.append(video_features)
    return np.array(features)

print("Extracting features...")
X_train_features = extract_features(X_train, base_model)
X_test_features = extract_features(X_test, base_model)

# Build classification model
model = Sequential([
    GlobalAveragePooling2D(input_shape=X_train_features.shape[1:]),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("Training model...")
model.fit(X_train_features, y_train, validation_data=(X_test_features, y_test), epochs=10, batch_size=32)

# Save model
model.save("violence_classifier.h5")
print("Model saved.")
