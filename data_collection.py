import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

from mediapipe_keypoints import extract_keypoints, mp_holistic, mediapipe_detection, draw_styled_landmarks



#FOLDER SETUP FOR COLLECTION ####################################################


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 1

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()


# PREPROCESS DATA AND CREATE LABELS ####################################################


label_map = {label:num for num, label in enumerate(actions)}

label_map

# FRAME EXTRACTION ####################################################

sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    
    # Get only numeric folder names and sort them
    sequence_folders = [f for f in os.listdir(action_path) if f.isdigit()]
    sequence_folders.sort(key=int)  # Sort numerically
    
    print(f"Found {len(sequence_folders)} sequence folders for {action}: {sequence_folders}")
    
    for sequence_folder in sequence_folders:
        sequence_path = os.path.join(action_path, sequence_folder)
        window = []
        
        # Check if we have enough frames in this sequence
        frame_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
        if len(frame_files) < sequence_length:
            print(f"Warning: {sequence_path} has only {len(frame_files)} frames, expected {sequence_length}")
            continue
            
        # Load each frame
        for frame_num in range(sequence_length):
            frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
            try:
                res = np.load(frame_path)
                window.append(res)
            except FileNotFoundError:
                print(f"Error: Missing frame file {frame_path}")
                break
        
        # Only add complete sequences
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Skipping incomplete sequence {sequence_path}: only {len(window)}/{sequence_length} frames")

np.array(sequences).shape
np.array(labels).shape

X = np.array(sequences)
X.shape

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

y_test.shape

