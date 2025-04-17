"""
Minimal sign language detection script.
Only displays the detected word, without additional visual elements.
"""

import cv2
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model # type: ignore

from config import actions, sequence_length, detection_threshold, MODEL_PATH
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints

def load_sign_language_model():
    """
    Load the trained sign language model.
    
    Returns:
        keras.Model: Loaded model or None if not found
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first by running model.py")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_minimal_detection(threshold=detection_threshold, show_landmarks=True):
    """
    Run minimal sign language detection that only shows the detected word.
    
    Args:
        threshold: Confidence threshold for predictions
        show_landmarks: Whether to show MediaPipe landmarks
    """
    # Load model
    model = load_sign_language_model()
    if model is None:
        return
    
    # Initialize variables
    sequence = []
    current_detection = None
    predictions = []
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting minimal detection (press 'q' to quit)...")
    
    # Set up MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks if enabled
            if show_landmarks:
                draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            # Keep only the most recent frames
            sequence = sequence[-sequence_length:]
            
            # Make prediction once we have enough frames
            if len(sequence) == sequence_length:
                # Reshape for model input
                input_data = np.expand_dims(sequence, axis=0)
                
                # Get prediction
                res = model.predict(input_data, verbose=0)[0]
                predicted_class = np.argmax(res)
                
                # Add to predictions list for temporal smoothing
                predictions.append(predicted_class)
                
                # Only keep recent predictions for stability
                if len(predictions) > 10:
                    predictions = predictions[-10:]
                
                # Temporal smoothing - check if predictions are consistent
                if len(predictions) >= 5:
                    # Get most common prediction in the window
                    prediction_counts = np.bincount(predictions[-5:])
                    most_common = prediction_counts.argmax()
                    consistency = prediction_counts[most_common] / 5.0
                    
                    # Only show word if confident and consistent enough
                    if res[most_common] > threshold and consistency >= 0.6:
                        current_detection = actions[most_common]
                    else:
                        current_detection = None
                
            # Show current detection (if any)
            clean_image = image.copy()
            if current_detection is not None:
                # Display the word in the center of the screen
                text_size = cv2.getTextSize(
                    current_detection, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    2
                )[0]
                
                text_x = (clean_image.shape[1] - text_size[0]) // 2
                text_y = clean_image.shape[0] - 50
                
                # Add a semi-transparent background for better readability
                cv2.rectangle(
                    clean_image,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    (0, 0, 0),
                    -1
                )
                
                # Draw the text
                cv2.putText(
                    clean_image,
                    current_detection,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Show frame
            cv2.imshow('Sign Language', clean_image)
            
            # Break on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

def main():
    """Main function to run minimal detection."""
    parser = argparse.ArgumentParser(description='Minimal sign language detection')
    parser.add_argument('--threshold', type=float, default=detection_threshold, 
                        help='Confidence threshold for detection')
    parser.add_argument('--no-landmarks', action='store_true', 
                        help='Hide MediaPipe landmarks')
    args = parser.parse_args()
    
    # Run detection with specified parameters
    run_minimal_detection(threshold=args.threshold, show_landmarks=not args.no_landmarks)

if __name__ == "__main__":
    main()