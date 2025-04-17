import cv2
import numpy as np
import os
import shutil
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

from mediapipe_utils import extract_keypoints, mp_holistic, mediapipe_detection, draw_styled_landmarks
from config import DATA_PATH, actions, no_sequences, sequence_length

def check_existing_actions():
    """
    Check which actions already have complete training data.
    
    Returns:
        tuple: (completed_actions, new_actions)
    """
    completed_actions = []
    new_actions = []
    
    # Create main data directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return [], list(actions)
    
    # Check each action
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        
        # If action directory doesn't exist, it's a new action
        if not os.path.exists(action_dir):
            new_actions.append(action)
            continue
        
        # Count sequence folders (expect 30 folders, each with 30 frames)
        sequence_folders = [f for f in os.listdir(action_dir) if f.isdigit()]
        complete_sequences = 0
        
        # Check each sequence folder for completeness
        for seq_folder in sequence_folders:
            seq_path = os.path.join(action_dir, seq_folder)
            frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            if len(frame_files) >= sequence_length:
                complete_sequences += 1
        
        # If we have enough complete sequences, consider it done
        if complete_sequences >= no_sequences:
            completed_actions.append(action)
        else:
            new_actions.append(action)
    
    return completed_actions, new_actions

def create_folders_for_new_actions(new_actions):
    """
    Create folder structure only for new actions.
    
    Args:
        new_actions: List of new actions that need folders
    """
    for action in new_actions:
        action_dir = os.path.join(DATA_PATH, action)
        
        # Create action directory if it doesn't exist
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)
            print(f"Created directory for action: {action}")
        
        # Create sequence directories
        for sequence in range(1, no_sequences + 1):
            sequence_dir = os.path.join(action_dir, str(sequence))
            if not os.path.exists(sequence_dir):
                os.makedirs(sequence_dir)
    
    print(f"Created directories for {len(new_actions)} new actions")

def collect_data_for_new_actions(new_actions):
    """
    Collect data only for new actions, preserving existing data.
    
    Args:
        new_actions: List of new actions to collect data for
    """
    if not new_actions:
        print("No new actions to collect data for.")
        return
    
    # Create folders for new actions
    create_folders_for_new_actions(new_actions)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Single prompt at the beginning
    print("\n=== STREAMLINED DATA COLLECTION ===")
    print(f"You will record {no_sequences} sequences for each of these NEW actions: {', '.join(new_actions)}")
    print(f"Each sequence will capture {sequence_length} frames")
    print("Press any key to start collection (or 'q' to quit)...")
    
    # Show preparation screen once
    prep_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(prep_frame, "Ready to start data collection", (80, 200), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(prep_frame, f"New actions: {', '.join(new_actions)}", (80, 250), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(prep_frame, "Press any key to begin", (120, 300), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(prep_frame, "Press 'q' to quit anytime", (120, 350), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('OpenCV Feed', prep_frame)
    
    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through new actions only
        for action in new_actions:
            # Loop through sequences
            for sequence in range(1, no_sequences + 1):
                # Notify upcoming sequence
                print(f"\nStarting '{action}' - Sequence {sequence}/{no_sequences}")
                
                # Brief pause before starting sequence
                notification_start = cv2.getTickCount()
                while (cv2.getTickCount() - notification_start) / cv2.getTickFrequency() < 1.5:  # 1.5 second pause
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    # Overlay text about the upcoming sequence
                    image = frame.copy()
                    cv2.putText(image, f"Get ready for: {action}", (15, 12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f"Sequence: {sequence}/{no_sequences}", (15, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    
                    # Check for quit
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # Loop through frames for this sequence
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        continue

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Status info
                    cv2.putText(image, f'Collecting: {action}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Sequence: {sequence}/{no_sequences}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Frame: {frame_num+1}/{sequence_length}', (15, 48), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Data collection canceled by user")
                        return
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nData collection completed successfully!")

def preprocess_data():
    """
    Load and preprocess all data (both existing and new).
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Create label map
    label_map = {label: num for num, label in enumerate(actions)}
    
    # Collect sequences data
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        # Skip if action folder doesn't exist
        if not os.path.exists(action_path):
            print(f"Warning: Action folder '{action}' not found")
            continue
        
        # Get numeric sequence folders and sort them
        sequence_folders = [f for f in os.listdir(action_path) if f.isdigit()]
        sequence_folders.sort(key=int)
        
        print(f"Found {len(sequence_folders)} sequences for '{action}'")
        
        # Process each sequence
        for sequence_folder in sequence_folders:
            sequence_path = os.path.join(action_path, sequence_folder)
            window = []
            
            # Check if we have enough frames
            frame_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
            if len(frame_files) < sequence_length:
                print(f"Warning: Sequence {sequence_path} has only {len(frame_files)} frames")
                continue
            
            # Load frames
            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                try:
                    res = np.load(frame_path)
                    window.append(res)
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
                    break
            
            # Add to dataset if complete
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                print(f"Skipping incomplete sequence: {sequence_path}")
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    print(f"Dataset - Total: {len(sequences)} sequences")
    print(f"Training: {len(X_train)} sequences, Testing: {len(X_test)} sequences")
    
    return X_train, X_test, y_train, y_test

def main():
    """Run the intelligent data collection pipeline."""
    print("Sign Language Recognition - Data Collection")
    print("===========================================")
    
    # Check which actions are new vs. already completed
    completed_actions, new_actions = check_existing_actions()
    
    # Show status of actions
    if completed_actions:
        print(f"\nActions with complete training data: {', '.join(completed_actions)}")
    
    if new_actions:
        print(f"\nActions that need training: {', '.join(new_actions)}")
        
        # Ask if user wants to collect data for new actions
        choice = input(f"Collect data for {len(new_actions)} new actions? (y/n): ").lower()
        if choice == 'y':
            collect_data_for_new_actions(new_actions)
        else:
            print("Data collection skipped.")
    else:
        print("\nAll actions already have complete data. Nothing new to train!")
        
        # Option to recollect all data if desired
        choice = input("Do you want to recollect ALL data? This will DELETE existing data. (y/n): ").lower()
        if choice == 'y':
            # Revert to original behavior for complete recollection
            print("Recollecting all data...")
            
            # Original create_folders function with reset=True
            if os.path.exists(DATA_PATH):
                print(f"Removing existing data directory: {DATA_PATH}")
                shutil.rmtree(DATA_PATH)
            
            for action in actions:
                action_dir = os.path.join(DATA_PATH, action)
                os.makedirs(action_dir, exist_ok=True)
                print(f"Created directory for action: {action}")
                
                for sequence in range(1, no_sequences + 1):
                    sequence_dir = os.path.join(action_dir, str(sequence))
                    os.makedirs(sequence_dir, exist_ok=True)
            
            # Collect data for all actions
            collect_data_for_new_actions(actions)
    
    # Preprocess all data
    print("\nPreprocessing all available data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    print("\nData preprocessing completed.")
    print("You can now run model.py to train the model.")

if __name__ == "__main__":
    main()