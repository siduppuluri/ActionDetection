"""
Configuration parameters for the sign language recognition system.
"""

import os
import numpy as np

# Project paths
DATA_PATH = os.path.join('MP_Data')  
MODEL_PATH = 'action.keras'  
LOG_DIR = os.path.join('Logs')

# Data collection parameters
actions = np.array(['hello', 'thanks', 'iloveyou', ' '])  
no_sequences = 30  
sequence_length = 30  
camera_index = 0

# MediaPipe parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Real-time detection parameters
detection_threshold = 0.5  