import cv2
import numpy as np
import os
import argparse
import openai
import time
from openai import OpenAI
from tensorflow.keras.models import load_model  # type: ignore

from config import actions, sequence_length, detection_threshold, MODEL_PATH
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Set your OpenAI key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_sign_language_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def query_llm_for_sentence(action_list):
    prompt = (
        "You are a helpful assistant that translates American Sign Language gestures into full English sentences. "
        "Given the following words, construct a grammatically correct and properly punctuated sentence using ONLY these words. "
        "Do not add any new words or information unless it is clearly and unambiguously implied by the context of the listed words.\n"
        f"Words: {', '.join(action_list)}\n"
        "Sentence:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You only translate given words into proper English sentences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None

def run_sentence_detection(threshold=detection_threshold, show_landmarks=True):
    model = load_sign_language_model()
    if model is None:
        return

    sequence = []
    current_sentence = []
    predictions = []
    final_sentence = ""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Starting sentence detection (press 'q' to quit)...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            if show_landmarks:
                draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                input_data = np.expand_dims(sequence, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                predicted_class = np.argmax(res)

                predictions.append(predicted_class)
                predictions = predictions[-10:]

                if len(predictions) >= 5:
                    prediction_counts = np.bincount(predictions[-5:])
                    most_common = prediction_counts.argmax()
                    consistency = prediction_counts[most_common] / 5.0

                    if res[most_common] > threshold and consistency >= 0.6:
                        detected_word = actions[most_common]
                        if not current_sentence or detected_word != current_sentence[-1]:
                            current_sentence.append(detected_word)

            if len(current_sentence) >= 6:
                sentence = query_llm_for_sentence(current_sentence)
                if sentence:
                    final_sentence = sentence
                    current_sentence = []

            clean_image = image.copy()

            # Show finalized sentence only (no live action words)
            if final_sentence:
                sentence_size = cv2.getTextSize(final_sentence, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                sentence_x = (clean_image.shape[1] - sentence_size[0]) // 2
                sentence_y = clean_image.shape[0] - 30
                cv2.rectangle(clean_image, (sentence_x - 20, sentence_y - sentence_size[1] - 20),
                              (sentence_x + sentence_size[0] + 20, sentence_y + 20), (0, 0, 255), -1)
                cv2.putText(clean_image, final_sentence, (sentence_x, sentence_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow('ASL to Sentence', clean_image)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

def main():
    parser = argparse.ArgumentParser(description='Refined ASL detection with sentence construction')
    parser.add_argument('--threshold', type=float, default=detection_threshold,
                        help='Confidence threshold for detection')
    parser.add_argument('--no-landmarks', action='store_true',
                        help='Hide MediaPipe landmarks')
    args = parser.parse_args()

    run_sentence_detection(threshold=args.threshold, show_landmarks=not args.no_landmarks)

if __name__ == "__main__":
    main()