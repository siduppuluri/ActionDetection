"""
Enhanced model definition and training for sign language recognition.
Dynamically adapts to whatever words are present in the data.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

from config import actions as config_actions, DATA_PATH, MODEL_PATH, LOG_DIR
from data_collection_refined import preprocess_data  # Use the standard data_collection.py

def get_available_actions():
    """
    Check which actions actually have data in MP_Data.
    This makes the system dynamic, adapting to whatever data is available.
    
    Returns:
        list: Actions that have data available
    """
    available_actions = []
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data directory {DATA_PATH} does not exist.")
        return config_actions  # Return configured actions as fallback
    
    # Check for each action from config
    for action in config_actions:
        action_dir = os.path.join(DATA_PATH, action)
        if os.path.exists(action_dir):
            # Check if there's at least one sequence with data
            sequence_folders = [f for f in os.listdir(action_dir) if f.isdigit()]
            if sequence_folders:
                available_actions.append(action)
    
    # If no actions found, fallback to config actions
    if not available_actions:
        print(f"Warning: No action data found. Using configured actions: {config_actions}")
        return config_actions
    
    return available_actions

# Get actions that actually have data
actions = get_available_actions()
print(f"Found data for these actions: {actions}")

def build_model(input_shape=(30, 1662), num_classes=len(actions)):
    """
    Build an enhanced LSTM model for action recognition.
    
    Args:
        input_shape: Shape of input sequences
        num_classes: Number of action classes
        
    Returns:
        keras.Model: Compiled model
    """
    model = Sequential()
    
    # First LSTM layer with more units
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    
    # Output layer - automatically adjusted to the number of classes
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model with lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_test, y_test, epochs=500, patience=100):
    """
    Train the model with longer training and reduced learning rate.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        keras.Model: Trained model
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Build model with the correct number of classes
    model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    
    # Display model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        TensorBoard(log_dir=LOG_DIR),
        EarlyStopping(
            monitor='val_categorical_accuracy', 
            patience=patience,
            restore_best_weights=True,
            mode='max'  # Stop when accuracy stops improving (not loss)
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce learning rate by half when plateauing
            patience=30,
            min_lr=0.00001
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False
        )
    ]
    
    # Train model
    print(f"\nTraining model for up to {epochs} epochs (with early stopping patience={patience})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,  # Smaller batch size for better generalization
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(MODEL_PATH)
    
    # Return the trained model and history
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and display detailed results.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert to class indices
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel accuracy: {accuracy * 100:.2f}%")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=actions))
    
    # Generate confusion matrix
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    
    # Display confusion matrix for each class
    for i, matrix in enumerate(conf_matrix):
        print(f"\nConfusion Matrix for '{actions[i]}':")
        print(matrix)
        
        # Calculate metrics for this class
        tn, fp, fn, tp = matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
    
    # Plot confusion matrices
    try:
        # Handle the case of a single action differently
        if len(actions) == 1:
            plt.figure(figsize=(6, 5))
            plt.matshow(conf_matrix[0], cmap='Blues')
            for (j, k), val in np.ndenumerate(conf_matrix[0]):
                plt.text(k, j, f'{val}', ha='center', va='center', 
                        color='white' if val > conf_matrix[0].max()/2 else 'black')
            plt.title(f"Confusion Matrix for '{actions[0]}'")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks([0, 1], ['Neg', 'Pos'])
            plt.yticks([0, 1], ['Neg', 'Pos'])
        else:
            # Create a visualization for multiple classes
            fig, axes = plt.subplots(1, len(actions), figsize=(5*len(actions), 5))
            
            for i, (matrix, ax) in enumerate(zip(conf_matrix, axes)):
                # Display the matrix values
                im = ax.matshow(matrix, cmap='Blues')
                
                # Add text annotations
                for (j, k), val in np.ndenumerate(matrix):
                    ax.text(k, j, f'{val}', ha='center', va='center', 
                            color='white' if val > matrix.max()/2 else 'black')
                
                # Set title and labels
                ax.set_title(f"'{actions[i]}'")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Neg', 'Pos'])
                ax.set_yticklabels(['Neg', 'Pos'])
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()
        print("\nConfusion matrices saved as 'confusion_matrices.png'")
    except Exception as e:
        print(f"Could not generate confusion matrix visualization: {e}")
    
    return accuracy, conf_matrix

def plot_training_history(history):
    """
    Plot training history with more details.
    
    Args:
        history: Model training history
    """
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved as 'training_history.png'")
    
    # If learning rate was tracked, plot it
    if 'lr' in history.history:
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('learning_rate.png')
        plt.close()
        print("Learning rate history saved as 'learning_rate.png'")

def main():
    """Main function to train and evaluate the model."""
    print("Sign Language Recognition - Enhanced Model Training")
    print("=================================================")
    
    # Check if we have actions to train on
    if not actions:
        print("Error: No actions found with data. Please run data_collection.py first.")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data for training. Please run data_collection.py first.")
        return
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Training for these actions: {actions}")
    
    # Ask about training parameters
    try:
        epochs = int(input("Enter maximum number of epochs [500]: ") or "500")
        patience = int(input("Enter early stopping patience (epochs without improvement) [100]: ") or "100")
    except ValueError:
        print("Invalid input. Using defaults: 500 epochs, patience=100")
        epochs = 500
        patience = 100
    
    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=epochs, patience=patience)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, conf_matrix = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nModel saved to {MODEL_PATH}")
    print("You can now run realtime_detection.py to test the model in real-time.")

if __name__ == "__main__":
    main()