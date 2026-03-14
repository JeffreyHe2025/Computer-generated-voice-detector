import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- 1. Audio Processing Parameters ---
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 128
MAX_TIME_STEPS = int((SAMPLE_RATE * DURATION) / 512) + 1 # 512 is default hop_length

def extract_mel_spectrogram(file_path):
    """
    Loads an audio file and converts it into a Mel-spectrogram.
    Pads or truncates the audio to ensure a consistent input shape.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad with zeros if audio is shorter than DURATION
        if len(audio) < SAMPLE_RATE * DURATION:
            pad_length = (SAMPLE_RATE * DURATION) - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        
        # Convert to decibels (log scale) which neural networks handle better
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Reshape for CNN input: (n_mels, time_steps, channels)
        return np.expand_dims(mel_spec_db, axis=-1)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 2. Define the Neural Network Architecture ---
def build_model(input_shape):
    """
    Builds a Convolutional Neural Network (CNN) for binary classification.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Helps prevent overfitting
        
        # Output Layer: Sigmoid for binary classification (0 = Human, 1 = AI)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- 3. Data Loading Function ---
def load_training_data(real_folder, fake_folder):
    """
    Loops through the folders, extracts spectrograms, and labels them.
    REAL = 0, FAKE = 1.
    """
    X = [] # This will hold the spectrogram images
    y = [] # This will hold the labels (0 or 1)
    
    # 1. Process REAL audio
    print(f"Loading REAL audio from '{real_folder}'...")
    if os.path.exists(real_folder):
        for filename in os.listdir(real_folder):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(real_folder, filename)
                features = extract_mel_spectrogram(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(0) # Label 0 for Human/Real
    else:
        print(f"Error: Could not find folder named '{real_folder}'")
                
    # 2. Process FAKE audio
    print(f"Loading FAKE audio from '{fake_folder}'...")
    if os.path.exists(fake_folder):
        for filename in os.listdir(fake_folder):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(fake_folder, filename)
                features = extract_mel_spectrogram(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(1) # Label 1 for AI/Fake
    else:
        print(f"Error: Could not find folder named '{fake_folder}'")
                    
    # Convert lists to NumPy arrays for the neural network
    return np.array(X), np.array(y)

# --- 4. Execution, Training & Inference ---
if __name__ == "__main__":
    # Define folder names (Make sure these exist in the same directory as this script)
    REAL_DIR = "REAL"
    FAKE_DIR = "FAKE"
    
    # Load and process all the audio files
    X, y = load_training_data(REAL_DIR, FAKE_DIR)
    
    # Check if data was actually loaded
    if len(X) == 0:
        print("No audio files were processed. Check your folder names and contents.")
    else:
        print(f"Successfully loaded {len(X)} total audio files.")
        
        # Split data: 80% for training, 20% for testing/validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build the model
        input_shape = (N_MELS, MAX_TIME_STEPS, 1)
        model = build_model(input_shape)
        model.summary()
        
        # --- THE ACTUAL TRAINING LOOP ---
        print("\nStarting neural network training...")
        history = model.fit(
            X_train, y_train, 
            epochs=15,          # How many times it loops through the whole dataset
            batch_size=16,      # How many files it looks at before updating its math
            validation_data=(X_val, y_val) # Tests itself on unseen data each epoch
        )
        
        # Save the trained brain so you don't have to train it again
        model.save('trained_voice_detector.keras')
        print("\nTraining complete! Model saved as 'trained_voice_detector.keras'.")
        
        # --- INFERENCE (Testing your specific file after training) ---
        test_file = "human voice.mp3" # Replace with your file path if it changes
        
        if os.path.exists(test_file):
            print(f"\nAnalyzing {test_file} using newly trained model...")
            features = extract_mel_spectrogram(test_file)
            
            if features is not None:
                # Expand dimensions to create a "batch" of 1
                features_batch = np.expand_dims(features, axis=0)
                
                # Get prediction
                prediction = model.predict(features_batch)[0][0]
                
                print(f"Raw prediction score: {prediction:.4f}")
                if prediction > 0.5:
                    print("Result: This sounds like AI/Machine generated voice.")
                else:
                    print("Result: This sounds like a Human voice.")
        else:
            print(f"Test file not found at {test_file}. Check the path.")