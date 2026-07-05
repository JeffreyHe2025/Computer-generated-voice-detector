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

# --- 3. Example Usage & Inference ---
if __name__ == "__main__":
    # The expected shape of our spectrograms
    input_shape = (N_MELS, MAX_TIME_STEPS, 1)
    
    # Build the model
    model = build_model(input_shape)
    model.summary()

    # --- NOTE ON TRAINING ---
    # To actually train this, you need folders of audio files:
    # X_train = []
    # y_train = []
    # For file in human_folder:
    #     X_train.append(extract_mel_spectrogram(file))
    #     y_train.append(0) # 0 for Human
    # For file in ai_folder:
    #     X_train.append(extract_mel_spectrogram(file))
    #     y_train.append(1) # 1 for AI
    # model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)
    # model.save('voice_classifier.h5')
    
    # --- INFERENCE (Testing a file) ---
    test_file = "./REAL/linus-original.wav" # Replace with your file path
    
    if os.path.exists(test_file):
        print(f"Analyzing {test_file}...")
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
        print("Test file not found. Place a 'test_audio.wav' file in the directory to test inference.")