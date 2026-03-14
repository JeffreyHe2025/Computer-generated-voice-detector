import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# --- 1. Audio Processing Parameters (Must match training exactly!) ---
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 128
MAX_TIME_STEPS = int((SAMPLE_RATE * DURATION) / 512) + 1

def extract_mel_spectrogram(file_path):
    """
    Loads an audio file and converts it into a Mel-spectrogram.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad with zeros if audio is shorter than DURATION
        if len(audio) < SAMPLE_RATE * DURATION:
            pad_length = (SAMPLE_RATE * DURATION) - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return np.expand_dims(mel_spec_db, axis=-1)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 2. Load the Saved Model & Predict ---
if __name__ == "__main__":
    
    # 1. Point this to the exact file you want to test
    test_file = "human voice.mp3" 
    
    # 2. Point this to your saved Keras file
    model_path = "trained_voice_detector.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find the saved model at '{model_path}'.")
    elif not os.path.exists(test_file):
        print(f"Error: Could not find the audio file at '{test_file}'.")
    else:
        print("Loading trained AI model...")
        # Load the saved brain
        model = load_model(model_path)
        
        print(f"Analyzing '{test_file}'...")
        features = extract_mel_spectrogram(test_file)
        
        if features is not None:
            # Expand dimensions to create a "batch" of 1 image
            features_batch = np.expand_dims(features, axis=0)
            
            # Run the prediction
            prediction = model.predict(features_batch, verbose=0)[0][0]
            
            # Print the results
            print("-" * 30)
            print(f"Raw AI Score: {prediction:.4f} (0 = Human, 1 = AI)")
            
            if prediction > 0.5:
                print("RESULT: 🚨 This sounds like an AI/Machine generated voice.")
            else:
                print("RESULT: 🟢 This sounds like a real Human voice.")
            print("-" * 30)