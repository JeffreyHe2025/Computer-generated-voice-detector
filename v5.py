import librosa
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



def extract_features(file_path, max_pad_len=400):
    try:
        # Load audio file (librosa automatically resamples and converts to mono)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate the MFCCs so every input to the network is the exact same size
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path} - {e}")
        return None

def prepare_dataset(human_dir, ai_dir):
    features = []
    labels = []
    
    
    print("Processing human audio...")
    for file in os.listdir(human_dir):
        if file.endswith('.mp3') or file.endswith('.wav'):
            data = extract_features(os.path.join(human_dir, file))
            if data is not None:
                features.append(data)
                labels.append(0)
                
    
    print("Processing AI audio...")
    for file in os.listdir(ai_dir):
        if file.endswith('.wav'):
            data = extract_features(os.path.join(ai_dir, file))
            if data is not None:
                features.append(data)
                labels.append(1)
                
    
    X = np.array(features)
    y = np.array(labels)
    
    
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test = prepare_dataset('', 'ai_clips')