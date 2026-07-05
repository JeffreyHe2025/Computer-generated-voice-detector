"""
Test a single audio file against the trained voice detector.

Usage:
    python predict.py path/to/clip.wav
    python predict.py "ai_clips/custom2.wav"
    python predict.py /Users/jeffreyhe/Desktop/whatever.mp3
"""

import os
import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 94
MODEL_PATH = "trained_voice_detector_newest.keras"


def extract_mel_spectrogram(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    return np.expand_dims(mel_db, axis=-1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/clip.wav")
        sys.exit(1)

    test_file = sys.argv[1]

    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    print(f"Loading model...")
    model = load_model(MODEL_PATH)

    print(f"Analyzing '{test_file}'...")
    features = extract_mel_spectrogram(test_file)
    batch = np.expand_dims(features, axis=0)
    score = float(model.predict(batch, verbose=0)[0][0])

    print("-" * 40)
    print(f"Raw AI Score: {score:.4f}  (0 = Human, 1 = AI)")
    if score > 0.5:
        verdict = "AI / machine-generated voice"
    else:
        verdict = "Human voice"
    confidence = abs(score - 0.5) * 2
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.1%}")
    print("-" * 40)