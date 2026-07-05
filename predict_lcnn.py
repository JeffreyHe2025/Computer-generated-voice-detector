"""
Test a single audio file against the LCNN model with multi-feature input
(mel-spectrogram + MFCC concatenated on the frequency axis).

Usage:
    python predict_lcnn.py path/to/clip.wav
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
N_MFCC = 40
MAX_TIME_STEPS = 94
MODEL_PATH = "trained_voice_detector_lcnn.keras"


# Must match the MFM in train_lcnn.py exactly so the loader can reconstruct it
class MFM(layers.Layer):
    def call(self, inputs):
        n = inputs.shape[-1] // 2
        return tf.maximum(inputs[..., :n], inputs[..., n:])

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [input_shape[-1] // 2])


def fit_time(arr):
    if arr.shape[1] < MAX_TIME_STEPS:
        return np.pad(arr, ((0, 0), (0, MAX_TIME_STEPS - arr.shape[1])))
    return arr[:, :MAX_TIME_STEPS]


def extract_combined_features(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Match training: trim leading/trailing silence (clicks, room tone, breath).
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio_trimmed) > SAMPLE_RATE * 0.5:
        audio = audio_trimmed

    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE * DURATION]

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = fit_time(librosa.power_to_db(mel, ref=np.max))

    mfcc = fit_time(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC))
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    mfcc = mfcc * 15.0 - 40.0

    combined = np.concatenate([mel_db, mfcc], axis=0)
    return np.expand_dims(combined, axis=-1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_lcnn.py path/to/clip.wav")
        sys.exit(1)

    test_file = sys.argv[1]
    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Run train_lcnn.py first.")
        sys.exit(1)

    print(f"Loading LCNN model...")
    model = load_model(MODEL_PATH, custom_objects={"MFM": MFM})

    print(f"Analyzing '{test_file}'...")
    features = extract_combined_features(test_file)
    batch = np.expand_dims(features, axis=0)
    score = float(model.predict(batch, verbose=0)[0][0])

    print("-" * 40)
    print(f"Raw AI Score: {score:.4f}  (0 = Human, 1 = AI)")
    verdict = "AI / machine-generated voice" if score > 0.5 else "Human voice"
    confidence = abs(score - 0.5) * 2
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.1%}")
    print("-" * 40)
