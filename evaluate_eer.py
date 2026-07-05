"""
Compute Equal Error Rate (EER) for a trained voice-detector model on a
held-out test set drawn from the same data sources as train_lcnn.py.

Usage:
    python evaluate_eer.py                          # uses LCNN model
    python evaluate_eer.py --model cnn              # uses standard CNN model

Outputs:
    - Total clips evaluated and per-class counts
    - EER as a percentage
    - Threshold at which FAR == FRR (use this in predict_lcnn.py if you want
      the optimal threshold instead of 0.5)
    - Accuracy at the EER threshold AND at threshold 0.5 for comparison
"""

import argparse
import glob
import os
import random
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, accuracy_score

# ---------- Config (must match train_lcnn.py / train.py) ----------
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
N_MFCC = 40
MAX_TIME_STEPS = 94
RANDOM_SEED = 123  # different from train.py's 42, so test set differs from train set
N_PER_CLASS_EVAL = 2000  # how many clips per class to evaluate on

HUMAN_DIR = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-old/filtered_human_clips"
PARLER_AI_DIR = "ai_clips"
MLAAD_AI_DIR = "mlaad_clips/fake/en"

LCNN_MODEL = "trained_voice_detector_lcnn.keras"
CNN_MODEL = "trained_voice_detector.keras"


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


def extract_mel(file_path):
    """For the CNN model: 128-mel only, shape (128, 94, 1)."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio_trimmed) > SAMPLE_RATE * 0.5:
        audio = audio_trimmed
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE * DURATION]

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = fit_time(librosa.power_to_db(mel, ref=np.max))
    return np.expand_dims(mel_db, axis=-1)


def extract_combined(file_path):
    """For the LCNN model: mel + MFCC concatenated, shape (168, 94, 1)."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
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


def list_audio_files(root_dir):
    paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return paths


def equal_error_rate(y_true, y_scores):
    """Returns (eer_fraction, threshold)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lcnn", "cnn"], default="lcnn")
    args = parser.parse_args()

    if args.model == "lcnn":
        model_path = LCNN_MODEL
        extract_fn = extract_combined
        feature_shape = (N_MELS + N_MFCC, MAX_TIME_STEPS, 1)
    else:
        model_path = CNN_MODEL
        extract_fn = extract_mel
        feature_shape = (N_MELS, MAX_TIME_STEPS, 1)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading {args.model.upper()} model from {model_path} ...")
    if args.model == "lcnn":
        model = load_model(model_path, custom_objects={"MFM": MFM})
    else:
        model = load_model(model_path)

    print(f"Expected input shape: {feature_shape}")

    # Sample clips from each source
    random.seed(RANDOM_SEED)
    human_paths = list_audio_files(HUMAN_DIR)
    parler_paths = list_audio_files(PARLER_AI_DIR)
    mlaad_paths = list_audio_files(MLAAD_AI_DIR)
    random.shuffle(human_paths)
    random.shuffle(parler_paths)
    random.shuffle(mlaad_paths)

    sampled_humans = human_paths[:N_PER_CLASS_EVAL]
    half = N_PER_CLASS_EVAL // 2
    sampled_parler = parler_paths[:half]
    sampled_mlaad = mlaad_paths[:N_PER_CLASS_EVAL - len(sampled_parler)]

    print(f"\nEvaluating on:")
    print(f"  humans: {len(sampled_humans)}")
    print(f"  Parler: {len(sampled_parler)}")
    print(f"  MLAAD:  {len(sampled_mlaad)}")

    # Extract features and predict
    all_paths = sampled_humans + sampled_parler + sampled_mlaad
    all_labels = [0] * len(sampled_humans) + [1] * (len(sampled_parler) + len(sampled_mlaad))

    print("\nExtracting features + predicting...")
    y_true, y_scores = [], []
    for i, (path, label) in enumerate(zip(all_paths, all_labels)):
        try:
            feat = extract_fn(path)
            score = float(model.predict(np.expand_dims(feat, 0), verbose=0)[0][0])
            y_true.append(label)
            y_scores.append(score)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_paths)} done")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Compute EER + accuracy at multiple thresholds
    eer, eer_threshold = equal_error_rate(y_true, y_scores)
    acc_at_0_5 = accuracy_score(y_true, (y_scores > 0.5).astype(int))
    acc_at_eer = accuracy_score(y_true, (y_scores > eer_threshold).astype(int))

    # Per-class breakdown
    human_scores = y_scores[y_true == 0]
    ai_scores = y_scores[y_true == 1]

    print("\n" + "=" * 50)
    print(f"Model: {args.model.upper()}")
    print(f"Total clips: {len(y_true)}")
    print(f"Mean score on humans: {human_scores.mean():.4f}  (lower is better)")
    print(f"Mean score on AI:     {ai_scores.mean():.4f}  (higher is better)")
    print("-" * 50)
    print(f"Equal Error Rate (EER):  {eer * 100:.2f}%")
    print(f"Optimal threshold:        {eer_threshold:.4f}")
    print("-" * 50)
    print(f"Accuracy @ threshold 0.5:           {acc_at_0_5 * 100:.2f}%")
    print(f"Accuracy @ EER threshold ({eer_threshold:.3f}): {acc_at_eer * 100:.2f}%")
    print("=" * 50)
