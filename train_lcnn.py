"""
LCNN (Light CNN) trainer for human-vs-AI voice classification, with
multi-feature input: mel-spectrogram + MFCC concatenated along the
frequency axis.

Why both?
- Mel-spectrogram captures *which frequencies* are present.
- MFCC captures *the shape of the spectral envelope* — more compact,
  invariant to overall loudness, better at capturing vocoder artifacts.
Stacking them gives the CNN two complementary views.

Input shape changes from (128, 94, 1) to (168, 94, 1)
                          ^mel             ^mel + mfcc

NOTE: predict.py / v4.py expect (128, 94, 1) for the OLD model. To run
inference with this LCNN model, you must use the matching feature
extractor — copy `extract_combined_features` into predict.py and update
its input shape, or write a dedicated `predict_lcnn.py`.
"""

import os
import glob
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split

# ---------- Audio config ----------
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
N_MFCC = 40
COMBINED_FREQ_BINS = N_MELS + N_MFCC  # 168
MAX_TIME_STEPS = 94

# ---------- Data paths ----------
HUMAN_DIR = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-old/filtered_human_clips"
PARLER_AI_DIR = "ai_clips"
MLAAD_AI_DIR = "mlaad_clips/fake/en"
MODEL_OUT = "trained_voice_detector_lcnn.keras"

# ---------- Sampling ----------
N_PER_CLASS = 8000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------- Max-Feature-Map activation (the LCNN trick) ----------
class MFM(layers.Layer):
    """Max-Feature-Map: split channels in half, take elementwise max.

    Input:  (..., 2 * C)
    Output: (..., C)
    """
    def call(self, inputs):
        n = inputs.shape[-1] // 2
        return tf.maximum(inputs[..., :n], inputs[..., n:])

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [input_shape[-1] // 2])


def conv_mfm(x, filters, kernel_size, name):
    """Conv2D producing 2*filters channels, then MFM down to `filters`."""
    x = layers.Conv2D(2 * filters, kernel_size, padding='same', name=f"{name}_conv")(x)
    return MFM(name=f"{name}_mfm")(x)


# ---------- LCNN-9 architecture ----------
def build_lcnn(input_shape):
    inp = Input(shape=input_shape, name="mel_mfcc_input")

    x = conv_mfm(inp, 32, (5, 5), name="block1")
    x = layers.MaxPool2D((2, 2), name="pool1")(x)

    x = conv_mfm(x,    32, (1, 1), name="block2_1x1")
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = conv_mfm(x,    48, (3, 3), name="block2_3x3")
    x = layers.MaxPool2D((2, 2), name="pool2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)

    x = conv_mfm(x,    48, (1, 1), name="block3_1x1")
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = conv_mfm(x,    64, (3, 3), name="block3_3x3")
    x = layers.MaxPool2D((2, 2), name="pool3")(x)

    x = conv_mfm(x,    64, (1, 1), name="block4_1x1")
    x = layers.BatchNormalization(name="block4_bn")(x)
    x = conv_mfm(x,    32, (3, 3), name="block4_3x3")
    x = layers.BatchNormalization(name="block4_bn2")(x)
    x = conv_mfm(x,    32, (1, 1), name="block5_1x1")
    x = layers.BatchNormalization(name="block5_bn")(x)
    x = conv_mfm(x,    32, (3, 3), name="block5_3x3")
    x = layers.MaxPool2D((2, 2), name="pool4")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(160, name="fc1")(x)
    x = MFM(name="fc1_mfm")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    out = layers.Dense(1, activation='sigmoid', name="out")(x)
    return models.Model(inp, out, name="LCNN9_MultiFeature")


# ---------- Multi-feature extraction ----------
def fit_time(arr):
    """Pad or truncate the time (last) axis of a 2D feature to MAX_TIME_STEPS."""
    if arr.shape[1] < MAX_TIME_STEPS:
        return np.pad(arr, ((0, 0), (0, MAX_TIME_STEPS - arr.shape[1])))
    return arr[:, :MAX_TIME_STEPS]


def extract_combined_features(file_path, augment=False):
    """Return (168, 94, 1) tensor: mel-spectrogram + MFCC concatenated on freq axis.

    Two preprocessing steps now applied to defeat the click/noise shortcut:
      1. Silence trim from both ends — removes mouse clicks and other
         non-speech artifacts that live in low-energy regions.
      2. (Train only, when augment=True) random noise injection — makes
         background noise non-discriminative across classes so the model
         can't use noise level as a shortcut to label clips human.
    """
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # 1. Trim leading/trailing silence (clicks, room tone, breath at boundaries).
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio_trimmed) > SAMPLE_RATE * 0.5:  # keep only if at least 0.5s remains
            audio = audio_trimmed

        # 2. Symmetric noise augmentation (applied to both classes during training)
        if augment and np.random.random() < 0.5:
            snr_db = np.random.uniform(15, 35)
            noise = np.random.randn(len(audio)).astype(np.float32)
            sig_p = float(np.mean(audio ** 2) + 1e-12)
            noise_p = sig_p / (10 ** (snr_db / 10))
            noise *= np.sqrt(noise_p / (np.mean(noise ** 2) + 1e-12))
            audio = audio + noise

        # Pad/truncate back to fixed length
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * DURATION]

        # Mel-spectrogram (dB scale, roughly [-80, 0])
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = fit_time(mel_db)

        # MFCC (varies in scale; normalize to roughly [-80, 0] for compatibility)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = fit_time(mfcc)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        mfcc = mfcc * 15.0 - 40.0

        # Concatenate on frequency axis: (128 + 40, 94) = (168, 94)
        combined = np.concatenate([mel_db, mfcc], axis=0)
        return np.expand_dims(combined, axis=-1)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None


def list_audio_files(root_dir):
    paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return paths


def load_paths_into_features(paths, label, augment=False):
    X, y = [], []
    for p in paths:
        feat = extract_combined_features(p, augment=augment)
        if feat is not None:
            X.append(feat)
            y.append(label)
    return X, y


if __name__ == "__main__":
    print("Discovering files...")
    human_paths = list_audio_files(HUMAN_DIR)
    parler_paths = list_audio_files(PARLER_AI_DIR)
    mlaad_paths = list_audio_files(MLAAD_AI_DIR)
    print(f"  humans:  {len(human_paths)}")
    print(f"  Parler:  {len(parler_paths)}")
    print(f"  MLAAD:   {len(mlaad_paths)}")

    half = N_PER_CLASS // 2
    random.shuffle(human_paths)
    random.shuffle(parler_paths)
    random.shuffle(mlaad_paths)
    sampled_humans = human_paths[:N_PER_CLASS]
    sampled_parler = parler_paths[:half]
    sampled_mlaad = mlaad_paths[:N_PER_CLASS - len(sampled_parler)]
    print(f"\nSampling: {len(sampled_humans)} humans, "
          f"{len(sampled_parler)} Parler, {len(sampled_mlaad)} MLAAD")

    print("\nExtracting mel-spectrogram + MFCC features with silence-trim "
          "and symmetric noise augmentation (this can take a while)...")
    Xh, yh = load_paths_into_features(sampled_humans, label=0, augment=True)
    print(f"  humans loaded: {len(Xh)}")
    Xp, yp = load_paths_into_features(sampled_parler, label=1, augment=True)
    print(f"  Parler loaded: {len(Xp)}")
    Xm, ym = load_paths_into_features(sampled_mlaad, label=1, augment=True)
    print(f"  MLAAD loaded:  {len(Xm)}")

    X = np.array(Xh + Xp + Xm)
    y = np.array(yh + yp + ym)
    print(f"\nTotal: {len(X)} clips  |  humans: {int((y==0).sum())}  |  AI: {int((y==1).sum())}")
    print(f"Feature shape per clip: {X.shape[1:]}")

    if len(X) == 0:
        raise SystemExit("No audio loaded. Check folder paths and contents.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y, shuffle=True,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_lcnn((COMBINED_FREQ_BINS, MAX_TIME_STEPS, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
    )

    model.save(MODEL_OUT)
    print(f"\nSaved {MODEL_OUT}")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
