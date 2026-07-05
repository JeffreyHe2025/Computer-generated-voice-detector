import os
import glob
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 94

HUMAN_DIR = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-old/filtered_human_clips"
PARLER_AI_DIR = "ai_clips"
MLAAD_AI_DIR = "mlaad_clips/fake/en"
MODEL_OUT = "trained_voice_detector.keras"

# Balance knobs. Final dataset has ~N_PER_CLASS human and ~N_PER_CLASS AI clips.
# AI side is split evenly between Parler and MLAAD.
N_PER_CLASS = 8000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def extract_mel_spectrogram(file_path):
    try:
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
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None


def list_audio_files(root_dir):
    """Return all .wav/.mp3/.flac files under root_dir, recursively."""
    paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return paths


def load_paths_into_features(paths, label):
    X, y = [], []
    for p in paths:
        feat = extract_mel_spectrogram(p)
        if feat is not None:
            X.append(feat)
            y.append(label)
    return X, y


def build_model(input_shape):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])


if __name__ == "__main__":
    # --- Discover available files in each source ---
    print("Discovering files...")
    human_paths = list_audio_files(HUMAN_DIR)
    parler_paths = list_audio_files(PARLER_AI_DIR)
    mlaad_paths = list_audio_files(MLAAD_AI_DIR)
    print(f"  humans:  {len(human_paths)} in {HUMAN_DIR}")
    print(f"  Parler:  {len(parler_paths)} in {PARLER_AI_DIR}")
    print(f"  MLAAD:   {len(mlaad_paths)} in {MLAAD_AI_DIR}")

    # --- Balanced sampling ---
    half = N_PER_CLASS // 2
    random.shuffle(human_paths)
    random.shuffle(parler_paths)
    random.shuffle(mlaad_paths)

    sampled_humans = human_paths[:N_PER_CLASS]
    sampled_parler = parler_paths[:half]
    sampled_mlaad = mlaad_paths[:N_PER_CLASS - len(sampled_parler)]
    print(f"\nSampling: {len(sampled_humans)} humans, "
          f"{len(sampled_parler)} Parler, {len(sampled_mlaad)} MLAAD")

    # --- Feature extraction ---
    print("\nExtracting mel-spectrograms (this can take a while)...")
    Xh, yh = load_paths_into_features(sampled_humans, label=0)
    print(f"  humans loaded: {len(Xh)}")
    Xp, yp = load_paths_into_features(sampled_parler, label=1)
    print(f"  Parler loaded: {len(Xp)}")
    Xm, ym = load_paths_into_features(sampled_mlaad, label=1)
    print(f"  MLAAD loaded:  {len(Xm)}")

    X = np.array(Xh + Xp + Xm)
    y = np.array(yh + yp + ym)
    print(f"\nTotal: {len(X)} clips  |  humans: {int((y==0).sum())}  |  AI: {int((y==1).sum())}")

    if len(X) == 0:
        raise SystemExit("No audio loaded. Check folder paths and contents.")

    # --- Train/test split (stratified to preserve class balance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
        shuffle=True,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # --- Build & train ---
    model = build_model((N_MELS, MAX_TIME_STEPS, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=16,
        validation_data=(X_test, y_test),
    )

    model.save(MODEL_OUT)
    print(f"\nSaved {MODEL_OUT}")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
