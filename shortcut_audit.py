"""
shortcut_audit.py — Detect whether trained_voice_detector_newest.keras is
relying on dataset shortcuts rather than genuine human-vs-AI cues.

Two independent probes:

  Part 1  (fast, no retraining)  — CODEC CONTROL
      Take AI clips the model correctly labels "AI", add MP3 compression
      (+ telephone band-limit, + mild noise) and measure how much the
      score moves toward "human". A big drop == the model keys on codec /
      recording artifacts, not synthesis.

  Part 2  (retrains a small model) — LEAVE-ONE-DOMAIN-OUT
      Train on Common Voice (human) + Parler (AI) ONLY, then test on the
      entirely-unseen MLAAD TTS systems. If in-domain accuracy is high but
      unseen-TTS accuracy collapses, the model is fingerprinting the
      training corpora instead of learning "AI-ness".

Usage:
    python shortcut_audit.py            # run both parts
    python shortcut_audit.py --part 1   # codec control only (fast)
    python shortcut_audit.py --part 2   # leave-one-domain-out only
"""

import os
import glob
import random
import argparse
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf

# ---- Config (mirrors train_nn.py) ---------------------------------------
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 94
SEED = 42

HUMAN_DIR = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-old/filtered_human_clips"
PARLER_DIR = "ai_clips"
MLAAD_DIR = "mlaad_clips/fake/en"
MODEL_PATH = "trained_voice_detector_newest.keras"

random.seed(SEED)
np.random.seed(SEED)


# ---- Feature extraction (identical math to train_nn.py) -----------------
def mel_from_audio(audio):
    """audio: 1-D float32 @ 16 kHz -> (128, 94, 1) log-mel, exactly as trained."""
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    else:
        audio = audio[: SAMPLE_RATE * DURATION]
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    return np.expand_dims(mel_db, axis=-1)


def mel_from_file(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    return mel_from_audio(audio)


def list_audio(root):
    paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return paths


# ---- Perturbations (via ffmpeg) -----------------------------------------
def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def perturb_mp3(audio, bitrate="32k"):
    """Re-encode audio through a low-bitrate MP3 and decode back."""
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "in.wav")
        mp3 = os.path.join(d, "c.mp3")
        out = os.path.join(d, "out.wav")
        sf.write(wav, audio, SAMPLE_RATE)
        _ffmpeg(["-i", wav, "-b:a", bitrate, mp3])
        _ffmpeg(["-i", mp3, "-ar", str(SAMPLE_RATE), out])
        y, _ = librosa.load(out, sr=SAMPLE_RATE)
        return y


def perturb_telephone(audio):
    """Downsample to 8 kHz (telephone band) and back to 16 kHz."""
    down = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=8000)
    return librosa.resample(down, orig_sr=8000, target_sr=SAMPLE_RATE)


def perturb_noise(audio, snr_db=20):
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = rms / (10 ** (snr_db / 20))
    return audio + np.random.normal(0, noise_rms, size=audio.shape).astype(np.float32)


# ---- Part 1: codec control ----------------------------------------------
def part1_codec_control(model, n=40):
    print("\n" + "=" * 64)
    print("PART 1 — CODEC CONTROL  (does MP3 compression fool the model?)")
    print("=" * 64)

    ai_files = glob.glob(os.path.join(PARLER_DIR, "*.wav")) + \
               list_audio(MLAAD_DIR)
    random.shuffle(ai_files)

    rows = []  # (baseline, mp3, phone, noise)
    used = 0
    for f in ai_files:
        if used >= n:
            break
        try:
            audio, _ = librosa.load(f, sr=SAMPLE_RATE, duration=DURATION)
            if len(audio) < 1600:
                continue
            variants = {
                "baseline": audio,
                "mp3_32k": perturb_mp3(audio, "32k"),
                "telephone_8k": perturb_telephone(audio),
                "noise_20dB": perturb_noise(audio, 20),
            }
            batch = np.stack([mel_from_audio(v) for v in variants.values()])
            scores = model.predict(batch, verbose=0).ravel()
            rows.append(scores)
            used += 1
        except Exception as e:
            continue

    rows = np.array(rows)  # (used, 4)
    labels = ["baseline", "mp3_32k", "telephone_8k", "noise_20dB"]
    print(f"\nAI clips tested: {rows.shape[0]}  (1.0 = AI, 0.0 = human)\n")
    base = rows[:, 0]
    print(f"  {'variant':<14} {'mean score':>11} {'% still AI':>11} {'mean drop':>10}")
    for i, lab in enumerate(labels):
        col = rows[:, i]
        pct_ai = 100.0 * (col > 0.5).mean()
        drop = (base - col).mean()
        print(f"  {lab:<14} {col.mean():>11.4f} {pct_ai:>10.1f}% {drop:>10.4f}")

    mp3_flip = 100.0 * ((base > 0.5) & (rows[:, 1] <= 0.5)).mean()
    print(f"\n  --> {mp3_flip:.1f}% of correctly-detected AI clips FLIP to 'human' "
          f"after MP3 compression alone.")
    if mp3_flip >= 15 or (base.mean() - rows[:, 1].mean()) > 0.15:
        print("  VERDICT: strong CODEC shortcut. The model treats MP3 artifacts "
              "as a 'human' cue.")
    else:
        print("  VERDICT: codec effect is mild; shortcut is elsewhere (domain).")


# ---- Part 2: leave-one-domain-out ---------------------------------------
def _load_features(paths, label, cap):
    random.shuffle(paths)
    X, y = [], []
    for p in paths[:cap]:
        try:
            X.append(mel_from_file(p))
            y.append(label)
        except Exception:
            pass
    return X, y


def part2_leave_one_domain_out(per_class=2500, epochs=12):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 64)
    print("PART 2 — LEAVE-ONE-DOMAIN-OUT  (train: CommonVoice+Parler, test: MLAAD)")
    print("=" * 64)

    human = list_audio(HUMAN_DIR)
    parler = glob.glob(os.path.join(PARLER_DIR, "*.wav"))

    print(f"\nLoading train features (~{per_class}/class)...")
    Xh, yh = _load_features(human, 0, per_class)
    Xp, yp = _load_features(parler, 1, per_class)

    # Hold out 15% of each TRAINING domain for an in-domain reference.
    def split(X, y, frac=0.15):
        k = int(len(X) * frac)
        return X[k:], y[k:], X[:k], y[:k]

    Xh_tr, yh_tr, Xh_te, yh_te = split(Xh, yh)
    Xp_tr, yp_tr, Xp_te, yp_te = split(Xp, yp)

    X_train = np.array(Xh_tr + Xp_tr)
    y_train = np.array(yh_tr + yp_tr)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    print(f"Train: {len(X_train)}  (human {len(yh_tr)}, Parler {len(yp_tr)})")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(N_MELS, MAX_TIME_STEPS, 1)),
        layers.MaxPooling2D((2, 2)), layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'), layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=2)

    def acc(X, y):
        if len(X) == 0:
            return float('nan')
        p = (model.predict(np.array(X), verbose=0).ravel() > 0.5).astype(int)
        return accuracy_score(y, p)

    print("\n---- IN-DOMAIN (held-out slices of the TRAINING corpora) ----")
    print(f"  Common Voice (human)   acc = {acc(Xh_te, yh_te):.3f}")
    print(f"  Parler (AI, seen)      acc = {acc(Xp_te, yp_te):.3f}")

    print("\n---- OUT-OF-DOMAIN (MLAAD systems NEVER seen in training; all AI) ----")
    systems = sorted(
        d for d in os.listdir(MLAAD_DIR)
        if os.path.isdir(os.path.join(MLAAD_DIR, d))
    )
    overall_hits, overall_tot = 0, 0
    for s in systems:
        files = list_audio(os.path.join(MLAAD_DIR, s))[:120]
        X, y = _load_features(files, 1, 120)
        a = acc(X, y)
        overall_hits += int(a * len(X)) if len(X) else 0
        overall_tot += len(X)
        print(f"  {s:<32} acc = {a:.3f}  (n={len(X)})")
    if overall_tot:
        print(f"\n  MLAAD overall (unseen TTS) acc = {overall_hits/overall_tot:.3f} "
              f"over {overall_tot} clips")
    print("\n  If in-domain acc is high but MLAAD acc is much lower, the model is "
          "fingerprinting\n  the training corpora, not detecting synthesis.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["1", "2", "all"], default="all")
    ap.add_argument("--n", type=int, default=40, help="clips for part 1")
    ap.add_argument("--per-class", type=int, default=2500, help="train clips/class part 2")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    if args.part in ("1", "all"):
        from tensorflow.keras.models import load_model
        print(f"Loading {MODEL_PATH} ...")
        model = load_model(MODEL_PATH)
        part1_codec_control(model, n=args.n)

    if args.part in ("2", "all"):
        part2_leave_one_domain_out(per_class=args.per_class, epochs=args.epochs)
