"""
Convert trained_voice_detector.keras -> VoiceDetector.mlpackage for iOS.

Run this in TWO stages because the .keras file was saved by Keras 3
(uses 'batch_shape'), but coremltools needs the legacy Keras 2 API
(which expects 'batch_input_shape'). We extract weights under Keras 3,
then rebuild the architecture under Keras 2 and convert.
"""

import os
import sys

WEIGHTS_PATH = "voice_detector.weights.h5"
KERAS_PATH = "trained_voice_detector.keras"
OUTPUT_PATH = "VoiceDetector.mlpackage"
INPUT_SHAPE = (128, 94, 1)  # (n_mels, time_steps, channels) — matches trained model


def stage1_extract_weights():
    """Run with Keras 3 (no TF_USE_LEGACY_KERAS env var)."""
    from tensorflow.keras.models import load_model

    m = load_model(KERAS_PATH)
    m.save_weights(WEIGHTS_PATH)
    print(f"Input shape: {m.input_shape}")
    print(f"Saved weights to {WEIGHTS_PATH}")
    print("\nNow re-run this script with: STAGE=2 python createSwift.py")


def stage2_convert():
    """Run with Keras 2 legacy mode."""
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import tensorflow as tf
    from tensorflow.keras import layers, models
    import coremltools as ct

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

    seq = build_model(INPUT_SHAPE)
    seq.load_weights(WEIGHTS_PATH)

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="mel_spectrogram")
    outputs = seq(inputs)
    functional = tf.keras.Model(inputs=inputs, outputs=outputs)

    mlmodel = ct.convert(
        functional,
        source="tensorflow",
        inputs=[ct.TensorType(name="mel_spectrogram", shape=(1, *INPUT_SHAPE))],
        convert_to="mlprogram",
    )
    mlmodel.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    stage = os.environ.get("STAGE", "1")
    if stage == "1":
        stage1_extract_weights()
    elif stage == "2":
        stage2_convert()
    else:
        print(f"Unknown STAGE={stage}. Use STAGE=1 or STAGE=2.")
        sys.exit(1)
